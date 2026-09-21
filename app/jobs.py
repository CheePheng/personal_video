"""Job store + runner for face swap renders.

Design notes (forced by constraints, not taste):

* A render takes minutes; Cloudflare's proxy read timeout is 125s, so the HTTP
  request that starts a job MUST return a job id immediately. Renders therefore
  run on a background thread and the browser polls for status.

* Progress is a real count of work done. The V2 pipeline reports
  ``frames_processed / total_frames`` directly, so the bar reflects frames
  actually encoded rather than anything scraped from a console.

* State lives in SQLite so a browser refresh -- or an app restart -- does not
  lose a running job.

* Two engines are selectable. ``v2`` is the current pipeline (tracking, masks,
  colour match, automatic benchmarking); ``v1`` is the earlier single-model
  renderer, kept as a regression baseline so the two can be compared on
  identical inputs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

DATA = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA / "jobs" / "jobs.db"

# Each stage's share of the overall bar. 'processing' dominates because it is
# the only stage whose cost scales with video length; the benchmark stages only
# appear in AUTO mode and are bounded by the sample count.
STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("preparing", 0.02),
    ("analysing video", 0.02),
    ("benchmarking swap models", 0.05),
    ("testing restoration", 0.04),
    ("selecting best pipeline", 0.01),
    ("detecting", 0.01),
    ("processing", 0.80),
    ("encoding", 0.03),
    ("restoring audio", 0.02),
]

_lock = threading.Lock()
_runners: dict[str, "JobRunner"] = {}


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id            TEXT PRIMARY KEY,
                status        TEXT NOT NULL,
                phase         TEXT,
                progress      REAL NOT NULL DEFAULT 0,
                message       TEXT,
                source_path   TEXT,
                target_path   TEXT,
                output_path   TEXT,
                options       TEXT,
                created_at    REAL NOT NULL,
                started_at    REAL,
                finished_at   REAL,
                error         TEXT,
                log_tail      TEXT
            )
            """
        )
        # A job left 'running' with no live thread means the app died mid-render.
        # Surface that rather than showing a bar frozen at 40% forever.
        c.execute(
            "UPDATE jobs SET status='interrupted', error='app restarted during render' "
            "WHERE status IN ('running','queued')"
        )


def create_job(source_path: str, target_path: str, output_path: str, options: str) -> str:
    jid = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id,status,progress,source_path,target_path,output_path,options,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (jid, "queued", 0.0, source_path, target_path, output_path, options, time.time()),
        )
    return jid


def get_job(jid: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    return dict(r) if r else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _conn() as c:
        rs = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rs]


def update(jid: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ",".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), jid))


def delete_job(jid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (jid,))


class JobRunner(threading.Thread):
    """Runs one render on a background thread, writing progress to SQLite."""

    def __init__(self, jid: str, sources: list[str], target: str, output: str,
                 opts: dict[str, Any]):
        super().__init__(daemon=True, name=f"job-{jid}")
        self.jid = jid
        self.sources = sources if isinstance(sources, list) else [sources]
        self.source = self.sources[0]
        self.target = target
        self.output = output
        self.opts = opts
        self.cancelled = False
        # The pipeline reports progress many times a second. Writing each one
        # would hammer SQLite and collide with the 1.5s status poller
        # ("database is locked"), so persist at most ~2/sec -- still finer than
        # the poller can display.
        self._last_write = 0.0

    def _overall(self, stage: str, pct: float) -> float:
        """Map a within-stage percentage onto the whole-render bar."""
        names = [s for s, _ in STAGE_WEIGHTS]
        if stage not in names:
            return pct
        idx = names.index(stage)
        base = sum(w for _, w in STAGE_WEIGHTS[:idx])
        return min(99.9, (base + STAGE_WEIGHTS[idx][1] * (pct / 100.0)) * 100.0)

    def _on_progress(self, stage: str, pct: float, extra: dict[str, Any]) -> None:
        now = time.time()
        frames, total = extra.get("frames"), extra.get("total")
        if frames and total:
            msg = f"frame {frames} of {total}"
            fps = extra.get("fps")
            if fps:
                msg += f"  -  {fps:.1f} fps"
                remain = (total - frames) / max(fps, 1e-6)
                if remain > 1:
                    msg += f"  -  ~{int(remain // 60)}m {int(remain % 60)}s left"
        elif extra.get("candidate"):
            msg = f"{extra['candidate']}  ({extra.get('index', '?')}/{extra.get('total', '?')})"
        elif extra.get("winner"):
            msg = f"selected: {extra['winner']}"
        else:
            msg = stage

        # Always persist a stage change; throttle mid-stage ticks.
        final = stage in ("complete", "encoding", "restoring audio",
                          "selecting best pipeline")
        if not final and (now - self._last_write) < 0.5:
            return
        self._last_write = now
        update(self.jid, progress=self._overall(stage, pct), phase=stage,
               message=msg[:300], log_tail=msg[:300])

    def cancel(self) -> None:
        self.cancelled = True

    def run(self) -> None:
        update(self.jid, status="running", started_at=time.time(), progress=0.0,
               phase="preparing", message="preparing")
        engine = self.opts.get("engine", "v2")
        try:
            result = self._run_v1() if engine == "v1" else self._run_v2()
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            return
        finally:
            with _lock:
                _runners.pop(self.jid, None)

        out = Path(self.output)
        if not out.exists() or out.stat().st_size == 0:
            update(self.jid, status="error", finished_at=time.time(),
                   error="render finished but produced no output file")
            return

        # Persist the full render record so the library can show exactly which
        # pipeline produced this file months later.
        opts = dict(self.opts)
        opts["result"] = result
        update(self.jid, status="done", progress=100.0, finished_at=time.time(),
               phase="complete", options=json.dumps(opts, default=str)[:60000],
               message=result.get("summary", "complete"))

    def _fail(self, e: Exception) -> None:
        from app.render.types import RenderError

        if self.cancelled or str(e) == "cancelled":
            update(self.jid, status="cancelled", finished_at=time.time())
            return
        if isinstance(e, RenderError):
            # A RenderError is an explained technical fault -- show it verbatim.
            update(self.jid, status="error", error=str(e)[:4000],
                   finished_at=time.time())
            return
        update(self.jid, status="error", finished_at=time.time(),
               error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"[:4000])

    # ------------------------------------------------------------- engines
    def _run_v2(self) -> dict[str, Any]:
        from app.render import benchmark, pipeline
        from app.render.types import PipelineConfig

        quality = self.opts.get("quality", "quality")
        opts = pipeline.RenderOptions(
            quality=quality,
            swap_all_faces=(self.opts.get("face_mode") == "many"),
        )

        identity = pipeline.load_source(self.sources)
        bench_report = None

        if quality == "auto":
            cfg, bench_report = benchmark.run(
                self.target, identity, opts, self.jid, self._on_progress)
            opts.config = cfg
        elif quality == "fast":
            # Preview mode: one proven model, no parsing/occlusion/colour work.
            opts.config = PipelineConfig(swapper="hyperswap_1a_256", enhancer=None,
                                         mask="model", color_match=False)
            opts.use_parsing = False
            opts.use_occlusion = False
        else:
            opts.config = PipelineConfig(swapper="hyperswap_1a_256",
                                         enhancer="gpen_bfr_512", enhancer_blend=0.7,
                                         mask="model")

        res = pipeline.render(self.sources, self.target, self.output, opts,
                              self._on_progress, lambda: self.cancelled, identity)
        d = res.as_dict()
        if bench_report:
            d["benchmark_winner"] = bench_report["winner"]["label"]
            d["benchmark_score"] = bench_report["winner"]["score"]
            d["benchmark_candidates"] = len(bench_report["candidates"])
        d["engine"] = "v2"
        d["summary"] = (f"complete - {res.frames} frames, {res.faces_swapped} faces, "
                        f"{res.config.describe() if res.config else ''}")
        return d

    def _run_v1(self) -> dict[str, Any]:
        """The V1 engine, kept as a regression baseline and fallback."""
        from app import swapper

        res = swapper.run_swap(
            source_image=self.source, target_video=self.target,
            output_path=self.output, on_progress=self._on_progress,
            swap_all_faces=(self.opts.get("face_mode") == "many"),
            quality="balanced", should_cancel=lambda: self.cancelled)
        res["engine"] = "v1"
        res["summary"] = (f"complete - {res['frames']} frames, "
                          f"{res['faces_swapped']} faces swapped (v1)")
        return res


def start(jid: str, sources: list[str], target: str, output: str,
          opts: dict[str, Any]) -> None:
    r = JobRunner(jid, sources, target, output, opts)
    with _lock:
        _runners[jid] = r
    r.start()


def cancel(jid: str) -> bool:
    with _lock:
        r = _runners.get(jid)
    if r:
        r.cancel()
        return True
    return False


def is_active(jid: str) -> bool:
    with _lock:
        return jid in _runners


def active_count() -> int:
    with _lock:
        return len(_runners)
