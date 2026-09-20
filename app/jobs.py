"""Job store + runner for face swap renders.

Design notes (forced by constraints, not taste):

* A render takes minutes; Cloudflare's proxy read timeout is 125s, so the HTTP
  request that starts a job MUST return a job id immediately. Renders therefore
  run on a background thread and the browser polls for status.

* Progress is a real count of work done. The swap pipeline reports
  ``frames_processed / total_frames`` directly, so the bar reflects frames
  actually encoded rather than anything scraped from a console.

* State lives in SQLite so a browser refresh -- or an app restart -- does not
  lose a running job.
"""

from __future__ import annotations

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
# the only stage whose cost scales with video length; the rest are near-fixed.
STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("preparing", 0.02),
    ("detecting", 0.03),
    ("processing", 0.90),
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

    def __init__(self, jid: str, source: str, target: str, output: str, opts: dict[str, Any]):
        super().__init__(daemon=True, name=f"job-{jid}")
        self.jid = jid
        self.source = source
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
        msg = f"{stage} - frame {frames} of {total}" if frames and total else stage

        # Always persist a stage change; throttle mid-stage ticks.
        final = stage in ("complete", "encoding", "restoring audio")
        if not final and (now - self._last_write) < 0.5:
            return
        self._last_write = now
        update(self.jid, progress=self._overall(stage, pct), phase=stage,
               message=msg[:300], log_tail=msg[:300])

    def cancel(self) -> None:
        self.cancelled = True

    def run(self) -> None:
        # Imported here, not at module import: loading onnxruntime pulls in the
        # whole CUDA stack, which should not happen just because the web app
        # started.
        from app import swapper

        update(self.jid, status="running", started_at=time.time(), progress=0.0,
               phase="preparing", message="preparing")
        try:
            result = swapper.run_swap(
                source_image=self.source,
                target_video=self.target,
                output_path=self.output,
                on_progress=self._on_progress,
                swap_all_faces=(self.opts.get("face_mode") == "many"),
                quality=self.opts.get("quality", "balanced"),
                should_cancel=lambda: self.cancelled,
            )
        except swapper.SwapError as e:
            if self.cancelled or str(e) == "cancelled":
                update(self.jid, status="cancelled", finished_at=time.time())
            else:
                # A SwapError is a real, explainable fault -- show it verbatim.
                update(self.jid, status="error", error=str(e)[:4000],
                       finished_at=time.time())
            return
        except Exception as e:  # noqa: BLE001
            update(self.jid, status="error", finished_at=time.time(),
                   error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"[:4000])
            return
        finally:
            with _lock:
                _runners.pop(self.jid, None)

        out = Path(self.output)
        if not out.exists() or out.stat().st_size == 0:
            update(self.jid, status="error", finished_at=time.time(),
                   error="render finished but produced no output file")
            return

        update(self.jid, status="done", progress=100.0, finished_at=time.time(),
               phase="complete",
               message=f"complete - {result['frames']} frames, "
                       f"{result['faces_swapped']} faces swapped")


def start(jid: str, source: str, target: str, output: str, opts: dict[str, Any]) -> None:
    r = JobRunner(jid, source, target, output, opts)
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
