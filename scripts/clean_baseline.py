"""Clean-machine hardware baseline for the RTX 5070 Ti.

Every previous performance number in this project was measured while other
GPU-heavy programs were running (a game, browsers, and -- as it turned out --
two stale copies of the app server holding 2.2 GB between them). That matters
more than it sounds, because ``sessions.gpu_info()`` reports
``nvidia-smi --query-gpu=memory.used``, which is **whole-card** usage. Every
"peak VRAM" figure we ever quoted therefore included whatever else happened to
be resident. This script re-measures with the machine dedicated to rendering
and reports two separate numbers:

    peak_card_mb    whole-card peak, directly comparable to the old figures
    peak_attrib_mb  peak minus the idle floor -- what the renderer actually costs

It changes no production code. It only runs the existing pipeline and watches.

Run:  python scripts/clean_baseline.py [all|perf|boost|auto|restore]
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRATCH = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad")
PERF = SCRATCH / "perf"
OUT = SCRATCH / "baseline"
CLIPS = ROOT / "data" / "testclips"
SRC = CLIPS / "_face_a.png"


# ------------------------------------------------------------------ sampler
class GpuSampler:
    """Streams nvidia-smi from ONE long-lived process.

    Spawning nvidia-smi per sample was measured to cost real time during a
    previous benchmark (hundreds of process launches per run), so this uses
    the built-in ``-lms`` loop and reads its stdout instead.
    """

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.rows: list[tuple[float, float, float, float]] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._stop.is_set():
                break
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                self.rows.append((float(parts[0]), float(parts[1]),
                                  float(parts[2]), float(parts[3])))
            except ValueError:
                continue

    def __enter__(self) -> "GpuSampler":
        self._proc = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits", "-lms=%d" % self.interval_ms],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)

    def stats(self) -> dict[str, Any]:
        if not self.rows:
            return {"samples": 0}
        mem = [r[0] for r in self.rows]
        util = [r[1] for r in self.rows]
        pwr = [r[2] for r in self.rows]
        tmp = [r[3] for r in self.rows]
        return {
            "samples": len(self.rows),
            "peak_card_mb": int(max(mem)),
            "mean_card_mb": int(sum(mem) / len(mem)),
            "util_mean_pct": round(sum(util) / len(util), 1),
            "util_max_pct": int(max(util)),
            "power_mean_w": round(sum(pwr) / len(pwr), 1),
            "power_max_w": round(max(pwr), 1),
            "temp_max_c": int(max(tmp)),
        }


def idle_floor(seconds: float = 4.0) -> int:
    """Whole-card VRAM with nothing of ours running."""
    with GpuSampler(200) as s:
        time.sleep(seconds)
    st = s.stats()
    return int(st.get("mean_card_mb") or 0)


# ------------------------------------------------------------------ identity
def identity_scores(out_path: Path, n: int = 8) -> dict[str, Any]:
    """Score the rendered output against the source, in-loop AND holdout.

    ArcFace is the recogniser the swap is conditioned on, so it is reported
    as the selector metric. SFace is independently trained and never
    participates in selection; it is the honest second opinion.
    """
    from app.render import detection, judges, recognition
    import app.render.video as V

    src = cv2.imread(str(SRC))
    sf = detection.detect_robust(src, 0.3)
    if not sf:
        return {"error": "no face in source photo"}
    src_arc = recognition.embed(src, sf[0].kps)
    src_sface = judges.embed(src, sf[0].kps) if judges.available() else None

    info = V.probe(str(out_path))
    total = info.total_frames or 0
    if total <= 0:
        return {"error": "empty output"}
    idx = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    got = V.sample_frames(str(out_path), info, idx)

    arc: list[float] = []
    sfc: list[float] = []
    for _, frame in sorted(got.items()):
        faces = detection.detect_robust(frame, 0.4, "quality")
        if not faces:
            continue
        face = max(faces, key=lambda f: f.area)
        arc.append(recognition.similarity(src_arc,
                                          recognition.embed(frame, face.kps)))
        if src_sface is not None:
            sfc.append(judges.similarity(src_sface, judges.embed(frame, face.kps)))

    def agg(v: list[float]) -> dict[str, Any]:
        if not v:
            return {"mean": None, "worst": None, "n": 0}
        return {"mean": round(float(np.mean(v)), 4),
                "worst": round(float(np.min(v)), 4), "n": len(v)}

    return {"arcface_selector": agg(arc), "sface_holdout": agg(sfc)}


# ------------------------------------------------------------------ runner
def run_case(label: str, clip: Path, opts: Any, note: str = "",
             score: bool = True) -> dict[str, Any]:
    from app.render import pipeline, sessions

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / (label + ".mp4")

    # Count eviction: benchmark.py releases sessions to manage VRAM, and a
    # release mid-render means a model had to be rebuilt. Wrap rather than
    # edit production code.
    evictions: dict[str, Any] = {"n": 0, "names": []}
    real_release = sessions.release

    def counting_release(name: Optional[str] = None) -> None:
        evictions["n"] += 1
        evictions["names"].append(name or "ALL")
        return real_release(name)

    sessions.release = counting_release  # type: ignore[assignment]
    loaded_before = sessions.loaded()

    try:
        with GpuSampler(200) as smp:
            t0 = time.time()
            res = pipeline.render([str(SRC)], str(clip), str(out_path), opts)
            wall = time.time() - t0
        gpu = smp.stats()
    finally:
        sessions.release = real_release  # type: ignore[assignment]

    frames = int(res.frames or 0)
    row: dict[str, Any] = {
        "label": label,
        "note": note,
        "clip": clip.name,
        "frames": frames,
        "wall_s": round(wall, 2),
        "ms_per_frame": round(wall * 1000.0 / frames, 1) if frames else None,
        "fps": round(frames / wall, 2) if wall > 0 else None,
        "config": res.config.describe() if res.config else None,
        "cuda_active": bool(res.gpu.get("cuda_active")),
        "session_providers": res.gpu.get("session_providers"),
        "evictions": evictions["n"],
        "eviction_targets": evictions["names"][:8],
        "sessions_before": len(loaded_before),
        "sessions_after": len(sessions.loaded()),
        "encoder": res.encoder,
        "timings": {k: round(v, 2) for k, v in (res.timings or {}).items()},
        "fallbacks": res.fallbacks,
    }
    row.update(gpu)
    if score:
        row["identity"] = identity_scores(out_path)
    return row


def banner(row: dict[str, Any], floor_mb: int) -> None:
    peak = row.get("peak_card_mb")
    attrib = (peak - floor_mb) if isinstance(peak, int) else None
    row["peak_attrib_mb"] = attrib
    ident = row.get("identity") or {}
    arc = (ident.get("arcface_selector") or {}).get("mean")
    sfc = (ident.get("sface_holdout") or {}).get("mean")
    print("  %-26s %8s ms/f %7s fps  peak %s MB (attrib %s MB)  "
          "util %s%%/%s%%  evict %s  arc %s sface %s"
          % (row["label"], row["ms_per_frame"], row["fps"], peak, attrib,
             row.get("util_mean_pct"), row.get("util_max_pct"),
             row["evictions"], arc, sfc), flush=True)


def save(rows: list[dict[str, Any]], floor_mb: int, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (name + ".json")
    p.write_text(json.dumps(
        {"idle_floor_mb": floor_mb, "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    return p
