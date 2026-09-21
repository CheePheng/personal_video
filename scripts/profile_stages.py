"""Per-stage timing breakdown of the render loop.

The clean hardware baseline showed the GPU sitting 86-94% idle during a
render, which means the interesting question is no longer "is the model too
heavy" but "where does the wall-clock actually go". This answers that by
wrapping the functions the frame loop calls, rather than editing the loop --
production code stays untouched and the measured path is the real one.

Buckets are attributed to the caller, so nested work is not double counted:
``swap`` excludes nothing, but ``mask``/``colour``/``paste`` are separate
because they are separate calls from ``FrameRenderer.render_face``.

Run:  python scripts/profile_stages.py <clip.mp4> [label] [results.json]
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.clean_baseline import GpuSampler, OUT, SRC, idle_floor  # noqa: E402

TIMES: dict[str, float] = defaultdict(float)
COUNTS: dict[str, int] = defaultdict(int)

# Buckets nested INSIDE another bucket. They are reported for detail but
# excluded from the "unattributed" arithmetic, which would otherwise double
# count them and drive the remainder negative.
NESTED: set[str] = set()


class _Timer:
    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        TIMES[self.name] += time.perf_counter() - self.t0
        COUNTS[self.name] += 1


def _wrap(mod: Any, attr: str, bucket: str, nested: bool = False) -> None:
    real = getattr(mod, attr)
    if nested:
        NESTED.add(bucket)

    def wrapper(*a, **k):
        with _Timer(bucket):
            return real(*a, **k)

    wrapper.__name__ = getattr(real, "__name__", attr)
    setattr(mod, attr, wrapper)


class _TimedStdin:
    """Proxy for the encoder's stdin so the blocking write is measured."""

    def __init__(self, real):
        self._real = real

    def write(self, data):
        with _Timer("encode_write"):
            return self._real.write(data)

    def close(self):
        return self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


def install() -> None:
    from app.render import (alignment, color, detection, masking, metrics,
                            recognition, restoration, sessions, swapping,
                            video)

    _wrap(detection, "detect_with_fallback", "detect")
    _wrap(recognition, "embed", "recognise")
    _wrap(swapping, "swap", "swap_infer")
    _wrap(masking, "build", "mask")
    # Nested inside "mask" -- reported for detail, excluded from the totals.
    _wrap(masking, "parsing_masks", "  .mask/bisenet", nested=True)
    _wrap(masking, "occlusion_mask", "  .mask/xseg", nested=True)
    _wrap(sessions, "run", "  .onnx_run(all)", nested=True)
    _wrap(color, "match", "colour")          # legacy one-shot path
    _wrap(color, "estimate", "colour_estimate")
    _wrap(color, "apply", "colour_apply")
    _wrap(alignment, "paste_back", "paste")
    _wrap(alignment, "warp", "warp")
    _wrap(restoration, "restore", "restore")

    # Scene-cut detection is a per-frame CPU histogram pass.
    real_cut = metrics.SceneCutDetector.__call__

    def timed_cut(self, prev, cur):
        with _Timer("scene_cut"):
            return real_cut(self, prev, cur)

    metrics.SceneCutDetector.__call__ = timed_cut  # type: ignore[method-assign]

    # Decode: time spent blocked reading raw frames from the ffmpeg pipe.
    real_frames = video.frames

    def timed_frames(proc, info):
        gen = real_frames(proc, info)
        while True:
            with _Timer("decode_read"):
                try:
                    f = next(gen)
                except StopIteration:
                    return
            yield f

    video.frames = timed_frames  # type: ignore[assignment]

    # Encoder: the blocking pipe write, plus the full-frame copy feeding it.
    real_open = video.open_encoder

    def timed_open(out_path, info, quality="quality"):
        proc, name = real_open(out_path, info, quality)
        proc.stdin = _TimedStdin(proc.stdin)  # type: ignore[assignment]
        return proc, name

    video.open_encoder = timed_open  # type: ignore[assignment]

    # np.ascontiguousarray(...).tobytes() is a full-frame copy per frame.
    real_ascontig = np.ascontiguousarray

    def timed_ascontig(a, *args, **kw):
        with _Timer("frame_copy"):
            return real_ascontig(a, *args, **kw)

    np.ascontiguousarray = timed_ascontig  # type: ignore[assignment]


def main() -> None:
    clip = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else clip.stem
    results = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT / "profile.json"

    install()
    from app.render import pipeline
    from app.render.pipeline import RenderOptions

    OUT.mkdir(parents=True, exist_ok=True)
    floor = idle_floor(3.0)

    with GpuSampler(200) as smp:
        t0 = time.perf_counter()
        res = pipeline.render([str(SRC)], str(clip),
                              str(OUT / ("prof_" + label + ".mp4")),
                              RenderOptions(quality="quality"))
        wall = time.perf_counter() - t0
    gpu = smp.stats()

    frames = int(res.frames or 0)
    measured = sum(v for k, v in TIMES.items() if k not in NESTED)

    # Frame-loop stages only: setup (model load, benchmark) is excluded from
    # the per-frame view but reported separately so nothing is hidden.
    rows = []
    for k in sorted(TIMES, key=lambda x: -TIMES[x]):
        rows.append({
            "stage": k,
            "total_s": round(TIMES[k], 3),
            "ms_per_frame": round(TIMES[k] * 1000.0 / frames, 2) if frames else None,
            "calls": COUNTS[k],
            "pct_wall": round(TIMES[k] / wall * 100.0, 1) if wall else None,
        })
    other = wall - measured
    rows.append({"stage": "unattributed (setup, loop, progress)",
                 "total_s": round(other, 3),
                 "ms_per_frame": round(other * 1000.0 / frames, 2) if frames else None,
                 "calls": 0,
                 "pct_wall": round(other / wall * 100.0, 1) if wall else None})

    print("\n%s  --  %d frames, %.1fs wall, %.1f ms/frame, GPU util %.1f%%"
          % (label, frames, wall, wall * 1000.0 / max(frames, 1),
             gpu.get("util_mean_pct") or 0.0))
    print("  %-38s %9s %12s %7s" % ("stage", "total s", "ms/frame", "% wall"))
    for r in rows:
        print("  %-38s %9s %12s %7s"
              % (r["stage"], r["total_s"], r["ms_per_frame"], r["pct_wall"]))

    payload = {
        "label": label, "clip": clip.name, "frames": frames,
        "wall_s": round(wall, 3),
        "ms_per_frame": round(wall * 1000.0 / frames, 2) if frames else None,
        "idle_floor_mb": floor,
        "peak_card_mb": gpu.get("peak_card_mb"),
        "peak_attrib_mb": (gpu.get("peak_card_mb") - floor)
        if gpu.get("peak_card_mb") else None,
        "util_mean_pct": gpu.get("util_mean_pct"),
        "util_max_pct": gpu.get("util_max_pct"),
        "stages": rows,
    }
    existing = []
    if results.is_file():
        try:
            existing = json.loads(results.read_text(encoding="utf-8"))
        except ValueError:
            existing = []
    existing = [r for r in existing if r.get("label") != label]
    existing.append(payload)
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
