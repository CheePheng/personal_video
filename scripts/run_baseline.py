"""Run ONE clean-baseline case in its own process.

Each case gets a fresh interpreter so its peak VRAM is attributable to that
case alone -- if every case shared a process, cached sessions from earlier
cases would inflate the peak of every later one.

Run:  python scripts/run_baseline.py <case> <results.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.clean_baseline import (  # noqa: E402
    OUT, PERF, CLIPS, SRC, banner, idle_floor, run_case)


def build(case: str):
    from app.render.pipeline import RenderOptions
    from app.render.types import PipelineConfig

    HS = "hyperswap_1a_256"

    # (clip, options, note)
    table = {
        "quality_720p": (
            PERF / "perf_720p.mp4",
            RenderOptions(quality="quality"),
            "default Quality pipeline, 1280x720"),
        "quality_1080p": (
            PERF / "perf_1080p.mp4",
            RenderOptions(quality="quality"),
            "default Quality pipeline, 1920x1080"),
        "quality_4k": (
            PERF / "perf_4k.mp4",
            RenderOptions(quality="quality"),
            "default Quality pipeline, 3840x2160"),
        "automax_cold": (
            CLIPS / "G_talking.mp4",
            RenderOptions(quality="auto"),
            "Auto Max with the benchmark cache cleared"),
        "automax_cached": (
            CLIPS / "G_talking.mp4",
            RenderOptions(quality="auto"),
            "Auto Max reusing the cached verdict"),
        "boost_256": (
            PERF / "perf_1080p.mp4",
            RenderOptions(quality="quality",
                          config=PipelineConfig(swapper=HS, pixel_boost=0)),
            "HyperSwap 1a native 256"),
        "boost_768": (
            PERF / "perf_1080p.mp4",
            RenderOptions(quality="quality",
                          config=PipelineConfig(swapper=HS, pixel_boost=768)),
            "HyperSwap 1a, pixel boost 768"),
        "boost_1024": (
            PERF / "perf_1080p.mp4",
            RenderOptions(quality="quality",
                          config=PipelineConfig(swapper=HS, pixel_boost=1024)),
            "HyperSwap 1a, pixel boost 1024"),
        "restore_heavy": (
            PERF / "perf_1080p.mp4",
            RenderOptions(quality="quality",
                          config=PipelineConfig(
                              swapper=HS, enhancer="restoreformer_plus_plus",
                              enhancer_blend=0.7, mask="parsing",
                              pixel_boost=1024, color_match=True)),
            "heaviest pipeline: boost 1024 + RestoreFormer++ 70% + parsing mask"),
    }
    if case not in table:
        raise SystemExit("unknown case %r; known: %s"
                         % (case, ", ".join(sorted(table))))
    return table[case]


def main() -> None:
    case = sys.argv[1]
    results = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "results.json"

    if case == "automax_cold":
        from app.render import benchmark
        n = 0
        if benchmark.CACHE_DIR.is_dir():
            for p in benchmark.CACHE_DIR.glob("*.json"):
                p.unlink()
                n += 1
        print("  (cleared %d cached benchmark verdict(s))" % n, flush=True)

    clip, opts, note = build(case)
    if not clip.is_file():
        raise SystemExit("missing clip: %s" % clip)

    # Measure the idle floor BEFORE anything of ours loads. Taking it after
    # the Auto Max benchmark made the "floor" include ~11 GB of benchmark
    # sessions, so peak-minus-floor came out as 252 MB for a phase that
    # actually peaked at 12.3 GB.
    floor = idle_floor(3.0)

    if case.startswith("automax"):
        # Auto Max lives in the job layer, not in render(): it has to
        # benchmark before it can name a winner. Driving it the way the app
        # does is the only way to measure it -- passing quality='auto'
        # straight to render() is now refused rather than silently defaulted.
        import time as _t
        from app.render import benchmark, pipeline as _p
        identity = _p.load_source([str(SRC)])
        t0 = _t.time()
        cfg, report = benchmark.run(str(clip), identity, opts, "baseline", None)
        bench_s = _t.time() - t0
        opts.config = cfg
        print("  benchmark: %.1fs -> %s" % (bench_s, cfg.describe()), flush=True)

    row = run_case(case, clip, opts, note)
    banner(row, floor)
    row["idle_floor_mb"] = floor

    results.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if results.is_file():
        try:
            existing = json.loads(results.read_text(encoding="utf-8"))
        except ValueError:
            existing = []
    existing = [r for r in existing if r.get("label") != case]
    existing.append(row)
    results.write_text(json.dumps(existing, indent=2, default=str),
                       encoding="utf-8")


if __name__ == "__main__":
    main()
