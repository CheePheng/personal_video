"""Three-way comparison: V1, V2 (fixed Quality), V2.1 (Auto Max).

V2.1 has to clear two bars, not one:
  * it must not regress against the V2 baseline on the important metrics
  * Auto Max must justify its extra cost by choosing better than a fixed preset

Measurement is shared with the V1-vs-V2 harness so all three engines are
scored by identical code on identical clips -- and, importantly, by code that
scores the SWAPPED face rather than the largest one, which is what made the
first V2 comparison misleading.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v1_vs_v2 import CLIPS, contact_sheet, measure, render_v1  # noqa: E402

OUT = ROOT / "data" / "benchmarks" / "v1_v2_v21"

CLIP_SET = ["A_single_frontal", "K_other_larger", "J_two_crossing",
            "F_occlusion", "I_closeup", "N_dark", "B_profile", "G_talking"]


def _peak_vram() -> int:
    from app.render import sessions
    return int(sessions.gpu_info().get("vram_used_mb") or 0)


def render_v2_fixed(source: str, target: str, out: str) -> dict[str, Any]:
    """The V2 baseline: fixed hyperswap_1a + GPEN@70."""
    from app.render import pipeline
    from app.render.types import PipelineConfig
    opts = pipeline.RenderOptions(quality="quality")
    opts.config = PipelineConfig(swapper="hyperswap_1a_256", enhancer="gpen_bfr_512",
                                 enhancer_blend=0.7, mask="model")
    t0 = time.time()
    res = pipeline.render([source], target, out, opts)
    d = res.as_dict()
    d["wall_s"] = round(time.time() - t0, 1)
    d["vram_mb"] = _peak_vram()
    return d


def render_v21_auto(source: str, target: str, out: str, job: str) -> dict[str, Any]:
    """V2.1: Auto Max benchmarks candidates and renders with the winner."""
    from app.render import benchmark, pipeline
    opts = pipeline.RenderOptions(quality="auto")
    identity = pipeline.load_source([source])
    t0 = time.time()
    cfg, report = benchmark.run(target, identity, opts, job)
    bench_s = time.time() - t0
    opts.config = cfg
    res = pipeline.render([source], target, out, opts, identity=identity)
    d = res.as_dict()
    d["wall_s"] = round(time.time() - t0, 1)
    d["benchmark_s"] = round(bench_s, 1)
    d["winner"] = report["winner"]["label"]
    d["from_cache"] = report.get("from_cache", False)
    d["candidates"] = len(report["candidates"])
    d["vram_mb"] = _peak_vram()
    return d


def main(argv: list[str]) -> int:
    from app.render import detection, recognition

    clips = argv[1:] or CLIP_SET
    OUT.mkdir(parents=True, exist_ok=True)

    source = str(CLIPS / "_face_a.png")
    src_img = cv2.imread(source)
    sf = detection.detect_robust(src_img, 0.3)
    if not sf:
        print("no face in source image")
        return 2
    src_emb = recognition.embed(src_img, max(sf, key=lambda f: f.area).kps)

    rows: list[dict[str, Any]] = []
    for name in clips:
        clip = CLIPS / f"{name}.mp4"
        if not clip.exists():
            print(f"  skip {name}: no clip")
            continue
        print(f"\n=== {name} ===", flush=True)
        row: dict[str, Any] = {"clip": name}

        for tag, fn in (("v1", lambda o: render_v1(source, str(clip), o)),
                        ("v2", lambda o: render_v2_fixed(source, str(clip), o)),
                        ("v21", lambda o: render_v21_auto(source, str(clip), o,
                                                          f"cmp_{name}"))):
            out = str(OUT / f"{name}__{tag}.mp4")
            row[f"{tag}_output"] = out
            try:
                r = fn(out)
                row[f"{tag}_wall_s"] = r.get("wall_s")
                row[f"{tag}_vram_mb"] = r.get("vram_mb")
                if tag == "v21":
                    row["v21_winner"] = r.get("winner")
                    row["v21_benchmark_s"] = r.get("benchmark_s")
                    row["v21_cached"] = r.get("from_cache")
                row[tag] = measure(out, src_emb, str(clip))
                extra = f" ({r.get('winner')})" if tag == "v21" else ""
                print(f"  {tag:4} {r.get('wall_s')}s{extra}")
            except Exception as e:  # noqa: BLE001
                row[f"{tag}_error"] = f"{type(e).__name__}: {e}"[:200]
                print(f"  {tag:4} FAILED: {row[f'{tag}_error']}")
        rows.append(row)

    # ---- table
    keys = [("identity_mean", "identity", 1),
            ("identity_min", "identity worst", 1),
            ("identity_switches", "id switches", -1),
            ("identity_stability_mean", "temporal stab", 1),
            ("flow_flicker_mean", "flicker", -1),
            ("expression_delta_mean", "expr drift", -1),
            ("sharpness_mean", "sharpness", 1)]

    hdr = f"\n{'clip':<18}{'metric':<16}{'V1':>10}{'V2':>10}{'V2.1':>10}  best"
    print(hdr); print("-" * (len(hdr) + 4))
    tally = {"V1": 0, "V2": 0, "V2.1": 0, "tie": 0}
    v21_vs_v2 = {"better": 0, "worse": 0, "same": 0}

    for r in rows:
        a, b, c = r.get("v1"), r.get("v2"), r.get("v21")
        if not (a and b and c):
            continue
        for key, label, direction in keys:
            va, vb, vc = a.get(key), b.get(key), c.get(key)
            if None in (va, vb, vc):
                continue
            vals = {"V1": va, "V2": vb, "V2.1": vc}
            best = (max(vals, key=lambda k: vals[k]) if direction > 0
                    else min(vals, key=lambda k: vals[k]))
            if len({round(v, 6) for v in vals.values()}) == 1:
                best = "tie"
            tally[best] += 1
            if abs(vc - vb) < 1e-9:
                v21_vs_v2["same"] += 1
            elif (vc > vb) == (direction > 0):
                v21_vs_v2["better"] += 1
            else:
                v21_vs_v2["worse"] += 1
            print(f"{r['clip']:<18}{label:<16}{va:>10.4f}{vb:>10.4f}{vc:>10.4f}  {best}")

    print("-" * (len(hdr) + 4))
    print(f"  best-of-three  ->  V1: {tally['V1']}   V2: {tally['V2']}   "
          f"V2.1: {tally['V2.1']}   tie: {tally['tie']}")
    print(f"  V2.1 vs V2     ->  better: {v21_vs_v2['better']}   "
          f"worse: {v21_vs_v2['worse']}   same: {v21_vs_v2['same']}")

    print(f"\n{'clip':<18}{'V2.1 winning pipeline':<44}{'bench s':>9}{'cached':>8}")
    for r in rows:
        if r.get("v21_winner"):
            print(f"{r['clip']:<18}{r['v21_winner'][:44]:<44}"
                  f"{r.get('v21_benchmark_s', 0):>9}{str(r.get('v21_cached')):>8}")

    print(f"\n{'clip':<18}{'V1 s':>8}{'V2 s':>8}{'V2.1 s':>9}{'V2.1 VRAM':>11}")
    for r in rows:
        print(f"{r['clip']:<18}{r.get('v1_wall_s', 0):>8}{r.get('v2_wall_s', 0):>8}"
              f"{r.get('v21_wall_s', 0):>9}{r.get('v21_vram_mb', 0):>11}")

    (OUT / "three_way.json").write_text(
        json.dumps({"rows": rows, "tally": tally, "v21_vs_v2": v21_vs_v2},
                   indent=2, default=str), encoding="utf-8")

    # Contact sheet uses the V1/V2 columns plus V2.1 in place of V2's slot.
    sheet_rows = [{"clip": r["clip"], "v1_output": r.get("v1_output", ""),
                   "v2_output": r.get("v21_output", "")} for r in rows]
    try:
        contact_sheet(sheet_rows, OUT / "contact_sheet_v21.jpg")
    except Exception as e:  # noqa: BLE001
        print(f"  contact sheet skipped: {type(e).__name__}")
    print(f"\n  wrote {OUT / 'three_way.json'}")

    # V2.1 passes only if it does not regress against the V2 baseline.
    ok = v21_vs_v2["worse"] <= v21_vs_v2["better"]
    print(f"\n  VERDICT: V2.1 {'PASSES' if ok else 'REGRESSES'} against the V2 baseline")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
