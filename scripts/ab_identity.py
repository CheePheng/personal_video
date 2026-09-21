"""Identity measurement on the A -> B set, where source and subject differ.

Every identity number this project has quoted came from clips containing the
source's own face, so "0.97 similarity" largely meant the pipeline did not
wreck a face that already matched. This measures the thing that was actually
being claimed: how much of person A's identity survives onto person B.

Two recognisers, reported separately and never mixed:

  ArcFace  the SELECTOR. The swap is conditioned on it, so it is inside the
           loop and cannot be the only witness.
  SFace    the HOLDOUT. Independently trained, never used for selection.

Run:  python scripts/ab_identity.py [boost|restore]
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

AB = ROOT / "data" / "testclips" / "ab"
SOURCE = AB / "_source_a.png"
OUT = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/ab")

# Single-person clips only: this measures identity transfer, not tracking.
CLIPS = ["A_single_frontal", "B_profile", "G_talking", "I_closeup",
         "N_dark", "D_motion_blur", "E_glasses", "P_1080p"]


def score(out_path: Path, src_arc, src_sface, n: int = 10) -> dict[str, Any]:
    from app.render import detection, judges, recognition
    import app.render.video as V

    info = V.probe(str(out_path))
    total = info.total_frames or 0
    if total <= 0:
        return {}
    idx = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    got = V.sample_frames(str(out_path), info, idx)
    arc: list[float] = []
    sfc: list[float] = []
    for _, frame in sorted(got.items()):
        faces = detection.detect_robust(frame, 0.4, "quality")
        if not faces:
            continue
        f = max(faces, key=lambda x: x.area)
        arc.append(recognition.similarity(src_arc, recognition.embed(frame, f.kps)))
        if src_sface is not None:
            sfc.append(judges.similarity(src_sface, judges.embed(frame, f.kps)))
    return {"arc": arc, "sface": sfc}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "boost"
    from app.render import detection, judges, pipeline, recognition
    from app.render.pipeline import RenderOptions
    from app.render.types import PipelineConfig

    if not SOURCE.is_file():
        raise SystemExit("no A->B fixtures; run scripts/make_ab_fixtures.py first")

    OUT.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(SOURCE))
    sf = detection.detect_robust(src, 0.3)
    if not sf:
        raise SystemExit("no face in the source identity")
    src_arc = recognition.embed(src, sf[0].kps)
    src_sface = judges.embed(src, sf[0].kps) if judges.available() else None

    HS = "hyperswap_1a_256"
    if mode == "boost":
        variants = [
            ("256", PipelineConfig(swapper=HS, pixel_boost=0, mask="full")),
            ("768", PipelineConfig(swapper=HS, pixel_boost=768, mask="full")),
            ("1024", PipelineConfig(swapper=HS, pixel_boost=1024, mask="full")),
        ]
    else:
        variants = [
            ("none", PipelineConfig(swapper=HS, mask="full")),
            ("gpen@40", PipelineConfig(swapper=HS, enhancer="gpen_bfr_512",
                                       enhancer_blend=0.4, mask="full")),
            ("gpen@70", PipelineConfig(swapper=HS, enhancer="gpen_bfr_512",
                                       enhancer_blend=0.7, mask="full")),
            ("codeformer@70", PipelineConfig(swapper=HS, enhancer="codeformer",
                                             enhancer_blend=0.7, mask="full")),
            ("restoreformer@70", PipelineConfig(
                swapper=HS, enhancer="restoreformer_plus_plus",
                enhancer_blend=0.7, mask="full")),
        ]

    rows: list[dict[str, Any]] = []
    for label, cfg in variants:
        arc_all: list[float] = []
        sfc_all: list[float] = []
        t0 = time.time()
        frames = 0
        for clip in CLIPS:
            p = AB / (clip + ".mp4")
            if not p.is_file():
                continue
            dst = OUT / ("%s__%s.mp4" % (clip, label.replace("@", "")))
            opts = RenderOptions(quality="quality", config=cfg)
            try:
                res = pipeline.render([str(SOURCE)], str(p), str(dst), opts)
            except Exception as e:  # noqa: BLE001
                print("  %-16s %-18s FAILED %s" % (label, clip, str(e)[:60]))
                continue
            frames += int(res.frames or 0)
            s = score(dst, src_arc, src_sface)
            arc_all += s.get("arc", [])
            sfc_all += s.get("sface", [])
        dt = time.time() - t0
        row = {
            "variant": label,
            "n_samples": len(arc_all),
            "arc_mean": round(float(np.mean(arc_all)), 4) if arc_all else None,
            "arc_worst": round(float(np.min(arc_all)), 4) if arc_all else None,
            "sface_mean": round(float(np.mean(sfc_all)), 4) if sfc_all else None,
            "sface_worst": round(float(np.min(sfc_all)), 4) if sfc_all else None,
            "frames": frames,
            "ms_per_frame": round(dt * 1000.0 / frames, 1) if frames else None,
        }
        rows.append(row)
        print("  %-18s n=%-3d  ArcFace %s (worst %s)   SFace %s (worst %s)   %s ms/f"
              % (label, row["n_samples"], row["arc_mean"], row["arc_worst"],
                 row["sface_mean"], row["sface_worst"], row["ms_per_frame"]),
              flush=True)

    (OUT / ("ab_%s.json" % mode)).write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    print("\nA->B identity, source is NOT the person in the clip.")
    if rows and rows[0]["arc_mean"] is not None:
        base = rows[0]
        print("Relative to %s:" % base["variant"])
        for r in rows[1:]:
            if r["arc_mean"] is None:
                continue
            print("  %-18s ArcFace %+.4f   SFace %+.4f"
                  % (r["variant"], r["arc_mean"] - base["arc_mean"],
                     (r["sface_mean"] - base["sface_mean"])
                     if r["sface_mean"] and base["sface_mean"] else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
