"""Phase 3: find and classify the frames where identity collapses.

The A->B baseline is ArcFace 0.6262 mean but 0.3331 at its worst frame, and
SFace 0.5922 / 0.2334. A system that looks right 95% of the time and falls
apart for the rest reads as broken, because the eye lands on the bad frames.
Averages hide exactly that, so this reports the distribution and attributes
each bad frame to a cause rather than optimising the mean further.

Causes are measured, not guessed. For every frame it records the conditions
that plausibly explain a collapse -- pose, blur, mouth opening, face scale,
brightness, detector confidence, alignment movement, mask coverage -- then
correlates them against identity so the ranking reflects this footage rather
than an assumption about what "should" be hard.

Run:  python scripts/worst_frames.py [swapper] [enhancer] [blend]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
SOURCE = AB / "_source_a.png"
OUT = ROOT / "data" / "benchmarks" / "ab_tournament"
DUMP = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/worst")

CLIPS = ["A_single_frontal", "B_profile", "G_talking", "I_closeup", "N_dark",
         "O_bright", "D_motion_blur", "E_glasses", "F_occlusion",
         "H_small_face", "P_1080p", "Q_4k"]

# Below this a frame is not recognisably the source any more. Two strangers
# score ~0.02 and the pipeline averages 0.63, so 0.45 sits well clear of both.
COLLAPSE = 0.45


def conditions(frame: np.ndarray, face, mask_cov: float,
               prev_kps: Optional[np.ndarray]) -> dict[str, float]:
    from app.render import alignment

    yaw, roll = alignment.pose_from_kps(face.kps)
    x1, y1, x2, y2 = [int(v) for v in face.box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else np.zeros((1, 1), np.uint8)

    # Eye/mouth landmarks: 0,1 eyes, 2 nose, 3,4 mouth corners.
    eye_d = float(np.linalg.norm(face.kps[1] - face.kps[0])) or 1.0
    mouth_w = float(np.linalg.norm(face.kps[4] - face.kps[3]))
    nose_mouth = float(np.linalg.norm(
        face.kps[2] - (face.kps[3] + face.kps[4]) / 2.0))

    return {
        "yaw_abs": abs(float(yaw)),
        "roll_abs": abs(float(roll)),
        "blur": float(cv2.Laplacian(grey, cv2.CV_64F).var()) if grey.size > 4 else 0.0,
        "face_px": float(face.size),
        "brightness": float(grey.mean()) if grey.size else 0.0,
        "contrast": float(grey.std()) if grey.size else 0.0,
        "detector_conf": float(face.score),
        "mouth_open": nose_mouth / eye_d,
        "mouth_width": mouth_w / eye_d,
        "mask_coverage": float(mask_cov),
        "kps_motion": (float(np.linalg.norm(face.kps - prev_kps, axis=1).mean()) / eye_d
                       if prev_kps is not None else 0.0),
    }


def main() -> int:
    swapper = sys.argv[1] if len(sys.argv) > 1 else "hyperswap_1a_256"
    enhancer = sys.argv[2] if len(sys.argv) > 2 else "gpen_bfr_512"
    blend = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    if enhancer in ("none", "None", ""):
        enhancer, blend = None, 0.0

    from app.render import detection, judges, pipeline, recognition
    from app.render.pipeline import FrameRenderer, RenderOptions
    from app.render.types import PipelineConfig, RenderError
    import app.render.video as V

    identity = pipeline.load_source([str(SOURCE)])
    src = cv2.imread(str(SOURCE))
    sf = detection.detect_robust(src, 0.3)
    src_sface = judges.embed(src, sf[0].kps) if sf and judges.available() else None

    cfg = PipelineConfig(swapper=swapper, enhancer=enhancer,
                         enhancer_blend=blend, mask="full")
    opts = RenderOptions(quality="quality", config=cfg)
    print("worst-frame analysis: %s" % cfg.describe())

    rows: list[dict[str, Any]] = []
    for stem in CLIPS:
        p = AB / (stem + ".mp4")
        if not p.is_file():
            continue
        info = V.probe(str(p))
        n = info.total_frames or 0
        # Every frame of the short clips; a stride on the long/large ones so
        # 4K does not dominate the runtime.
        step = 1 if n <= 80 else max(1, n // 60)
        idx = list(range(0, n, step))
        got = V.sample_frames(str(p), info, idx)
        diag = float(np.hypot(info.width, info.height))
        renderer = FrameRenderer(cfg, opts, identity, diag)
        prev_kps = None

        for i, frame in sorted(got.items()):
            # detect_with_fallback, NOT detect_robust: production goes
            # through the SCRFD recall fallback, and measuring the primary
            # alone reports frames as lost that ship perfectly well. That
            # mistake made an occlusion sequence look like a pipeline defect
            # when the shipping path already rescues all four frames.
            faces = detection.detect_with_fallback(
                frame, opts.detect_threshold, "quality",
                opts.detector, opts.detector_fallback)
            if not faces:
                rows.append({"clip": stem, "frame": i, "identity": None,
                             "note": "no face detected in INPUT"})
                continue
            face = max(faces, key=lambda f: f.area)
            try:
                out, full_mask = renderer.render_face(frame, face)
            except RenderError as e:
                rows.append({"clip": stem, "frame": i, "identity": None,
                             "note": "render error: %s" % str(e)[:80]})
                continue

            cov = float((full_mask > 0.5).mean())
            cond = conditions(frame, face, cov, prev_kps)
            prev_kps = face.kps

            of_list = detection.detect_robust(out, 0.3, "quality")
            if not of_list:
                rows.append({"clip": stem, "frame": i, "identity": None,
                             "note": "no face detected in OUTPUT", **cond})
                continue
            of = max(of_list, key=lambda f: f.area)
            try:
                arc = recognition.similarity(
                    recognition.embed(out, of.kps), identity.embedding)
            except RenderError:
                continue
            sfc = None
            if src_sface is not None:
                try:
                    sfc = judges.similarity(src_sface, judges.embed(out, of.kps))
                except RenderError:
                    pass
            rows.append({"clip": stem, "frame": i, "identity": round(arc, 4),
                         "sface": None if sfc is None else round(sfc, 4), **cond})
        print("  %-18s %d frames" % (stem, len(got)), flush=True)

    scored = [r for r in rows if r.get("identity") is not None]
    ident = np.array([r["identity"] for r in scored])
    print("\n%d frames scored" % len(scored))
    for q in (1, 5, 10, 25, 50):
        print("  p%-3d %.4f" % (q, float(np.percentile(ident, q))))
    print("  mean %.4f   worst %.4f" % (ident.mean(), ident.min()))

    bad = [r for r in scored if r["identity"] < COLLAPSE]
    print("\n%d frames below %.2f (%.1f%%)"
          % (len(bad), COLLAPSE, 100.0 * len(bad) / max(len(scored), 1)))

    from collections import Counter
    print("\nby clip:")
    for clip, c in Counter(r["clip"] for r in bad).most_common():
        tot = sum(1 for r in scored if r["clip"] == clip)
        print("  %-18s %3d / %3d  (%.0f%%)" % (clip, c, tot, 100.0 * c / max(tot, 1)))

    # Which condition actually separates bad frames from good ones?
    keys = ["yaw_abs", "roll_abs", "blur", "face_px", "brightness", "contrast",
            "detector_conf", "mouth_open", "mouth_width", "mask_coverage",
            "kps_motion"]
    good = [r for r in scored if r["identity"] >= COLLAPSE]
    print("\ncondition        bad mean     good mean    corr(identity)")
    corrs = []
    for k in keys:
        b = [r[k] for r in bad if k in r]
        g = [r[k] for r in good if k in r]
        v = [r[k] for r in scored if k in r]
        iv = [r["identity"] for r in scored if k in r]
        c = float(np.corrcoef(v, iv)[0, 1]) if len(v) > 3 and np.std(v) > 1e-9 else 0.0
        corrs.append((abs(c), k, c))
        print("  %-14s %10s %13s %15.3f"
              % (k, round(float(np.mean(b)), 3) if b else "-",
                 round(float(np.mean(g)), 3) if g else "-", c))
    corrs.sort(reverse=True)
    print("\nstrongest predictors of identity (|corr| desc):")
    for a, k, c in corrs[:5]:
        print("  %-14s %+.3f  (%s identity as %s rises)"
              % (k, c, "higher" if c > 0 else "lower", k))

    worst = sorted(scored, key=lambda r: r["identity"])[:15]
    print("\n15 worst frames:")
    for r in worst:
        print("  %-16s f%-5d arc %.4f  yaw %5.1f  blur %7.1f  face %5.1f  "
              "mouth %.2f  cov %.2f"
              % (r["clip"], r["frame"], r["identity"], r.get("yaw_abs", 0),
                 r.get("blur", 0), r.get("face_px", 0),
                 r.get("mouth_open", 0), r.get("mask_coverage", 0)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "worst_frames.json").write_text(
        json.dumps({"config": cfg.describe(), "rows": rows}, indent=2,
                   default=str), encoding="utf-8")
    print("\nwritten: %s" % (OUT / "worst_frames.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
