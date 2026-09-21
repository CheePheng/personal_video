"""Does source VIDEO help, and does POSE-AWARE retrieval help further?

Two separate questions, deliberately not merged:

  1. Does adding source-video observations improve the source identity
     representation at all?
  2. Given those observations, does choosing them per target pose beat one
     global vector?

The previous experiment answered neither, because its five references
spanned -17 to +6 degrees. This uses a real single-person interview reel
with 51 degrees of genuine yaw span, so a pose-aware method has something
to choose between.

Strategies, all conditioning the SAME renderer on a different 512-d vector:

  A  1 best photo
  B  5 photos, unweighted mean
  C  5 photos, quality-weighted        <- current production
  D  photos + video, one global quality-weighted vector
  E  photos + video, pose-aware per-frame fusion
  F  E plus temporal smoothing

Results are reported BY TARGET POSE, because a global mean hides exactly
the frames a pose-aware method is supposed to rescue.

Run:  python scripts/test_source_video.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
FIX = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/fixtures")
WSRC = FIX / "wsrc"
VIDEO = FIX / "src_video_w.mp4"

# Targets: person B clips. Includes the ones with real yaw movement.
CLIPS = ["A_single_frontal", "B_profile", "C_fast_motion", "G_talking",
         "I_closeup", "N_dark", "O_bright", "D_motion_blur", "P_1080p"]

POSE_SIGMA = 18.0
VECTOR_EMA = 0.75


def pose_weights(ref_yaws: np.ndarray, quals: np.ndarray,
                 yaw: float) -> np.ndarray:
    d = np.abs(ref_yaws - yaw)
    w = np.exp(-(d ** 2) / (2.0 * POSE_SIGMA ** 2)) * quals
    s = float(w.sum())
    if s < 1e-8:
        w = quals.copy()
        s = float(w.sum()) or 1.0
    return w / s


def main() -> int:
    from app.render import (alignment, detection, judges, pipeline,
                            recognition, source_bank as SB, swapping)
    from app.render.pipeline import FrameRenderer, RenderOptions, apply_preset
    import app.render.video as V

    photos = sorted(WSRC.glob("w_photo*.png"))
    if not photos or not VIDEO.is_file():
        raise SystemExit("source fixtures missing; run the ingestion step first")

    rec_p = SB.build(photos, None, use_cache=True)
    rec_pv = SB.build(photos, VIDEO, use_cache=True)
    refs_p = SB.refs_from_record(rec_p)
    refs_pv = SB.refs_from_record(rec_pv)
    print("photos only : %d refs, coverage %s"
          % (len(refs_p), rec_p["coverage"]["yaw_span"]))
    print("photos+video: %d refs, coverage %s, bins %d/9"
          % (len(refs_pv), rec_pv["coverage"]["yaw_span"],
             rec_pv["coverage"]["bins_covered"]))

    # The identity we are trying to reproduce is person W. Anchor on the
    # quality-weighted photo consensus, which is what the user supplied.
    pv = np.stack([r.embedding for r in refs_p])
    pq = np.array([r.quality for r in refs_p], np.float32)
    anchor = (pv * pq[:, None]).sum(0) / max(float(pq.sum()), 1e-6)
    anchor_n = anchor / (np.linalg.norm(anchor) or 1.0)

    # SFace holdout on the best photo, so the judge never sees the fused
    # vector the selector was built from.
    best_photo = max(refs_p, key=lambda r: r.quality)
    bp_img = cv2.imread(str(WSRC / best_photo.origin))
    bpf = detection.detect_robust(bp_img, 0.3)
    ja = judges.embed(bp_img, max(bpf, key=lambda x: x.area).kps) if bpf else None

    vecs_pv = np.stack([r.embedding for r in refs_pv])
    q_pv = np.array([r.quality for r in refs_pv], np.float32)
    yaws_pv = np.array([r.yaw for r in refs_pv], np.float32)

    strategies: dict[str, Any] = {
        "A 1 photo": ("static", best_photo.embedding),
        "B photos mean": ("static", pv.mean(0)),
        "C photos qual-w": ("static", anchor),
        "D +video global": ("static", (vecs_pv * q_pv[:, None]).sum(0) / max(float(q_pv.sum()), 1e-6)),
        "E +video pose-aware": ("pose", None),
        "F pose-aware+smooth": ("smooth", None),
    }

    cfg = apply_preset(RenderOptions(quality="quality")).config
    print("\npipeline: %s" % cfg.describe())
    print("\n%-22s %8s %8s %8s %8s %8s" %
          ("strategy", "arcMean", "arcP5", "arcWorst", "sfMean", "sfP5"))
    print("-" * 66)

    results: dict[str, dict[str, Any]] = {}
    for label, (mode, static_vec) in strategies.items():
        arc: list[float] = []
        sfc: list[float] = []
        by_pose: dict[str, list[float]] = {}
        ema: Optional[np.ndarray] = None

        for clip in CLIPS:
            p = AB / (clip + ".mp4")
            if not p.is_file():
                continue
            info = V.probe(str(p))
            idx = [int(i * (info.total_frames - 1) / 9) for i in range(10)]
            o = RenderOptions(quality="quality", config=cfg)
            o.use_parsing = False
            o.use_occlusion = False
            r = FrameRenderer(cfg, o, pipeline.SourceIdentity(
                embedding=anchor_n.reshape(1, -1)),
                float(np.hypot(info.width, info.height)))
            ema = None
            for _, fr in sorted(V.sample_frames(str(p), info, idx).items()):
                fs = detection.detect_with_fallback(fr, 0.5, "quality",
                                                    "yoloface_8n", "scrfd_2.5g")
                if not fs:
                    continue
                f = max(fs, key=lambda x: x.area)
                tyaw, _tp, _tr = alignment.pose_3d(f.kps, fr.shape)

                if mode == "static":
                    vec = static_vec
                else:
                    w = pose_weights(yaws_pv, q_pv, tyaw)
                    vec = (vecs_pv * w[:, None]).sum(0)
                    if mode == "smooth":
                        ema = vec if ema is None else (
                            VECTOR_EMA * ema + (1 - VECTOR_EMA) * vec)
                        vec = ema

                r.embedding = swapping.prepare_embedding(cfg.swapper, vec)
                out, _ = r.render_face(fr, f)
                os_ = detection.detect_robust(out, 0.3, "quality")
                if not os_:
                    continue
                of = max(os_, key=lambda x: x.area)
                a = recognition.similarity(recognition.embed(out, of.kps), anchor_n)
                arc.append(a)
                bucket = ("frontal" if abs(tyaw) < 8 else
                          "15-30" if abs(tyaw) < 30 else "30+")
                by_pose.setdefault(bucket, []).append(a)
                if ja is not None:
                    sfc.append(judges.similarity(ja, judges.embed(out, of.kps)))

        aa = np.array(arc)
        ss = np.array(sfc) if sfc else np.array([np.nan])
        results[label] = {"arc": aa, "sface": ss, "by_pose": by_pose}
        print("%-22s %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (label, aa.mean(), np.percentile(aa, 5), aa.min(),
                 np.nanmean(ss), np.nanpercentile(ss, 5)))

    print("\nBY TARGET POSE (ArcFace mean):")
    buckets = ["frontal", "15-30", "30+"]
    present = [b for b in buckets
               if any(b in r["by_pose"] for r in results.values())]
    print("%-22s %s" % ("strategy", "  ".join("%-12s" % b for b in present)))
    for label, r in results.items():
        cells = []
        for b in present:
            v = r["by_pose"].get(b)
            cells.append("%-12s" % ("%.4f (%d)" % (np.mean(v), len(v)) if v else "-"))
        print("%-22s %s" % (label, "  ".join(cells)))

    base = results["C photos qual-w"]["arc"].mean()
    print("\nvs production C (%.4f):" % base)
    for label, r in results.items():
        if label.startswith("C"):
            continue
        print("  %-22s %+.4f" % (label, r["arc"].mean() - base))

    d = results["D +video global"]["arc"].mean()
    e = results["E +video pose-aware"]["arc"].mean()
    print("\nQ1 does source VIDEO help?      D - C = %+.4f" % (d - base))
    print("Q2 does POSE-AWARE help more?   E - D = %+.4f" % (e - d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
