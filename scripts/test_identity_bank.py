"""Does a pose-aware source identity bank beat the current photo fusion?

The question the architecture raises is attractive: index source references
by measured pose and, for each target frame, condition on the references
nearest that pose instead of one all-purpose average. The current fusion
actively DOWN-weights profiles (exp(-|yaw|/45)), which is exactly backwards
when the target is in profile.

This measures it rather than assuming it. Strategies compared, on identical
target footage with one source identity:

  A  single best photo (highest quality score)
  B  unweighted mean of all photos
  C  quality-weighted fusion            <- current production
  F  pose-aware bank, per-frame vector
  G  pose-aware bank + temporal smoothing

Reported per target-pose bucket, because a pose-aware method can only help
where the target actually varies in pose -- a gain hidden inside a global
mean would be the wrong way to read this.

Run:  python scripts/test_identity_bank.py
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
EXTRA = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/fixtures")

CLIPS = ["A_single_frontal", "B_profile", "G_talking", "I_closeup",
         "N_dark", "O_bright", "D_motion_blur", "P_1080p"]


def load_sources() -> list[tuple[str, np.ndarray, Any]]:
    from app.render import detection, recognition
    anchor = cv2.imread(str(AB / "_source_a.png"))
    ea = recognition.embed(anchor, detection.detect_robust(anchor, 0.3)[0].kps)
    out = []
    paths = [AB / "_source_a.png"] + sorted(EXTRA.glob("a*_*.jpg"))
    for p in paths:
        img = cv2.imread(str(p))
        if img is None or img.shape[0] < 50:
            continue
        faces = detection.detect_robust(img, 0.3)
        best = None
        for f in faces:
            try:
                v = recognition.embed(img, f.kps)
            except Exception:  # noqa: BLE001
                continue
            s = recognition.similarity(v, ea)
            if best is None or s > best[0]:
                best = (s, f)
        # Same person only. A group shot must not contribute the wrong face.
        if best and best[0] >= 0.28:
            out.append((p.stem, img, best[1]))
    return out


def main() -> int:
    from app.render import detection, judges, recognition, swapping
    from app.render.identity_bank import IdentityBank
    from app.render.pipeline import FrameRenderer, RenderOptions, apply_preset
    from app.render import alignment
    import app.render.video as V

    srcs = load_sources()
    print("source references accepted: %d" % len(srcs))
    bank = IdentityBank.build(srcs)
    s = bank.summary()
    print("  yaw range %s  bins covered %d/9" % (s["yaw_range"], s["covered_bins"]))
    for r in s["refs"]:
        print("    %-16s yaw %7.1f  quality %.4f  consensus %.4f"
              % (r["name"], r["yaw"], r["quality"], r["consensus"]))

    if s["covered_bins"] <= 2:
        print("\n  NOTE: references span only %s degrees. A pose-aware bank can"
              % (s["yaw_range"],))
        print("  only help where it has references near the target pose, so a")
        print("  null result here is a statement about the SOURCE, not the method.")

    anchor = cv2.imread(str(AB / "_source_a.png"))
    af = detection.detect_robust(anchor, 0.3)[0]
    ea = recognition.embed(anchor, af.kps)
    ja = judges.embed(anchor, af.kps) if judges.available() else None

    # ---- strategies
    vecs = np.stack([r.vec for r in bank.refs])
    quals = np.array([r.quality for r in bank.refs], np.float32)
    best_i = int(np.argmax(quals))
    strategies: dict[str, Any] = {
        "A single best photo": vecs[best_i],
        "B unweighted mean": vecs.mean(0),
        "C quality-weighted": (vecs * quals[:, None]).sum(0) / max(float(quals.sum()), 1e-6),
    }

    cfg = apply_preset(RenderOptions(quality="quality")).config
    print("\npipeline: %s" % cfg.describe())
    print("\n%-24s %9s %9s %9s %9s" % ("strategy", "arcMean", "arcP5", "arcWorst", "sfMean"))
    print("-" * 64)

    results: dict[str, dict[str, Any]] = {}

    def run(label: str, vector_fn) -> None:
        arc: list[float] = []
        sfc: list[float] = []
        by_pose: dict[str, list[float]] = {}
        for clip in CLIPS:
            p = AB / (clip + ".mp4")
            if not p.is_file():
                continue
            info = V.probe(str(p))
            idx = [int(i * (info.total_frames - 1) / 7) for i in range(8)]
            o = RenderOptions(quality="quality", config=cfg)
            o.use_parsing = False
            o.use_occlusion = False
            r = FrameRenderer(cfg, o, bank.to_identity(),
                              float(np.hypot(info.width, info.height)))
            bank.reset_temporal()
            for _, fr in sorted(V.sample_frames(str(p), info, idx).items()):
                fs = detection.detect_with_fallback(fr, 0.5, "quality",
                                                    "yoloface_8n", "scrfd_2.5g")
                if not fs:
                    continue
                f = max(fs, key=lambda x: x.area)
                yaw, _pitch, _roll = alignment.pose_3d(f.kps, fr.shape)
                # Re-condition this frame on the chosen strategy's vector.
                r.embedding = swapping.prepare_embedding(cfg.swapper, vector_fn(yaw))
                out, _ = r.render_face(fr, f)
                os_ = detection.detect_robust(out, 0.3, "quality")
                if not os_:
                    continue
                of = max(os_, key=lambda x: x.area)
                a = recognition.similarity(recognition.embed(out, of.kps), ea)
                arc.append(a)
                bucket = ("frontal" if abs(yaw) < 8 else
                          "mild" if abs(yaw) < 20 else "turned")
                by_pose.setdefault(bucket, []).append(a)
                if ja is not None:
                    sfc.append(judges.similarity(ja, judges.embed(out, of.kps)))
        arr = np.array(arc)
        results[label] = {"arc": arr, "sface": np.array(sfc), "by_pose": by_pose}
        print("%-24s %9.4f %9.4f %9.4f %9.4f"
              % (label, arr.mean(), np.percentile(arr, 5), arr.min(),
                 np.mean(sfc) if sfc else float("nan")))

    for lab, vec in strategies.items():
        run(lab, lambda yaw, v=vec: v)
    run("F pose-aware", lambda yaw: bank.vector_for(yaw, smooth=False))
    run("G pose-aware + smooth", lambda yaw: bank.vector_for(yaw, smooth=True))

    print("\nby target pose bucket (ArcFace mean):")
    buckets = sorted({b for r in results.values() for b in r["by_pose"]})
    print("%-24s %s" % ("strategy", "  ".join("%-10s" % b for b in buckets)))
    for lab, r in results.items():
        cells = []
        for b in buckets:
            v = r["by_pose"].get(b)
            cells.append("%-10s" % ("%.4f" % np.mean(v) if v else "-"))
        print("%-24s %s" % (lab, "  ".join(cells)))

    base = results["C quality-weighted"]["arc"].mean()
    print("\nvs production (C quality-weighted, %.4f):" % base)
    for lab, r in results.items():
        if lab.startswith("C"):
            continue
        print("  %-24s %+.4f" % (lab, r["arc"].mean() - base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
