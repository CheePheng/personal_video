"""Does source video, used as reference ACQUISITION, improve final renders?

V2.6 tried video as a fusion input and it lost: dozens of frames overwhelmed
the photo anchors. This tests the other use -- video as a way to FIND better
reference images, with a hard cap of five handed to the unchanged production
fusion. Frame count cannot become voting power because the cap is structural.

Judged on RENDERED OUTPUT, not on how tidy the selected references look. A
better reference grid that does not improve the swap is not an improvement.

Strategies, all ending in the same production pipeline:

  A  photos only, production fusion          <- control
  B  video only, best 5 acquired frames
  C  2 photos + video, best 5 total
  D  5 photos + video, best 5 total

And the case that matters most: an IMPERFECT photo set (peace sign, blur,
duplicate, bad exposure) with a video that contains clean alternatives.

Run:  python scripts/test_source_acquire.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
R2 = ROOT / "data" / "testclips" / "real2"
FIX = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/fixtures")
WORK = FIX / "acq"
OUT = ROOT / "data" / "benchmarks" / "source_acquire"

# Source person W: photos are stills, and src_video_w.mp4 is 15 minutes of
# the same person with 51 degrees of real yaw. Target is person B's clips.
SRC_VIDEO = FIX / "src_video_w.mp4"
TARGETS = ["A_single_frontal", "B_profile", "G_talking", "I_closeup",
           "N_dark", "P_1080p"]


def contaminate(img: np.ndarray, kind: str) -> Optional[np.ndarray]:
    from app.render import detection
    fs = detection.detect_robust(img, 0.3)
    if not fs:
        return None
    f = max(fs, key=lambda x: x.area)
    o = img.copy().astype(np.float32)
    cx, cy = f.centre
    sz = f.size
    le = f.kps[0]

    if kind == "peace_sign":
        m = np.zeros(img.shape[:2], np.float32)
        hx, hy = int(cx - sz * 0.50), int(cy + sz * 0.20)
        for dx in (-int(sz * 0.10), int(sz * 0.10)):
            cv2.ellipse(m, (hx + dx, hy - int(sz * 0.24)),
                        (int(sz * 0.08), int(sz * 0.30)), 12, 0, 360, 1.0, -1)
        cv2.ellipse(m, (hx, hy + int(sz * 0.16)),
                    (int(sz * 0.26), int(sz * 0.22)), 8, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), sz * 0.02)[:, :, None]
        skin = np.full_like(o, (118, 150, 195))
        return np.clip(o * (1 - m) + skin * m, 0, 255).astype(np.uint8)
    if kind == "blur":
        k = max(3, int(sz * 0.16) | 1)
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "overexposed":
        return np.clip(o * 1.8 + 50, 0, 255).astype(np.uint8)
    return None


def render_score(identity, targets: list[str]) -> dict[str, Any]:
    """Render the targets with this identity and score the output."""
    from app.render import detection, judges, recognition
    from app.render.pipeline import FrameRenderer, RenderOptions, apply_preset
    import app.render.video as V

    cfg = apply_preset(RenderOptions(quality="max")).config
    ref = np.asarray(identity.embedding, np.float32).ravel()
    ref = ref / (float(np.linalg.norm(ref)) or 1.0)

    arc: list[float] = []
    sfc: list[float] = []
    by_pose: dict[str, list[float]] = {}

    # Judge against the SAME fixed anchor for every strategy, so strategies
    # are comparable. The anchor is the person's best clean photo, not the
    # fused vector, which would let each strategy grade its own homework.
    anchor_img = cv2.imread(str(WORK / "clean" / "w_photo0.png"))
    af = detection.detect_robust(anchor_img, 0.3)
    if not af:
        raise SystemExit("anchor photo unusable")
    a_arc = recognition.embed(anchor_img, af[0].kps)
    a_sf = judges.embed(anchor_img, af[0].kps) if judges.available() else None

    for name in targets:
        p = AB / (name + ".mp4")
        if not p.is_file():
            continue
        info = V.probe(str(p))
        idx = [int(i * (info.total_frames - 1) / 7) for i in range(8)]
        o = RenderOptions(quality="max", config=cfg)
        o.use_parsing = False
        o.use_occlusion = False
        r = FrameRenderer(cfg, o, identity,
                          float(np.hypot(info.width, info.height)))
        for _, fr in sorted(V.sample_frames(str(p), info, idx).items()):
            fs = detection.detect_with_fallback(fr, 0.5, "quality",
                                                "yoloface_8n", "scrfd_2.5g")
            if not fs:
                continue
            f = max(fs, key=lambda x: x.area)
            from app.render import alignment
            yaw, _p, _r = alignment.pose_3d(f.kps, fr.shape)
            out, _ = r.render_face(fr, f)
            os_ = detection.detect_robust(out, 0.3, "quality")
            if not os_:
                continue
            of = max(os_, key=lambda x: x.area)
            v = recognition.similarity(recognition.embed(out, of.kps), a_arc)
            arc.append(v)
            b = ("frontal" if abs(yaw) < 10 else
                 "10-30" if abs(yaw) < 30 else "30+")
            by_pose.setdefault(b, []).append(v)
            if a_sf is not None:
                sfc.append(judges.similarity(a_sf, judges.embed(out, of.kps)))

    return {
        "arc": float(np.mean(arc)) if arc else float("nan"),
        "arc_p5": float(np.percentile(arc, 5)) if arc else float("nan"),
        "sface": float(np.mean(sfc)) if sfc else float("nan"),
        "by_pose": {k: float(np.mean(v)) for k, v in by_pose.items()},
        "n": len(arc),
    }


def identity_from_paths(paths: list[Path]):
    from app.render import detection, recognition
    items = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        fs = detection.detect_robust(img, 0.3)
        if fs:
            items.append((p.name, img, max(fs, key=lambda x: x.area)))
    return recognition.build_source_identity(items)


def identity_from_analyses(chosen):
    """Feed acquired references into the UNCHANGED production fusion."""
    from app.render import recognition
    items = []
    for a in chosen:
        img = cv2.imread(str(WORK / "frames" / (a.name.replace("@", "_at_")
                                                .replace(":", "-") + ".png")))
        if img is None:
            continue
        from app.render import detection
        fs = detection.detect_robust(img, 0.3)
        if fs:
            items.append((a.name, img, max(fs, key=lambda x: x.area)))
    return recognition.build_source_identity(items)


def main() -> int:
    from app.render import source_acquire as ACQ

    WORK.mkdir(parents=True, exist_ok=True)
    clean = WORK / "clean"
    clean.mkdir(exist_ok=True)
    frames = WORK / "frames"
    frames.mkdir(exist_ok=True)

    srcs = sorted((FIX / "wsrc").glob("w_photo*.png"))
    if len(srcs) < 5 or not SRC_VIDEO.is_file():
        raise SystemExit("need 5 photos of W plus src_video_w.mp4")
    for p in srcs:
        if not (clean / p.name).is_file():
            shutil.copy(p, clean / p.name)
    clean_photos = sorted(clean.glob("w_photo*.png"))

    # Imperfect set: photo0 clean, photo1 peace sign, photo2 blur,
    # photo3 overexposed, photo4 duplicate of photo0.
    bad = WORK / "imperfect"
    bad.mkdir(exist_ok=True)
    if not (bad / "p4_dup.png").is_file():
        shutil.copy(clean_photos[0], bad / "p0_clean.png")
        for src, kind, out in ((clean_photos[1], "peace_sign", "p1_peace.png"),
                               (clean_photos[2], "blur", "p2_blur.png"),
                               (clean_photos[3], "overexposed", "p3_over.png")):
            c = contaminate(cv2.imread(str(src)), kind)
            if c is not None:
                cv2.imwrite(str(bad / out), c)
        shutil.copy(clean_photos[0], bad / "p4_dup.png")
    bad_photos = sorted(bad.glob("p*.png"))

    def dump(chosen) -> None:
        """Write acquired frames so production fusion can re-read them."""
        import app.render.video as V
        info = V.probe(str(SRC_VIDEO))
        need = {}
        for a in chosen:
            fn = frames / (a.name.replace("@", "_at_").replace(":", "-") + ".png")
            if fn.is_file():
                continue
            if "@" in a.name:
                ts = float(a.name.split("@")[1].rstrip("s"))
                need[int(round(ts * info.fps))] = fn
            else:
                for root in (clean, bad):
                    if (root / a.name).is_file():
                        shutil.copy(root / a.name, fn)
                        break
        if need:
            got = V.sample_frames(str(SRC_VIDEO), info, sorted(need))
            for k, fn in need.items():
                if k in got:
                    cv2.imwrite(str(fn), got[k])

    results: dict[str, Any] = {}

    def run(label: str, photos: list[Path], video: Optional[Path]) -> None:
        if video is None:
            ident = identity_from_paths(photos)
            note = "%d photos, production fusion" % len(photos)
        else:
            chosen, rep = ACQ.acquire([str(p) for p in photos],
                                      str(video) if video else None,
                                      use_cache=True)
            dump(chosen)
            ident = identity_from_analyses(chosen)
            note = "%d sel (%dp/%dv) bands=%s" % (
                rep["n_selected"], rep["from_photo"], rep["from_video"],
                ",".join(b[:5] for b in rep["bands_covered"]))
        s = render_score(ident, TARGETS)
        results[label] = {**s, "note": note}
        print("  %-30s %7.4f %7.4f %7.4f   %s"
              % (label, s["arc"], s["arc_p5"], s["sface"], note), flush=True)

    print("CLEAN photo set")
    print("  %-30s %7s %7s %7s   %s" % ("strategy", "arc", "arcP5", "sface", "detail"))
    print("  " + "-" * 86)
    run("A clean 5 photos (control)", clean_photos, None)
    run("B video only", [], SRC_VIDEO)
    run("C 2 photos + video", clean_photos[:2], SRC_VIDEO)
    run("D 5 photos + video", clean_photos, SRC_VIDEO)

    print("\nIMPERFECT photo set (peace sign, blur, overexposed, duplicate)")
    print("  " + "-" * 86)
    run("E imperfect 5 (control)", bad_photos, None)
    run("F imperfect 5 + video", bad_photos, SRC_VIDEO)

    print("\n" + "=" * 90)
    ctrl = results["A clean 5 photos (control)"]["arc"]
    print("vs clean control (%.4f):" % ctrl)
    for k, v in results.items():
        if k.startswith("A "):
            continue
        print("  %-30s %+.4f" % (k, v["arc"] - ctrl))
    e = results["E imperfect 5 (control)"]["arc"]
    f = results["F imperfect 5 + video"]["arc"]
    print("\nIMPERFECT-SET RECOVERY: %.4f -> %.4f  (%+.4f)" % (e, f, f - e))
    print("  recovers %.0f%% of the gap to the clean control"
          % (100 * (f - e) / max(ctrl - e, 1e-9)) if ctrl > e else "")

    print("\nby target pose:")
    poses = ["frontal", "10-30", "30+"]
    print("  %-30s %s" % ("strategy", " ".join("%-10s" % p for p in poses)))
    for k, v in results.items():
        print("  %-30s %s" % (k, " ".join(
            "%-10s" % ("%.4f" % v["by_pose"][p] if p in v["by_pose"] else "-")
            for p in poses)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acquire.json").write_text(json.dumps(results, indent=2, default=str),
                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
