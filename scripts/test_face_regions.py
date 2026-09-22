"""Eyes, mouth and temporal quality on REAL footage.

Global identity can look healthy while the eyes are dead or the mouth is
frozen. A single mean over the whole face cannot see either, so this scores
the regions separately, and scores them over time rather than per frame.

Two principles the measurements follow:

  PERFORMANCE BELONGS TO THE TARGET. Gaze, blink, mouth opening and jaw
  timing must survive the swap unchanged; only appearance should become the
  source's. So the eye and mouth metrics compare the OUTPUT against the
  INPUT frame, not against the source photo. A swap that imposes the
  source's expression scores badly here even if identity is perfect.

  TEMPORAL ERROR IS NOT FRAME ERROR. A pipeline can be right on average and
  still flicker. Every region is also measured as frame-to-frame variation,
  which is what a viewer actually notices.

Run:  python scripts/test_face_regions.py [clip ...]
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
REAL2 = ROOT / "data" / "testclips" / "real2"
SOURCE = AB / "_source_a.png"
OUT = ROOT / "data" / "benchmarks" / "regions"

CLIPS = ["r_frontal_talk", "r_right_mild", "r_right_strong",
         "r_left", "r_profile_max", "r_motion"]


def eye_boxes(kps: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Square regions around each eye, scaled to interocular distance."""
    le, re = kps[0], kps[1]
    d = float(np.linalg.norm(re - le)) or 1.0
    h = d * 0.42
    out = []
    for c in (le, re):
        out.append((int(c[0] - h), int(c[1] - h * 0.8),
                    int(c[0] + h), int(c[1] + h * 0.8)))
    return out


def mouth_box(kps: np.ndarray) -> tuple[int, int, int, int]:
    ml, mr, nose = kps[3], kps[4], kps[2]
    cx = (ml[0] + mr[0]) / 2.0
    cy = (ml[1] + mr[1]) / 2.0
    w = max(float(np.linalg.norm(mr - ml)), 1.0) * 1.35
    h = w * 0.85
    # Bias downward: the chin side carries jaw motion the corners do not.
    cy += h * 0.08
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def crop(img: np.ndarray, box: tuple[int, int, int, int],
         size: int = 64) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (size, size))


def region_delta(a: np.ndarray, b: np.ndarray) -> float:
    """How much a region CHANGED between input and output, 0 = identical."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Normalise for the brightness/contrast shift a swap legitimately makes,
    # so this measures STRUCTURE (where the lids and lips are) rather than skin.
    ga = (ga - ga.mean()) / (ga.std() + 1e-6)
    gb = (gb - gb.mean()) / (gb.std() + 1e-6)
    return float(np.abs(ga - gb).mean())


def geom(kps: np.ndarray) -> dict[str, float]:
    eye = float(np.linalg.norm(kps[1] - kps[0])) or 1.0
    mouth_w = float(np.linalg.norm(kps[4] - kps[3])) / eye
    mouth_open = float(np.linalg.norm(
        kps[2] - (kps[3] + kps[4]) / 2.0)) / eye
    # Eye-line tilt relative to the mouth line. Measuring each eye's offset
    # from their own midpoint is identically zero -- the two deviations
    # cancel by construction -- so that version measured nothing at all.
    # The angle between the eye axis and the mouth axis does change when a
    # swap skews one side of the face.
    eye_ang = float(np.degrees(np.arctan2(kps[1][1] - kps[0][1],
                                          kps[1][0] - kps[0][0])))
    mth_ang = float(np.degrees(np.arctan2(kps[4][1] - kps[3][1],
                                          kps[4][0] - kps[3][0])))
    return {"mouth_w": mouth_w, "mouth_open": mouth_open,
            "eye_mouth_skew": abs(eye_ang - mth_ang)}


def main() -> int:
    from app.render import detection, judges, recognition, pipeline
    from app.render.pipeline import FrameRenderer, RenderOptions, apply_preset
    import app.render.video as V

    clips = sys.argv[1:] or CLIPS
    src = cv2.imread(str(SOURCE))
    sf = detection.detect_robust(src, 0.3)
    if not sf:
        raise SystemExit("no face in source")
    ea = recognition.embed(src, sf[0].kps)
    ja = judges.embed(src, sf[0].kps) if judges.available() else None
    ident = pipeline.load_source([str(SOURCE)])
    cfg = apply_preset(RenderOptions(quality="quality")).config
    print("pipeline: %s\n" % cfg.describe())

    print("%-18s %7s %7s | %7s %7s | %7s %7s %7s | %7s"
          % ("clip", "arc", "sface", "eyeD", "mouthD", "eyeJit", "mthJit",
             "idJit", "flow"))
    print("-" * 92)

    allrows: list[dict[str, Any]] = []
    for name in clips:
        p = REAL2 / (name + ".mp4")
        if not p.is_file():
            print("  %-18s MISSING" % name)
            continue
        info = V.probe(str(p))
        idx = list(range(0, info.total_frames, 2))
        frames = sorted(V.sample_frames(str(p), info, idx).items())
        o = RenderOptions(quality="quality", config=cfg)
        o.use_parsing = False
        o.use_occlusion = False
        r = FrameRenderer(cfg, o, ident, float(np.hypot(info.width, info.height)))

        arc: list[float] = []
        sfc: list[float] = []
        eyeD: list[float] = []
        mthD: list[float] = []
        embs: list[np.ndarray] = []
        eye_seq: list[np.ndarray] = []
        mth_seq: list[np.ndarray] = []
        flow: list[float] = []
        gdelta: list[dict[str, float]] = []
        prev_out = None

        for _, fr in frames:
            fs = detection.detect_with_fallback(fr, 0.5, "quality",
                                                "yoloface_8n", "scrfd_2.5g")
            if not fs:
                continue
            f = max(fs, key=lambda x: x.area)
            out, _ = r.render_face(fr, f)
            os_ = detection.detect_robust(out, 0.3, "quality")
            if not os_:
                continue
            of = max(os_, key=lambda x: x.area)

            emb = recognition.embed(out, of.kps)
            arc.append(recognition.similarity(emb, ea))
            embs.append(emb.ravel())
            if ja is not None:
                sfc.append(judges.similarity(ja, judges.embed(out, of.kps)))

            # Eyes and mouth: compare OUTPUT against INPUT at the same
            # landmarks. Large change = the swap altered the performance.
            for box in eye_boxes(f.kps):
                a, b = crop(fr, box), crop(out, box)
                if a is not None and b is not None:
                    eyeD.append(region_delta(a, b))
                    eye_seq.append(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32))
            mb = mouth_box(f.kps)
            a, b = crop(fr, mb), crop(out, mb)
            if a is not None and b is not None:
                mthD.append(region_delta(a, b))
                mth_seq.append(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32))

            gi, go = geom(f.kps), geom(of.kps)
            gdelta.append({k: abs(go[k] - gi[k]) for k in gi})

            if prev_out is not None:
                x1, y1, x2, y2 = [int(v) for v in f.box]
                from app.render import metrics as M
                flow.append(M.flow_warped_difference(prev_out, out, f.box))
            prev_out = out

        if not arc:
            print("  %-18s no usable frames" % name)
            continue

        def jitter(seq: list[np.ndarray]) -> float:
            if len(seq) < 2:
                return float("nan")
            return float(np.mean([np.abs(seq[i] - seq[i - 1]).mean()
                                  for i in range(1, len(seq))]))

        idjit = float(np.mean([
            1.0 - float(np.dot(embs[i] / (np.linalg.norm(embs[i]) or 1),
                               embs[i - 1] / (np.linalg.norm(embs[i - 1]) or 1)))
            for i in range(1, len(embs))])) if len(embs) > 1 else float("nan")

        row = {
            "clip": name, "n": len(arc),
            "arc": float(np.mean(arc)), "arc_p5": float(np.percentile(arc, 5)),
            "sface": float(np.mean(sfc)) if sfc else float("nan"),
            "eye_delta": float(np.mean(eyeD)) if eyeD else float("nan"),
            "mouth_delta": float(np.mean(mthD)) if mthD else float("nan"),
            "eye_jitter": jitter(eye_seq), "mouth_jitter": jitter(mth_seq),
            "identity_jitter": idjit,
            "flow": float(np.mean(flow)) if flow else float("nan"),
            "mouth_open_drift": float(np.mean([g["mouth_open"] for g in gdelta])),
            "mouth_w_drift": float(np.mean([g["mouth_w"] for g in gdelta])),
            "eye_skew_drift": float(np.mean([g["eye_mouth_skew"] for g in gdelta])),
        }
        allrows.append(row)
        print("%-18s %7.4f %7.4f | %7.4f %7.4f | %7.2f %7.2f %7.4f | %7.2f"
              % (name, row["arc"], row["sface"], row["eye_delta"],
                 row["mouth_delta"], row["eye_jitter"], row["mouth_jitter"],
                 row["identity_jitter"], row["flow"]))

    if allrows:
        print("\ngeometry drift (output vs input, lower = performance preserved):")
        print("%-18s %14s %14s %14s" % ("clip", "mouth_open", "mouth_width", "eye_skew"))
        for r in allrows:
            print("%-18s %14.5f %14.5f %14.5f"
                  % (r["clip"], r["mouth_open_drift"], r["mouth_w_drift"],
                     r["eye_skew_drift"]))
        a = np.array([r["arc"] for r in allrows])
        print("\nOVERALL  arc %.4f   eyeD %.4f   mouthD %.4f   idJit %.5f"
              % (a.mean(), np.nanmean([r["eye_delta"] for r in allrows]),
                 np.nanmean([r["mouth_delta"] for r in allrows]),
                 np.nanmean([r["identity_jitter"] for r in allrows])))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "regions.json").write_text(json.dumps(allrows, indent=2),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
