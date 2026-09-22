"""Prove the occlusion detector reacts to OBJECTS, not to brightness.

The previous heuristic failed this: darkening an unmodified face changed its
verdict from "no occlusion" to "both eyes occluded". Any detector used for
source weighting has to pass two tests, in this order:

  INVARIANCE  the same face at three exposures must give the same verdict.
              Dark skin, shadow and low light are not occlusion.

  SENSITIVITY it must still fire on a real obstruction -- hand over an eye,
              peace sign on a cheek, hair across the face, sunglasses.

A detector that passes only the first is a constant; one that passes only the
second is the old bug. Both, or it is not used.

Run:  python scripts/test_occlusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
FIX = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/fixtures")

OK: list[str] = []
BAD: list[str] = []


def check(label: str, cond: bool, note: str = "") -> None:
    (OK if cond else BAD).append(label)
    print("  [%s] %-46s %s" % ("PASS" if cond else "FAIL", label, note))


def occlude(img: np.ndarray, kind: str) -> Optional[np.ndarray]:
    from app.render import detection
    fs = detection.detect_robust(img, 0.3)
    if not fs:
        return None
    f = max(fs, key=lambda x: x.area)
    o = img.copy().astype(np.float32)
    cx, cy = f.centre
    sz = f.size
    le, re = f.kps[0], f.kps[1]

    if kind == "hand_eye":
        m = np.zeros(img.shape[:2], np.float32)
        cv2.ellipse(m, (int(le[0]), int(le[1])),
                    (int(sz * 0.30), int(sz * 0.22)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), sz * 0.025)[:, :, None]
        return np.clip(o * (1 - m) + np.full_like(o, (118, 150, 195)) * m,
                       0, 255).astype(np.uint8)
    if kind == "peace_cheek":
        m = np.zeros(img.shape[:2], np.float32)
        hx, hy = int(cx - sz * 0.45), int(cy + sz * 0.22)
        for dx in (-int(sz * 0.10), int(sz * 0.10)):
            cv2.ellipse(m, (hx + dx, hy - int(sz * 0.24)),
                        (int(sz * 0.08), int(sz * 0.30)), 12, 0, 360, 1.0, -1)
        cv2.ellipse(m, (hx, hy + int(sz * 0.16)),
                    (int(sz * 0.26), int(sz * 0.22)), 8, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), sz * 0.02)[:, :, None]
        return np.clip(o * (1 - m) + np.full_like(o, (118, 150, 195)) * m,
                       0, 255).astype(np.uint8)
    if kind == "sunglasses":
        m = np.zeros(img.shape[:2], np.float32)
        for c in (le, re):
            cv2.ellipse(m, (int(c[0]), int(c[1])),
                        (int(sz * 0.26), int(sz * 0.17)), 0, 0, 360, 1.0, -1)
        cv2.line(m, (int(le[0]), int(le[1])), (int(re[0]), int(re[1])),
                 1.0, max(2, int(sz * 0.05)))
        m = cv2.GaussianBlur(m, (0, 0), 2.0)[:, :, None]
        return np.clip(o * (1 - m) + np.full_like(o, (18, 16, 20)) * m,
                       0, 255).astype(np.uint8)
    if kind == "hair":
        m = np.zeros(img.shape[:2], np.float32)
        for i in range(8):
            x0 = int(cx - sz * 0.5 + i * sz * 0.14)
            pts = np.array([[x0, int(cy - sz * 0.95)],
                            [x0 + int(sz * 0.12), int(cy)],
                            [x0 - int(sz * 0.04), int(cy + sz * 0.85)]], np.int32)
            cv2.polylines(m, [pts], False, 1.0, max(2, int(sz * 0.05)))
        m = cv2.GaussianBlur(m, (0, 0), sz * 0.012)[:, :, None]
        return np.clip(o * (1 - m) + np.full_like(o, (26, 24, 30)) * m,
                       0, 255).astype(np.uint8)
    return None


def measure(img: np.ndarray):
    from app.render import alignment, detection, occlusion
    fs = detection.detect_robust(img, 0.3)
    if not fs:
        return None
    f = max(fs, key=lambda x: x.area)
    yaw, _p, _r = alignment.pose_3d(f.kps, img.shape)
    r = occlusion.analyse_occlusion(img, f, yaw)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r["exposure"] = float(np.median(grey))
    return r


def main() -> int:
    sources = [AB / "_source_a.png"]
    w = sorted((FIX / "wsrc").glob("w_photo*.png"))
    if w:
        sources.append(w[0])
    m = sorted((FIX / "msrc").glob("m_*.jpg"))
    if m:
        sources.append(m[0])

    print("EXPOSURE INVARIANCE -- same face, three exposures\n")
    print("  %-22s %9s %9s %s" % ("image", "exposure", "visible", "occluded"))
    for src in sources:
        img = cv2.imread(str(src))
        if img is None:
            continue
        variants = {
            "normal": img,
            "darkened": np.clip(img.astype(np.float32) * 0.30, 0, 255).astype(np.uint8),
            "brightened": np.clip(img.astype(np.float32) * 1.70 + 30, 0, 255).astype(np.uint8),
        }
        fracs = {}
        for name, v in variants.items():
            r = measure(v)
            if r is None:
                print("  %-22s  face lost" % ("%s/%s" % (src.stem[:12], name)))
                continue
            fracs[name] = r["visible_fraction"]
            print("  %-22s %9.0f %9.3f %s"
                  % ("%s/%s" % (src.stem[:12], name), r["exposure"],
                     r["visible_fraction"], r["occluded_regions"] or "-"))
        if len(fracs) == 3:
            spread = max(fracs.values()) - min(fracs.values())
            check("%s: exposure-invariant" % src.stem[:16], spread < 0.15,
                  "visible spread %.3f across 3 exposures" % spread)
        print()

    print("SENSITIVITY -- does it fire on real obstructions?\n")
    base = cv2.imread(str(sources[0]))
    r0 = measure(base)
    print("  %-22s visible %.3f  occluded %s"
          % ("clean", r0["visible_fraction"], r0["occluded_regions"] or "-"))
    for kind in ("hand_eye", "peace_cheek", "sunglasses", "hair"):
        c = occlude(base, kind)
        if c is None:
            continue
        r = measure(c)
        if r is None:
            print("  %-22s face lost" % kind)
            continue
        drop = r0["visible_fraction"] - r["visible_fraction"]
        print("  %-22s visible %.3f  (drop %+.3f)  occluded %s"
              % (kind, r["visible_fraction"], -drop, r["occluded_regions"] or "-"))
        check("detects %s" % kind, drop > 0.03,
              "visibility drops %.3f" % drop)

    print("\n" + "=" * 68)
    print("  %d passed, %d failed" % (len(OK), len(BAD)))
    for b in BAD:
        print("    FAILED: %s" % b)
    print("  VERDICT: %s" % ("DETECTOR VALID" if not BAD else "DETECTOR INVALID"))
    print("=" * 68)
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
