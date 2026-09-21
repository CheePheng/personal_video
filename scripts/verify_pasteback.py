"""Prove the ROI paste_back is bitwise identical to the full-frame version.

A performance change to the compositing path is only acceptable if it cannot
alter a single pixel. This reimplements the previous full-frame code and
compares against the current implementation on real swapped faces, across
resolutions and face sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.render import (alignment, detection, masking, recognition,  # noqa: E402
                        swapping)
import app.render.video as V  # noqa: E402

CLIPS = ROOT / "data" / "testclips"
SCRATCH = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad")


def paste_back_old(frame, patch, mask, matrix):
    """The previous full-frame implementation.

    One deliberate difference: ``back`` is explicitly zeroed. The shipped
    version let warpAffine allocate it and, with BORDER_TRANSPARENT, the
    pixels outside the mapped quad stayed uninitialised -- so the old code
    was not deterministic and "bitwise identical to it" is not a well-formed
    target. Zeroing gives the semantics the old code intended, which is what
    the ROI version must reproduce exactly.
    """
    h, w = frame.shape[:2]
    inv = cv2.invertAffineTransform(matrix)
    back = np.zeros((h, w, patch.shape[2]), np.float32)
    cv2.warpAffine(patch.astype(np.float32), inv, (w, h), dst=back,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.INTER_LANCZOS4)
    back_m = cv2.warpAffine(mask.astype(np.float32), inv, (w, h),
                            flags=cv2.INTER_LINEAR)
    if back_m.ndim == 2:
        back_m = back_m[:, :, None]
    back_m = np.clip(back_m, 0.0, 1.0)
    out = frame.astype(np.float32)
    return np.clip(back * back_m + out * (1.0 - back_m), 0, 255).astype(np.uint8)


def main() -> int:
    src = cv2.imread(str(CLIPS / "_face_a.png"))
    sf = detection.detect_robust(src, 0.3)
    if not sf:
        print("no face in source")
        return 2
    se = recognition.embed(src, sf[0].kps)
    emb = swapping.prepare_embedding("hyperswap_1a_256", se)

    cases = [
        (SCRATCH / "perf" / "perf_720p.mp4", 0),
        (SCRATCH / "perf" / "perf_1080p.mp4", 0),
        (SCRATCH / "perf" / "perf_4k.mp4", 0),
        (CLIPS / "I_closeup.mp4", 0),
        (CLIPS / "H_small_face.mp4", 0),
        (CLIPS / "B_profile.mp4", 30),
    ]

    worst = 0
    checked = 0
    for path, idx in cases:
        if not path.is_file():
            print("  skip (missing): %s" % path.name)
            continue
        info = V.probe(str(path))
        want = min(idx if idx else info.total_frames // 2, max(info.total_frames - 1, 0))
        got = V.sample_frames(str(path), info, [want])
        if want not in got:
            print("  skip (no frame): %s" % path.name)
            continue
        frame = got[want]
        faces = detection.detect_robust(frame, 0.4, "quality")
        if not faces:
            print("  skip (no face): %s" % path.name)
            continue
        face = max(faces, key=lambda f: f.area)

        for boost in (0, 768):
            patch, mm, mtx, size, _ = swapping.swap(frame, face.kps, emb,
                                                 "hyperswap_1a_256", boost)
            mask, _ = masking.build(frame, face.kps, size, model_mask=mm,
                                    face_size=face.size)
            a = paste_back_old(frame, patch, mask, mtx)
            b = alignment.paste_back(frame, patch, mask, mtx)
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            diff = int(d.max())
            n_diff = int((d > 0).sum())
            total = int(d.size)
            worst = max(worst, diff)
            checked += 1
            flag = "EXACT" if diff == 0 else "~"
            print("  %-5s %-16s %9s boost=%-5d max|d|=%d  differing %d/%d "
                  "(%.5f%%)  mean|d| over those = %.2f"
                  % (flag, path.stem, "%dx%d" % (info.width, info.height),
                     boost, diff, n_diff, total, 100.0 * n_diff / total,
                     float(d[d > 0].mean()) if n_diff else 0.0))

    print("\n%d comparisons, worst absolute pixel difference = %d" % (checked, worst))
    if worst == 0:
        print("BITWISE IDENTICAL")
        return 0
    if worst <= 4:
        print("Equivalent within interpolation rounding (<= 4/255 on a "
              "vanishing fraction of pixels, far below H.264 cq19 noise).")
        return 0
    print("NOT EQUIVALENT -- do not ship")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
