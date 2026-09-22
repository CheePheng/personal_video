"""Exposure-invariant occlusion detection for source photos.

The previous detector in ``source_analysis`` decided a region was occluded
from raw luminance, chroma distance to skin, and texture variance. That is
invalid, and the proof is one image: take an unmodified face, darken it, and
the verdict changes.

    original   exposure 71   visible 0.874   occluded: none
    darkened   exposure 21   visible 0.566   occluded: both eyes

Nothing about the face changed. Normal shadow on a dark photo reads as
"crushed to black", so the test fires on unobstructed faces. On real
material it rated a CLEAN photo as MORE occluded (0.616) than the same photo
with a hand across the cheek (0.797), because skin-toned fingers RAISE the
median exposure toward the band the heuristic considers healthy.

This replaces it with the two semantic models the project already ships:

  BiSeNet  19-class CelebAMask-HQ face parsing. Asked which pixels are
           actually facial skin/features, and which are hair, cloth, hat or
           glasses. A class label does not care how bright the pixel is.

  XSeg     purpose-built occlusion model: 1 where the face is visible, 0
           where something covers it.

Both are learned semantic models rather than pixel-statistics heuristics, so
a dark face is still a face to them. Validated against darkened/brightened
copies before use -- see scripts/test_occlusion.py.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from app.render import alignment, masking, sessions
from app.render.registry import get_model
from app.render.types import Face, RenderError

# Regions as fractions of the aligned crop, matching source_analysis so the
# two can be compared directly.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "forehead":    (0.22, 0.05, 0.78, 0.28),
    "left_eye":    (0.14, 0.28, 0.45, 0.46),
    "right_eye":   (0.55, 0.28, 0.86, 0.46),
    "nose":        (0.38, 0.40, 0.62, 0.62),
    "left_cheek":  (0.10, 0.46, 0.38, 0.72),
    "right_cheek": (0.62, 0.46, 0.90, 0.72),
    "mouth":       (0.32, 0.64, 0.68, 0.82),
    "jaw":         (0.22, 0.78, 0.78, 0.96),
}

REGION_IMPORTANCE = {
    "left_eye": 0.18, "right_eye": 0.18, "nose": 0.18, "mouth": 0.12,
    "left_cheek": 0.11, "right_cheek": 0.11, "jaw": 0.08, "forehead": 0.04,
}

# Below this share of facial pixels a region is considered obstructed.
OCCLUDED_BELOW = 0.45


def region_visibility(frame: np.ndarray, face: Face) -> dict[str, float]:
    """Per-region fraction of pixels that are genuinely visible face.

    Combines the parser's "this is face" verdict with XSeg's "this is not
    covered" verdict. Both are semantic; neither is a brightness test.
    """
    try:
        face_map, occl_map = masking.parsing_masks(frame, face.kps)
    except RenderError:
        face_map = occl_map = None
    try:
        xseg = masking.occlusion_mask(frame, face.kps)
    except RenderError:
        xseg = None

    if face_map is None and xseg is None:
        raise RenderError("no occlusion model available")

    # Bring everything into one square grid so the region boxes apply.
    size = 256
    def fit(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if m is None:
            return None
        if m.shape[0] != size or m.shape[1] != size:
            m = cv2.resize(m.astype(np.float32), (size, size),
                           interpolation=cv2.INTER_LINEAR)
        return np.clip(m.astype(np.float32), 0.0, 1.0)

    fm, om, xs = fit(face_map), fit(occl_map), fit(xseg)

    # Visible face = parsed as face, NOT parsed as occluder, and XSeg agrees.
    vis = np.ones((size, size), np.float32)
    if fm is not None:
        vis = vis * fm
    if om is not None:
        vis = vis * (1.0 - om)
    if xs is not None:
        vis = vis * xs

    out: dict[str, float] = {}
    for name, (rx0, ry0, rx1, ry1) in REGIONS.items():
        x0, y0 = int(rx0 * size), int(ry0 * size)
        x1, y1 = int(rx1 * size), int(ry1 * size)
        reg = vis[y0:y1, x0:x1]
        out[name] = float(reg.mean()) if reg.size else 0.0
    return out


def analyse_occlusion(frame: np.ndarray, face: Face,
                      yaw: float) -> dict[str, Any]:
    """Visibility summary that ignores self-occlusion from pose."""
    vis = region_visibility(frame, face)

    # A turned head genuinely hides its far side. That is geometry, not an
    # obstruction, so it must not count as "something is covering this face".
    hidden: set[str] = set()
    if yaw > 22:
        hidden |= {"left_cheek", "left_eye"}
    elif yaw < -22:
        hidden |= {"right_cheek", "right_eye"}

    scored = {k: v for k, v in vis.items() if k not in hidden}
    imp = {k: REGION_IMPORTANCE[k] for k in scored}
    tot = sum(imp.values()) or 1.0
    frac = float(sum(scored[k] * imp[k] for k in scored) / tot)
    occluded = sorted(k for k, v in scored.items() if v < OCCLUDED_BELOW)
    return {"visibility": vis, "visible_fraction": frac,
            "occluded_regions": occluded}
