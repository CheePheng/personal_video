"""Quality measurement.

Everything the benchmark decides with is computed here. The metrics are kept
separate and reported individually -- collapsing them into one number early
would hide *why* a pipeline won, and the trade-offs (sharper but less like the
person) are exactly what we need to see.

Conventions: higher is better for identity/sharpness; lower is better for
anything named *delta*, *jitter*, *flicker* or *seam*.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.render import recognition
from app.render.types import Face


# ---------------------------------------------------------------- identity
def identity_similarity(frame: np.ndarray, kps: np.ndarray,
                        source_embedding: np.ndarray) -> Optional[float]:
    """Cosine between the source identity and the rendered face. Higher = better."""
    try:
        return recognition.similarity(recognition.embed(frame, kps), source_embedding)
    except Exception:  # noqa: BLE001
        return None


def identity_stability(prev: Optional[np.ndarray], cur: Optional[np.ndarray]) -> Optional[float]:
    """Cosine between consecutive rendered faces -- identity flicker."""
    if prev is None or cur is None:
        return None
    return recognition.similarity(prev, cur)


# ---------------------------------------------------------------- geometry
def landmark_delta(before: np.ndarray, after: np.ndarray, face_size: float) -> float:
    """How far the face geometry moved, normalised by face size.

    The target's *performance* (mouth, brows, head pose) should survive the
    swap. Large drift means the swapper imposed the source's geometry.
    """
    if before.shape != after.shape:
        return 0.0
    return float(np.linalg.norm(after - before, axis=1).mean() / max(face_size, 1.0))


def expression_delta(before: np.ndarray, after: np.ndarray) -> float:
    """Shape-only geometry change, with pose removed.

    Landmarks are centred and scale-normalised first, so a face that merely
    moved or got closer does not register as an expression change.
    """
    def norm(k: np.ndarray) -> np.ndarray:
        c = k - k.mean(0)
        s = float(np.linalg.norm(c)) or 1.0
        return c / s
    return float(np.linalg.norm(norm(after) - norm(before), axis=1).mean())


# ---------------------------------------------------------------- blending
def seam_score(frame: np.ndarray, mask_full: np.ndarray) -> float:
    """Gradient discontinuity along the mask boundary. Lower = better.

    A good blend has no edge the eye can find. We compare image gradient energy
    in a thin band around the mask boundary against nearby interior gradient:
    a visible seam shows up as a boundary spike that the interior lacks.
    """
    m = (mask_full > 0.5).astype(np.uint8)
    if m.sum() < 64:
        return 0.0

    k = np.ones((5, 5), np.uint8)
    band = (cv2.dilate(m, k, iterations=2) - cv2.erode(m, k, iterations=2)).astype(bool)
    interior = cv2.erode(m, k, iterations=4).astype(bool)
    if band.sum() < 32 or interior.sum() < 32:
        return 0.0

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    edge = float(mag[band].mean())
    inner = float(mag[interior].mean()) or 1.0
    # Raw ratio, NOT clamped at zero. A clean blend sits below 1.0 (the
    # boundary is smoother than the face interior); a visible seam pushes it
    # above 1.0. Clamping would collapse every good blend to the same 0.0 and
    # leave this metric unable to rank the candidates it exists to separate.
    return edge / inner


def color_discontinuity(frame: np.ndarray, mask_full: np.ndarray) -> float:
    """Skin-tone jump across the mask boundary. Lower = better."""
    m = (mask_full > 0.5).astype(np.uint8)
    if m.sum() < 64:
        return 0.0
    k = np.ones((5, 5), np.uint8)
    inside = (cv2.erode(m, k, iterations=2) > 0)
    outside = ((cv2.dilate(m, k, iterations=4) - cv2.dilate(m, k, iterations=1)) > 0)
    if inside.sum() < 32 or outside.sum() < 32:
        return 0.0
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.linalg.norm(lab[inside].mean(0) - lab[outside].mean(0)))


# ---------------------------------------------------------------- detail
def sharpness(frame: np.ndarray, box: np.ndarray) -> float:
    """Laplacian variance inside the face box. Higher = more detail."""
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def texture_retention(original: np.ndarray, rendered: np.ndarray, box: np.ndarray) -> float:
    """Ratio of high-frequency energy kept vs the original face.

    ~1.0 means detail comparable to the target; far below means smoothed to
    plastic; far above can mean a restorer inventing texture.
    """
    def hf(img: np.ndarray) -> float:
        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
        c = img[y1:y2, x1:x2]
        if c.size == 0:
            return 0.0
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.abs(g - cv2.GaussianBlur(g, (9, 9), 0)).mean())
    o = hf(original)
    return float(hf(rendered) / o) if o > 1e-3 else 0.0


# ---------------------------------------------------------------- temporal
def flow_warped_difference(prev: np.ndarray, cur: np.ndarray,
                           box: Optional[np.ndarray] = None) -> float:
    """Temporal flicker with real motion removed. Lower = steadier.

    Optical flow warps the previous frame onto the current one; whatever
    difference remains is not motion, it is the render changing its mind --
    shimmer, mask crawl, restorer noise.
    """
    if box is not None:
        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
        prev, cur = prev[y1:y2, x1:x2], cur[y1:y2, x1:x2]
    if prev.size == 0 or cur.size == 0 or prev.shape != cur.shape:
        return 0.0

    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    h, w = g0.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(prev, gx + flow[..., 0], gy + flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return float(np.abs(warped.astype(np.float32) - cur.astype(np.float32)).mean())


def scene_cut(prev: np.ndarray, cur: np.ndarray, threshold: float = 0.42) -> bool:
    """Detect a hard cut via colour-histogram correlation.

    Smoothing anything across a cut is wrong -- the face, mask and colour state
    all belong to a different shot.
    """
    def hist(img: np.ndarray) -> np.ndarray:
        small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
        h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)],
                         [0, 1], None, [32, 32], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()
    return float(cv2.compareHist(hist(prev), hist(cur), cv2.HISTCMP_CORREL)) < threshold


# ---------------------------------------------------------------- selection
# Priority order from the brief: identity first, speed last.
WEIGHTS = {
    "identity": 0.34,
    "identity_switches": 0.18,
    "temporal": 0.16,
    "blending": 0.12,
    "expression": 0.10,
    "detail": 0.07,
    "speed": 0.03,
}


def composite_score(m: dict) -> tuple[float, dict]:
    """Combine metrics into one ranking number, keeping every component.

    Returns (score, per-component contributions) so a decision can always be
    explained rather than asserted.
    """
    def norm(v: Optional[float], lo: float, hi: float, invert: bool = False) -> float:
        if v is None:
            return 0.0
        x = float(np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
        return 1.0 - x if invert else x

    parts = {
        # 0.2 cosine is a poor swap, 0.65 is excellent.
        "identity": norm(m.get("identity_mean"), 0.20, 0.65),
        "identity_switches": 1.0 if not m.get("identity_switches") else 0.0,
        "temporal": (norm(m.get("identity_stability_mean"), 0.80, 0.99) * 0.5
                     + norm(m.get("flow_flicker_mean"), 2.0, 14.0, invert=True) * 0.3
                     + norm(m.get("mask_jitter_mean"), 0.01, 0.12, invert=True) * 0.2),
        "blending": (norm(m.get("seam_mean"), 0.55, 1.45, invert=True) * 0.6
                     + norm(m.get("color_discontinuity_mean"), 2.0, 22.0, invert=True) * 0.4),
        "expression": norm(m.get("expression_delta_mean"), 0.008, 0.075, invert=True),
        "detail": (norm(m.get("sharpness_mean"), 60.0, 420.0) * 0.5
                   + norm(m.get("texture_retention_mean"), 0.45, 1.25) * 0.5),
        "speed": norm(m.get("ms_per_frame"), 60.0, 1400.0, invert=True),
    }
    contrib = {k: round(parts[k] * WEIGHTS[k], 5) for k in WEIGHTS}
    return round(float(sum(contrib.values())), 5), contrib


def summarise(values: list[Optional[float]]) -> dict[str, Optional[float]]:
    xs = [v for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return {"mean": None, "min": None, "max": None, "std": None, "n": 0}
    a = np.asarray(xs, np.float64)
    return {"mean": round(float(a.mean()), 5), "min": round(float(a.min()), 5),
            "max": round(float(a.max()), 5), "std": round(float(a.std()), 5),
            "n": int(a.size)}
