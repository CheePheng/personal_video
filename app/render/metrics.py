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

    # Farneback is CPU-bound and scales with area. On a 4K close-up the face
    # crop alone can be ~700px square, and the benchmark runs this for every
    # sampled frame of every candidate -- hundreds of times. Flicker is a
    # RELATIVE measure, so computing it on a bounded copy gives the same
    # ranking for a fraction of the cost. Measured: this was the single
    # largest contributor to Auto Max wall time, above all GPU inference.
    FLOW_MAX = 192
    h0, w0 = prev.shape[:2]
    scale = min(1.0, FLOW_MAX / max(h0, w0))
    if scale < 1.0:
        size = (max(16, int(w0 * scale)), max(16, int(h0 * scale)))
        prev = cv2.resize(prev, size, interpolation=cv2.INTER_AREA)
        cur = cv2.resize(cur, size, interpolation=cv2.INTER_AREA)

    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    h, w = g0.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(prev, gx + flow[..., 0], gy + flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return float(np.abs(warped.astype(np.float32) - cur.astype(np.float32)).mean())


def _small(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)


def _hist_of_small(small: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)],
                     [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _hist(img: np.ndarray) -> np.ndarray:
    return _hist_of_small(_small(img))


def _mad(prev: np.ndarray, cur: np.ndarray) -> float:
    """Mean absolute difference on a small greyscale copy."""
    a = cv2.cvtColor(cv2.resize(prev, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(cur, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def scene_cut(prev: np.ndarray, cur: np.ndarray, threshold: float = 0.42) -> bool:
    """Stateless cut test on colour distribution alone.

    Kept for callers that have no history. Prefer SceneCutDetector: a histogram
    cannot see a cut between two shots that share a palette, which is common
    (same room, same lighting, different framing).
    """
    return float(cv2.compareHist(_hist(prev), _hist(cur), cv2.HISTCMP_CORREL)) < threshold


class SceneCutDetector:
    """Detect hard cuts from BOTH colour distribution and frame content.

    Histogram correlation alone is not enough: two shots filmed in the same
    room have near-identical colour distributions, so a real cut between them
    barely moves the correlation even though every pixel changed. Conversely,
    raw pixel difference alone fires on fast camera motion.

    So we use each for what it is good at:
      * a large drop in histogram correlation  -> clearly a different scene
      * a spike in pixel difference *relative to this clip's recent motion*
        -> the content changed abruptly, whatever the palette

    The second test is adaptive, which is what makes it safe on handheld
    footage: a clip that is always moving has a high baseline, so only a jump
    well above that baseline counts.
    """

    def __init__(self, hist_threshold: float = 0.55, mad_floor: float = 8.0,
                 mad_ratio: float = 3.0, window: int = 24, warmup: int = 3):
        self.hist_threshold = hist_threshold
        self.mad_floor = mad_floor
        self.mad_ratio = mad_ratio
        self.window = window
        # The spike test compares against this clip's own motion baseline, so
        # it is meaningless until a baseline exists. Before then, ordinary
        # camera movement on frame 2 would look like a spike above zero and
        # fire a false cut. Until warmup completes we trust the histogram only.
        self.warmup = warmup
        self._recent: list[float] = []
        self.cuts = 0
        # Successive calls pass (frame N-1, frame N) and then (frame N,
        # frame N+1), so every frame was fully downsampled twice -- once as
        # "cur", once as "prev" -- and the histogram and greyscale copies
        # each re-did that downsample independently. Four full-frame resizes
        # per frame, measured at 15.6 ms/frame on 4K. One derivation per
        # frame, cached, is arithmetically identical.
        self._prev_ref: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._recent.clear()
        self._prev_ref = None
        self._prev_hist = None
        self._prev_gray = None

    @staticmethod
    def _derive(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        small = _small(frame)
        return (_hist_of_small(small),
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

    def __call__(self, prev: np.ndarray, cur: np.ndarray) -> bool:
        # A held reference keeps the cached id() unambiguous: without it the
        # array could be freed and a different array reuse the address.
        if self._prev_hist is None or self._prev_ref is not prev:
            self._prev_hist, self._prev_gray = self._derive(prev)
        cur_hist, cur_gray = self._derive(cur)

        corr = float(cv2.compareHist(self._prev_hist, cur_hist, cv2.HISTCMP_CORREL))
        assert self._prev_gray is not None
        mad = float(np.abs(self._prev_gray.astype(np.int16)
                           - cur_gray.astype(np.int16)).mean())

        self._prev_ref, self._prev_hist, self._prev_gray = cur, cur_hist, cur_gray

        have_baseline = len(self._recent) >= self.warmup
        baseline = float(np.median(self._recent)) if have_baseline else 0.0
        spike = (have_baseline and mad >= self.mad_floor
                 and mad >= self.mad_ratio * max(baseline, 1.0))
        cut = corr < self.hist_threshold or spike

        self._recent.append(mad)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        if cut:
            self.cuts += 1
            # A cut makes the pre-cut motion history meaningless.
            self._recent = [mad]
        return cut


# ---------------------------------------------------------------- selection
# Weighted for the SINGLE-PERSON use case: one source identity, one person in
# the target video. "Which person should I swap?" is trivial here, so the
# score is spent almost entirely on "how good does this one face look?".
#
# identity_switches drops from 0.18 to 0.02. It is kept, not deleted, because
# a collapse to zero would remove the only guard against a detector rescue or
# a reflection pulling the swap onto something that is not the subject -- but
# on single-person footage it should almost never fire, and 18% of the score
# riding on a term that is constant is exactly the dead-weight problem we hit
# before.
#
# identity_worst is NEW and weighted heavily. A pipeline with a good mean and
# an ugly worst frame is worse to watch than one that is merely good
# throughout: the eye lands on the bad frames.
WEIGHTS = {
    "identity": 0.30,          # mean similarity to the source face
    "identity_worst": 0.14,    # the worst frame, not just the average
    "temporal": 0.18,          # frame-to-frame stability / flicker
    "expression": 0.13,        # the target keeps their own performance
    "blending": 0.13,          # seam + colour continuity (incl. occlusion)
    "detail": 0.08,            # sharpness / texture
    "identity_switches": 0.02, # defensive only on single-person footage
    "speed": 0.02,
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
        # identity_min falls back to the mean when a run recorded only one
        # frame, so a short benchmark is not silently penalised.
        "identity_worst": norm(
            m.get("identity_min", m.get("identity_mean")), 0.25, 0.93),
        # Identity range must span what the CURRENT models actually produce,
        # or the highest-weighted term silently stops discriminating.
        # The original 0.20-0.65 was calibrated before the swapper set grew:
        # every registered model now scores 0.60-0.98 on a clear face, so both
        # a 0.77 candidate and a 0.96 one saturated at full marks and the
        # decision fell through to blending and speed. That is how a 0.77
        # pipeline once beat a 0.96 one. 0.30-0.95 keeps a real gradient
        # across the range these models occupy.
        "identity": norm(m.get("identity_mean"), 0.30, 0.95),
        "identity_switches": 1.0 if not m.get("identity_switches") else 0.0,
        "temporal": (norm(m.get("identity_stability_mean"), 0.80, 0.99) * 0.5
                     + norm(m.get("flow_flicker_mean"), 2.0, 14.0, invert=True) * 0.3
                     + norm(m.get("mask_jitter_mean"), 0.01, 0.12, invert=True) * 0.2),
        "blending": (norm(m.get("seam_mean"), 0.55, 1.45, invert=True) * 0.6
                     + norm(m.get("color_discontinuity_mean"), 2.0, 22.0, invert=True) * 0.4),
        # Range tightened from 0.008-0.075 to what real swappers actually
        # produce. Across every model measured on the A->B set the delta
        # spans 0.011-0.021, which sat in the top fifth of the old range and
        # compressed to 0.015 of weighted score -- while the top five Auto
        # Max candidates were separated by 0.0025. A swapper that distorts
        # jaw opening 4x more than another was therefore indistinguishable
        # to the selector, and Auto Max duly picked a GHOST variant whose
        # open smiles come back closed and grimacing. On 0.010-0.022 the
        # same spread is 0.65 normalised, so the term can carry its weight.
        "expression": norm(m.get("expression_delta_mean"), 0.010, 0.022, invert=True),
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
