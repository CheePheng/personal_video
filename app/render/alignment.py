"""Face alignment: warp a detected face onto a model's canonical template.

Every swap/recognition model was trained on faces placed at fixed positions in
the crop. Aligning to the matching template is what lets one identity vector
drive any head pose, and getting the template wrong degrades identity far more
than any amount of post-processing can recover.

Templates are expressed as fractions of the crop, so one table serves every
input resolution.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.render.types import RenderError

# 5-point templates (left eye, right eye, nose, left mouth, right mouth).
TEMPLATES: dict[str, np.ndarray] = {
    "arcface_112_v2": np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097589, 0.82469196],
        [0.63151696, 0.82325089],
    ], dtype=np.float32),
    "arcface_128": np.array([
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453],
    ], dtype=np.float32),
    "ffhq_512": np.array([
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ], dtype=np.float32),
    "styleganex_384": np.array([
        [0.42353745, 0.46026396],
        [0.57725341, 0.45967283],
        [0.50123859, 0.54532654],
        [0.43364461, 0.62796698],
        [0.57015325, 0.62766126],
    ], dtype=np.float32),
}


def estimate_matrix(kps: np.ndarray, template: str, size: int) -> np.ndarray:
    """Similarity transform mapping detected landmarks onto the template.

    A *partial* affine (translation, rotation, uniform scale) rather than a
    full affine: allowing shear would let a bad landmark stretch the face and
    change its geometry, which reads as a distorted identity.
    """
    tpl = TEMPLATES.get(template)
    if tpl is None:
        raise RenderError(f"unknown alignment template: {template}")

    m, _ = cv2.estimateAffinePartial2D(
        kps.astype(np.float32), (tpl * size).astype(np.float32),
        method=cv2.LMEDS)
    if m is None:
        raise RenderError("could not align face (degenerate landmarks)")
    return m.astype(np.float32)


def warp(frame: np.ndarray, kps: np.ndarray, template: str,
         size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (aligned crop, matrix used)."""
    m = estimate_matrix(kps, template, size)
    crop = cv2.warpAffine(frame, m, (size, size),
                          borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LANCZOS4)
    return crop, m


def paste_back(frame: np.ndarray, patch: np.ndarray, mask: np.ndarray,
               matrix: np.ndarray) -> np.ndarray:
    """Composite an aligned patch back into the full frame through a mask.

    Done in float so the blend does not band, and only inside the patch's
    bounding box so a 4K frame does not pay for a full-frame warp per face.
    """
    h, w = frame.shape[:2]
    inv = cv2.invertAffineTransform(matrix)

    back = cv2.warpAffine(patch.astype(np.float32), inv, (w, h),
                          borderMode=cv2.BORDER_TRANSPARENT,
                          flags=cv2.INTER_LANCZOS4)
    back_m = cv2.warpAffine(mask.astype(np.float32), inv, (w, h),
                            flags=cv2.INTER_LINEAR)
    if back_m.ndim == 2:
        back_m = back_m[:, :, None]
    back_m = np.clip(back_m, 0.0, 1.0)

    out = frame.astype(np.float32)
    return np.clip(back * back_m + out * (1.0 - back_m), 0, 255).astype(np.uint8)


def pose_from_kps(kps: np.ndarray) -> tuple[float, float]:
    """Rough (yaw, roll) in degrees from 5 landmarks.

    Used to weight source images and to decide how hard to smooth a transform,
    not for anything that needs true 3D accuracy.
    """
    le, re, nose = kps[0], kps[1], kps[2]
    eye_mid = (le + re) / 2.0
    eye_dist = float(np.linalg.norm(re - le)) or 1.0

    # Nose offset from the eye midpoint, as a fraction of eye distance.
    yaw = float((nose[0] - eye_mid[0]) / eye_dist) * 90.0
    roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    return yaw, roll


class TransformSmoother:
    """Temporally smooth the alignment matrix so the face does not vibrate.

    A detector jitters by a pixel or two per frame; pasted back, that reads as
    a buzzing face. We low-pass the matrix, but adaptively: when the head is
    genuinely moving fast, smoothing would drag the face behind the body, so
    the filter opens up. Motion is measured against the *predicted* matrix, so
    steady fast motion is not mistaken for jitter.
    """

    def __init__(self, base_alpha: float = 0.6, motion_scale: float = 6.0):
        self.base_alpha = base_alpha
        self.motion_scale = motion_scale
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, matrix: np.ndarray, face_size: float) -> np.ndarray:
        if self._prev is None:
            self._prev = matrix.copy()
            return matrix

        # Translation delta in output pixels, normalised by face scale so the
        # threshold means the same thing for a close-up and a distant face.
        delta = float(np.linalg.norm(matrix[:, 2] - self._prev[:, 2]))
        norm = delta / max(face_size, 1.0) * 100.0

        # Little motion -> heavy smoothing. Lots -> follow the detector.
        alpha = self.base_alpha * np.exp(-norm / self.motion_scale)
        alpha = float(np.clip(alpha, 0.0, self.base_alpha))

        out = alpha * self._prev + (1.0 - alpha) * matrix
        self._prev = out.copy()
        return out.astype(np.float32)
