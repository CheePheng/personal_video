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
    # The v1 template predates v2 and is what the GHOST and SimSwap families
    # were trained against. Using v2 for them misaligns every face slightly,
    # which reads as a subtly wrong identity rather than an obvious error.
    "arcface_112_v1": np.array([
        [0.35473214, 0.45658929],
        [0.64526786, 0.45658929],
        [0.50000000, 0.61154464],
        [0.37913393, 0.77687500],
        [0.62086607, 0.77687500],
    ], dtype=np.float32),
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
    "mtcnn_512": np.array([
        [0.36562865, 0.46733799],
        [0.63305391, 0.46585885],
        [0.50019127, 0.61942959],
        [0.39032951, 0.77598822],
        [0.61178945, 0.77476328],
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

    That bounding box matters more than it looks. This function used to warp
    the patch and the mask into full-frame float buffers and blend across
    every pixel, which was measured at 164 ms/frame on 4K -- 52% of the whole
    render -- to composite a face occupying a few percent of the image. The
    ROI form is arithmetically identical (warpAffine inverse-maps per output
    pixel, so translating the destination translates the result exactly) and
    the pixels outside the box were provably unchanged anyway: the mask is
    zero there, so the old blend returned ``frame`` verbatim.
    """
    h, w = frame.shape[:2]
    inv = cv2.invertAffineTransform(matrix)

    # Where does the patch actually land? Map its corners into frame space.
    ph, pw = patch.shape[:2]
    corners = np.array([[0.0, 0.0], [pw, 0.0], [pw, ph], [0.0, ph]], np.float32)
    mapped = corners @ inv[:, :2].T + inv[:, 2]
    # One pixel of slack for the interpolation footprint, then clip to frame.
    x0 = max(0, int(np.floor(mapped[:, 0].min())) - 1)
    y0 = max(0, int(np.floor(mapped[:, 1].min())) - 1)
    x1 = min(w, int(np.ceil(mapped[:, 0].max())) + 1)
    y1 = min(h, int(np.ceil(mapped[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return frame            # face maps entirely outside the frame

    # Shift the destination origin to the ROI instead of cropping afterwards.
    inv_roi = inv.copy()
    inv_roi[0, 2] -= x0
    inv_roi[1, 2] -= y0
    roi = (x1 - x0, y1 - y0)

    # BORDER_TRANSPARENT leaves untouched destination pixels ALONE, so the
    # destination must be initialised. Letting warpAffine allocate it left
    # those pixels as uninitialised memory; the mask is zero there, but
    # 0 * NaN is NaN, not 0, so stray garbage could reach the output and the
    # result differed between runs. Allocating zeros makes it deterministic.
    # (h, w) + trailing channel dims, so a 2-D patch is handled too.
    back = np.zeros((roi[1], roi[0]) + patch.shape[2:], np.float32)
    cv2.warpAffine(patch.astype(np.float32), inv_roi, roi, dst=back,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.INTER_LANCZOS4)
    back_m = cv2.warpAffine(mask.astype(np.float32), inv_roi, roi,
                            flags=cv2.INTER_LINEAR)
    if back_m.ndim == 2:
        back_m = back_m[:, :, None]
    back_m = np.clip(back_m, 0.0, 1.0)

    out = frame.copy()
    region = frame[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(
        back * back_m + region * (1.0 - back_m), 0, 255).astype(np.uint8)
    return out


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
