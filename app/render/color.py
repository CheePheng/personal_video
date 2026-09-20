"""Colour and lighting matching, confined to the face.

A swapped face carries the source photo's lighting and white balance. Dropped
into a differently-lit shot it reads as a sticker. We correct it in LAB space,
matching the mean/std of the swapped face to the target face it replaces:

  * L carries luminance -- the dominant cue, and the one that must follow the
    scene's lighting.
  * a/b carry chroma -- corrected more gently, because pushing them hard
    recolours the person and starts eroding the identity we just swapped in.

Statistics are computed **only inside the mask**, so background pixels in the
crop cannot drag the correction, and only the face is ever altered -- the rest
of the frame is untouched by construction.

Parameters are smoothed over time: recomputing per frame makes the face pulse
as the target's exposure wobbles.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Chroma is corrected less aggressively than luminance, on purpose.
L_STRENGTH = 1.0
AB_STRENGTH = 0.65
# Beyond this the correction is almost certainly fighting a bad mask.
MAX_SHIFT = 28.0


def _masked_stats(lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = mask.reshape(-1, 1).astype(np.float64)
    total = float(w.sum())
    if total < 16.0:
        return np.zeros(3), np.ones(3)
    flat = lab.reshape(-1, 3).astype(np.float64)
    mean = (flat * w).sum(0) / total
    var = (((flat - mean) ** 2) * w).sum(0) / total
    return mean, np.sqrt(np.maximum(var, 1e-6))


def match(swapped: np.ndarray, target: np.ndarray, mask: np.ndarray,
          params: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Recolour ``swapped`` to sit in ``target``'s lighting, inside ``mask``.

    Returns (corrected, params) where params can be fed back in (already
    smoothed) on the next frame to keep the correction stable.
    """
    s_lab = cv2.cvtColor(np.clip(swapped, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(np.clip(target, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    if params is None:
        s_mean, s_std = _masked_stats(s_lab, mask)
        t_mean, t_std = _masked_stats(t_lab, mask)
        gain = np.clip(t_std / np.maximum(s_std, 1e-3), 0.6, 1.6)
        shift = np.clip(t_mean - s_mean * gain, -MAX_SHIFT, MAX_SHIFT)
        strength = np.array([L_STRENGTH, AB_STRENGTH, AB_STRENGTH])
        gain = 1.0 + (gain - 1.0) * strength
        shift = shift * strength
        params = {"gain": gain.tolist(), "shift": shift.tolist()}

    gain = np.asarray(params["gain"], np.float32)
    shift = np.asarray(params["shift"], np.float32)
    out_lab = np.clip(s_lab * gain + shift, 0, 255).astype(np.uint8)
    corrected = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # Only apply where the mask says face; elsewhere keep the original pixels.
    m = mask[:, :, None] if mask.ndim == 2 else mask
    return swapped * (1.0 - m) + corrected * m, params


class ColorSmoother:
    """EMA over correction parameters to stop the face pulsing frame to frame."""

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        self._p: Optional[dict] = None

    def reset(self) -> None:
        self._p = None

    def __call__(self, params: dict) -> dict:
        if self._p is None:
            self._p = {k: np.asarray(v, np.float32) for k, v in params.items()}
        else:
            for k, v in params.items():
                self._p[k] = self.alpha * self._p[k] + (1 - self.alpha) * np.asarray(v, np.float32)
        return {k: v.tolist() for k, v in self._p.items()}
