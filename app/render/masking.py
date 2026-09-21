"""Face masks: decide which pixels of the swapped face actually get pasted.

A crude oval is what makes a swap look pasted-on. Three sources are combined,
each answering a different question:

  * **box mask** -- where is the crop border? Feathered so the rectangle edge
    never shows.
  * **parsing mask** (BiSeNet, 19 classes) -- which pixels are *face*, as
    opposed to hair, glasses, hat, neck or background? This is what puts hair
    back in front of the forehead.
  * **occlusion mask** (XSeg) -- is something in front of the face that is not
    part of it: a hand, a microphone, an object crossing frame?

The union of what to exclude is intersected with what to include, so a hand
over the cheek leaves the original pixels untouched and the replacement face
appears genuinely *behind* the occluder.

Masks are then temporally smoothed: a mask that changes shape every frame makes
the seam crawl even when each individual frame looks fine.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.render import alignment, sessions
from app.render.registry import get_model
from app.render.types import Normalization, RenderError

# BiSeNet's 19 classes. Index -> what it is.
#  0 background      1 skin         2 l-brow    3 r-brow    4 l-eye
#  5 r-eye           6 glasses      7 l-ear     8 r-ear     9 earring
# 10 nose           11 mouth       12 u-lip    13 l-lip    14 neck
# 15 necklace       16 cloth       17 hair     18 hat
FACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)          # the swappable face
OCCLUDER_CLASSES = (6, 9, 15, 16, 17, 18)               # in front of / not face


# BiSeNet was trained with ImageNet statistics, not the [-1,1] convention the
# swappers use. Feeding it the wrong normalisation yields a plausible-looking
# but subtly wrong segmentation -- the worst kind of bug, because the mask
# still has roughly the right shape.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def _prep_parser(crop: np.ndarray, spec) -> np.ndarray:
    """NCHW, normalised as the model's spec declares.

    Read from the ModelSpec rather than hard-coded, so the registry stays the
    single source of truth and cannot silently disagree with this code.
    """
    blob = crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    if spec.normalization is Normalization.IMAGENET:
        return ((blob - IMAGENET_MEAN) / IMAGENET_STD)[None]
    if spec.normalization is Normalization.NEG_ONE_ONE:
        return ((blob - 0.5) / 0.5)[None]
    return blob[None]


def _prep_xseg(crop: np.ndarray) -> np.ndarray:
    """NHWC, plain /255 -- XSeg keeps channels last, unlike every other model."""
    return (crop[:, :, ::-1].astype(np.float32) / 255.0)[None]


def box_mask(size: int, padding: float = 0.06, blur: float = 0.10) -> np.ndarray:
    """Feathered rectangle: hides the crop boundary itself."""
    m = np.ones((size, size), np.float32)
    pad = max(1, int(size * padding))
    m[:pad, :] = 0
    m[-pad:, :] = 0
    m[:, :pad] = 0
    m[:, -pad:] = 0
    k = max(3, int(size * blur) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def oval_mask(size: int, feather: float = 0.08) -> np.ndarray:
    """Fallback when no parser is available."""
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, size // 2),
                (int(size * 0.42), int(size * 0.52)), 0, 0, 360, 1.0, -1)
    k = max(3, int(size * feather) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def parsing_masks(frame: np.ndarray, kps: np.ndarray,
                  model: str = "bisenet_resnet_34") -> tuple[np.ndarray, np.ndarray]:
    """Semantic face/occluder masks, both in the SWAP crop's coordinate frame.

    Returns (face, occluder), each float32 in [0,1] at the parser's resolution.
    """
    spec = get_model(model)
    crop, _ = alignment.warp(frame, kps, spec.template, spec.input_size)
    out = sessions.run(spec, {"input": _prep_parser(crop, spec)})[0]

    labels = np.argmax(out[0], axis=0).astype(np.int32)
    face = np.isin(labels, FACE_CLASSES).astype(np.float32)
    occl = np.isin(labels, OCCLUDER_CLASSES).astype(np.float32)
    return face, occl


def occlusion_mask(frame: np.ndarray, kps: np.ndarray,
                   model: str = "xseg_1") -> np.ndarray:
    """XSeg: 1 where the face is visible, 0 where something covers it."""
    spec = get_model(model)
    crop, _ = alignment.warp(frame, kps, spec.template, spec.input_size)
    out = sessions.run(spec, {"input": _prep_xseg(crop)})[0]
    # Output is NHWC (N,256,256,1) -- squeeze rather than index a channel dim
    # that is not where NCHW code would expect it.
    return np.clip(np.squeeze(out).astype(np.float32), 0.0, 1.0)


def _fit(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] != size or mask.shape[1] != size:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask.astype(np.float32), 0.0, 1.0)


def build(frame: np.ndarray, kps: np.ndarray, size: int,
          model_mask: Optional[np.ndarray] = None,
          use_parsing: bool = True, use_occlusion: bool = True,
          face_size: float = 128.0,
          erode: int = 0, dilate: int = 0,
          padding: float = 0.06) -> tuple[np.ndarray, dict[str, bool]]:
    """Combine every available mask source into one blend mask.

    ``face_size`` drives adaptive feathering: a 40px face and a 600px close-up
    should not get the same absolute blur, or the small one dissolves and the
    large one shows a hard edge.
    """
    used = {"box": True, "model": False, "parsing": False, "occlusion": False}
    mask = box_mask(size, padding=padding)

    if model_mask is not None:
        mask = mask * _fit(model_mask, size)
        used["model"] = True

    if use_parsing:
        try:
            face, occl = parsing_masks(frame, kps)
            mask = mask * _fit(face, size) * (1.0 - _fit(occl, size))
            used["parsing"] = True
        except RenderError:
            pass          # parser unavailable: other sources still apply

    if use_occlusion:
        try:
            mask = mask * _fit(occlusion_mask(frame, kps), size)
            used["occlusion"] = True
        except RenderError:
            pass

    if not used["model"] and not used["parsing"]:
        mask = mask * oval_mask(size)

    if erode > 0:
        mask = cv2.erode(mask, np.ones((erode, erode), np.uint8))
    if dilate > 0:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))

    # Feather proportional to how big this face actually is on screen.
    k = int(np.clip(face_size * 0.06, 3, 41))
    mask = cv2.GaussianBlur(mask, (k | 1, k | 1), 0)
    return np.clip(mask, 0.0, 1.0), used


class MaskSmoother:
    """Temporally stabilise a mask so its boundary stops crawling.

    Straight EMA in mask space. Shape changes that persist (a hand arriving)
    come through within a few frames; per-frame parser noise averages out.
    Reset on a scene cut, where carrying a mask over is simply wrong.
    """

    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        if self._prev is None or self._prev.shape != mask.shape:
            self._prev = mask.copy()
            return mask
        out = self.alpha * self._prev + (1.0 - self.alpha) * mask
        self._prev = out
        return out

    def jitter(self, mask: np.ndarray) -> float:
        """Mean absolute change vs the previous mask -- a flicker measure."""
        if self._prev is None or self._prev.shape != mask.shape:
            return 0.0
        return float(np.abs(mask - self._prev).mean())
