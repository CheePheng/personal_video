"""The swap itself: identity vector + aligned target face -> new face.

Model-agnostic. Everything that differs between swappers (input size, template,
normalisation, whether the identity vector is pre-normalised, whether the model
emits its own mask) comes from the registry ModelSpec, so adding a swapper does
not touch this file.

**Pixel boost** here is real, not cosmetic. The models are 256px natively;
naively upscaling their output just blurs. Instead we align at the higher
resolution, split that crop into a grid of 256px tiles, run the model on each
tile, and reassemble. Each tile therefore carries genuine model detail at full
resolution. It costs (boost/256)^2 model calls per face, which is why it is a
benchmarked option rather than a default.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.render import alignment, sessions
from app.render.registry import get_model
from app.render.types import ModelSpec, Normalization, RenderError


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def _to_blob(crop: np.ndarray, norm: Normalization) -> np.ndarray:
    blob = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    if norm is Normalization.NEG_ONE_ONE:
        return (blob / 255.0 - 0.5) / 0.5
    if norm is Normalization.ARCFACE:
        return (blob - 127.5) / 127.5
    if norm is Normalization.IMAGENET:
        # simswap_256 is the only model here trained on ImageNet statistics.
        return (blob / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    if norm is Normalization.CENTRED_128:
        return (blob - 127.5) / 128.0
    return blob / 255.0


def _from_blob(out: np.ndarray, norm: Normalization) -> np.ndarray:
    img = out[0].transpose(1, 2, 0)
    if norm in (Normalization.NEG_ONE_ONE, Normalization.ARCFACE):
        img = np.clip(img, -1, 1) * 0.5 + 0.5
    elif norm is Normalization.IMAGENET:
        img = img * _IMAGENET_STD[0].transpose(1, 2, 0) + _IMAGENET_MEAN[0].transpose(1, 2, 0)
    return np.clip(img, 0, 1)[:, :, ::-1] * 255.0        # -> BGR float


def _feed(spec: ModelSpec, sess, blob: np.ndarray,
          embedding: np.ndarray) -> dict[str, np.ndarray]:
    """Build the input dict by inspecting the model's declared inputs.

    Names differ between model families (source/target vs embedding/img), so we
    match by shape rather than assuming a naming convention.
    """
    feeds: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else -1 for d in inp.shape]
        if len(shape) == 4:
            feeds[inp.name] = blob
        elif len(shape) == 2:
            feeds[inp.name] = embedding.astype(np.float32)
        else:
            raise RenderError(
                f"swapper '{spec.name}' has an input '{inp.name}' with "
                f"unsupported shape {inp.shape}")
    return feeds


def _run_tile(spec: ModelSpec, crop: np.ndarray,
              embedding: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    sess = sessions.get(spec)
    blob = _to_blob(crop, spec.normalization)
    outs = sessions.run(spec, _feed(spec, sess, blob, embedding))

    face = _from_blob(outs[0], spec.normalization)
    mask = None
    if spec.outputs_mask and len(outs) > 1:
        m = np.squeeze(outs[1]).astype(np.float32)
        if m.ndim == 3:
            m = m[0]
        mask = np.clip(m, 0.0, 1.0)
    return face, mask


def swap(frame: np.ndarray, kps: np.ndarray, embedding: np.ndarray,
         model: str = "hyperswap_1a_256",
         pixel_boost: int = 0) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, int]:
    """Swap one face.

    Returns (swapped_patch, model_mask_or_None, align_matrix, working_size).
    The patch is float BGR at ``working_size``; compositing is the caller's job
    so masks can be combined first.
    """
    spec = get_model(model)
    if not spec.needs_embedding:
        raise RenderError(f"'{model}' is not an identity-conditioned swapper")

    base = spec.input_size
    size = base
    if pixel_boost and pixel_boost > base:
        if spec.pixel_boost and pixel_boost not in spec.pixel_boost:
            raise RenderError(
                f"'{model}' does not support pixel boost {pixel_boost}; "
                f"supported: {spec.pixel_boost}")
        size = pixel_boost

    crop, matrix = alignment.warp(frame, kps, spec.template, size)

    if size == base:
        face, mask = _run_tile(spec, crop, embedding)
        return face, mask, matrix, size

    # Real pixel boost: tile the high-res crop, run the model per tile.
    n = size // base
    face = np.zeros((size, size, 3), np.float32)
    mask = np.zeros((size, size), np.float32) if spec.outputs_mask else None
    for r in range(n):
        for c in range(n):
            y, x = r * base, c * base
            tile = crop[y:y + base, x:x + base]
            t_face, t_mask = _run_tile(spec, tile, embedding)
            face[y:y + base, x:x + base] = t_face
            if mask is not None and t_mask is not None:
                mask[y:y + base, x:x + base] = t_mask
    return face, mask, matrix, size


def prepare_embedding(spec_name: str, identity: np.ndarray) -> np.ndarray:
    """Shape the source identity the way a given swapper expects it.

    Three conventions exist among the registered families, and getting this
    wrong produces a plausible-looking face that is simply the wrong person:

      * hyperswap  -- the L2-normalised ArcFace vector
      * alphaface  -- the RAW, unnormalised ArcFace vector
      * ghost / simswap -- ArcFace passed through a learned 512->512 converter
        into that family's own identity space, then optionally re-normalised

    Everything is read from the ModelSpec, so a new family is a registry entry.
    """
    spec = get_model(spec_name)
    vec = identity.reshape(1, -1).astype(np.float32)

    if spec.embedding_converter:
        conv = get_model(spec.embedding_converter)
        sess = sessions.get(conv)
        name = sess.get_inputs()[0].name
        vec = sessions.run(conv, {name: vec})[0].reshape(1, -1).astype(np.float32)
        if spec.converter_normalize:
            n = float(np.linalg.norm(vec))
            if n > 1e-6:
                vec = vec / n
        return vec

    if spec.embedding_normalized:
        n = float(np.linalg.norm(vec))
        if n > 1e-6:
            vec = vec / n
    return vec
