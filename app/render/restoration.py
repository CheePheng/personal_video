"""Face restoration, with identity treated as the thing to protect.

The swappers output 256px. On a close-up that is an upscale by the time it is
pasted back, which reads as soft. Restorers (GFPGAN, CodeFormer, GPEN,
RestoreFormer++) re-synthesise the face at 512px and recover convincing detail.

The catch, and the reason this is benchmarked rather than always-on: a restorer
is a generative model with its own idea of what a face looks like. At full
strength it pulls the result toward that prior and *away* from the identity we
just swapped in. Sharper and less like the person is not an improvement.

So: blend is a first-class parameter, the benchmark measures identity at each
blend level, and configurations that trade too much identity for sharpness are
rejected. Strength also scales with face size -- a 40px face needs far less
than a 600px close-up, and over-restoring a tiny face invents detail that
flickers between frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.render import alignment, sessions
from app.render.registry import get_model
from app.render.types import Normalization, RenderError

BLEND_CANDIDATES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _prep(crop: np.ndarray, norm: Normalization) -> np.ndarray:
    blob = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    if norm is Normalization.NEG_ONE_ONE:
        return (blob / 255.0 - 0.5) / 0.5
    return blob / 255.0


def _post(out: np.ndarray, norm: Normalization) -> np.ndarray:
    img = out[0].transpose(1, 2, 0)
    if norm is Normalization.NEG_ONE_ONE:
        img = np.clip(img, -1, 1) * 0.5 + 0.5
    return np.clip(img, 0, 1)[:, :, ::-1] * 255.0


def adaptive_blend(base_blend: float, face_size: float) -> float:
    """Scale restoration strength by how large the face actually is.

    Small faces get less: there is little real detail to recover, so a strong
    restorer is mostly inventing texture, and invented texture differs frame to
    frame -- which is visible as shimmer even when each frame looks fine alone.
    """
    if face_size < 64:
        return base_blend * 0.35
    if face_size < 128:
        return base_blend * 0.65
    if face_size > 400:
        return min(1.0, base_blend * 1.15)
    return base_blend


def restore(frame: np.ndarray, kps: np.ndarray, model: str,
            blend: float = 0.8, face_size: float = 128.0,
            adaptive: bool = True) -> np.ndarray:
    """Restore one face in a full frame; returns the full frame."""
    if blend <= 0.0:
        return frame

    spec = get_model(model)
    if spec.role != "enhancer":
        raise RenderError(f"'{model}' is not an enhancer")

    effective = adaptive_blend(blend, face_size) if adaptive else blend
    if effective <= 0.001:
        return frame

    crop, matrix = alignment.warp(frame, kps, spec.template, spec.input_size)
    sess = sessions.get(spec)

    feeds: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        if len(inp.shape) == 4:
            feeds[inp.name] = _prep(crop, spec.normalization)
        else:
            # CodeFormer takes a fidelity weight: 0 = prettier, 1 = truer to
            # input. We want identity, so we ask for fidelity.
            feeds[inp.name] = np.array([0.85], np.float64)

    out = sessions.run(spec, feeds)[0]
    restored = _post(out, spec.normalization)

    # Blend restored against the original crop, then composite through a soft
    # oval so the restored square never shows an edge.
    merged = restored * effective + crop.astype(np.float32) * (1.0 - effective)

    size = spec.input_size
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, size // 2),
                (int(size * 0.44), int(size * 0.54)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (31, 31), 0)

    return alignment.paste_back(frame, merged, m, matrix)
