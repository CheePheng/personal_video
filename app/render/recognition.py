"""Face recognition: turn a face into a 512-d identity vector.

This module is load-bearing for three separate jobs, which is why it lives on
its own rather than inside the swapper:

  * building the source identity that drives the swap
  * deciding which detected face is *our* person, frame after frame (tracking)
  * scoring how well a rendered face preserved that identity (benchmarking)

Vectors are always L2-normalised, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from app.render import alignment, sessions
from app.render.registry import get_model
from app.render.types import Face, Normalization, RenderError, SourceIdentity


def _prep(crop: np.ndarray, norm: Normalization) -> np.ndarray:
    blob = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    if norm is Normalization.ARCFACE:
        return (blob - 127.5) / 127.5
    if norm is Normalization.NEG_ONE_ONE:
        return (blob / 255.0 - 0.5) / 0.5
    return blob / 255.0


def embed(frame: np.ndarray, kps: np.ndarray, model: str = "arcface_w600k_r50") -> np.ndarray:
    """512-d L2-normalised identity vector for one face."""
    spec = get_model(model)
    crop, _ = alignment.warp(frame, kps, spec.template, spec.input_size)
    vec = sessions.run(spec, {"input": _prep(crop, spec.normalization)})[0].ravel()
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        raise RenderError("face embedding was empty (degenerate crop)")
    return (vec / n).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised embeddings, in [-1, 1]."""
    return float(np.dot(a.ravel(), b.ravel()))


# ---------------------------------------------------------------- source ID
def _blur_score(crop: np.ndarray) -> float:
    """Laplacian variance -- higher is sharper."""
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def assess_source_face(frame: np.ndarray, face: Face) -> dict[str, Any]:
    """Quality signals used to weight one source photo.

    A blurry, tiny or extremely profile face carries less reliable identity
    than a sharp frontal one, so it should contribute less to the fused
    embedding -- but it should still contribute, because an off-angle photo is
    exactly what helps the swap hold up when the target turns their head.
    """
    crop, _ = alignment.warp(frame, face.kps, "arcface_112_v2", 112)
    yaw, roll = alignment.pose_from_kps(face.kps)
    return {
        "size": face.size,
        "detector_score": face.score,
        "blur": _blur_score(crop),
        "yaw": yaw,
        "roll": roll,
    }


def build_source_identity(images: list[tuple[str, np.ndarray, Face]],
                          model: str = "arcface_w600k_r50") -> SourceIdentity:
    """Fuse 1..N source photos of the same person into one identity vector.

    Weighting, rather than a plain mean, because a plain mean lets one bad
    photo drag the identity. Weight combines detector confidence, face size,
    sharpness and frontality. Vectors are re-normalised after summing so the
    result stays a unit vector.

    A photo whose identity disagrees sharply with the consensus is dropped --
    that is almost always a different person in a group shot, and averaging it
    in would produce a face that resembles neither.
    """
    if not images:
        raise RenderError("no usable source faces")

    entries = []
    for name, frame, face in images:
        q = assess_source_face(frame, face)
        try:
            vec = embed(frame, face.kps, model)
        except RenderError:
            continue

        # Each factor in [0,1]-ish; multiplied so any single disqualifier
        # (tiny face, mush, extreme profile) suppresses the whole weight.
        w_size = min(1.0, q["size"] / 160.0)
        w_conf = float(np.clip(q["detector_score"], 0.0, 1.0))
        w_blur = min(1.0, q["blur"] / 120.0)
        w_pose = float(np.exp(-abs(q["yaw"]) / 45.0))
        weight = max(1e-3, w_size * w_conf * w_blur * (0.4 + 0.6 * w_pose))

        entries.append({"name": name, "vec": vec, "weight": weight, **q})

    if not entries:
        raise RenderError("no usable source faces after quality filtering")

    # Consensus check: compare each photo to the unweighted mean direction.
    mean = np.mean([e["vec"] for e in entries], axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-6)
    for e in entries:
        e["consensus"] = similarity(e["vec"], mean)

    if len(entries) >= 3:
        agree = [e for e in entries if e["consensus"] >= 0.35]
        if agree:
            entries = agree

    total = sum(e["weight"] for e in entries)
    fused = np.sum([e["vec"] * (e["weight"] / total) for e in entries], axis=0)
    fused /= max(float(np.linalg.norm(fused)), 1e-6)

    return SourceIdentity(
        embedding=fused.reshape(1, -1).astype(np.float32),
        per_image=[{k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in e.items() if k != "vec"} for e in entries],
        n_images=len(entries),
    )
