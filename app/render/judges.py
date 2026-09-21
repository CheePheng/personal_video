"""Independent identity judges — deliberately OUT of the optimisation loop.

The pipeline conditions its swap on ArcFace and then scores the result with
the same ArcFace. That is a closed loop: Auto Max ranks candidates on it, the
headline identity numbers are its numbers, and -- most importantly -- the
finding that gated the whole "is 256 px the ceiling?" question (pixel boost
measured harmful, identity fell at every level) is also its number.

Pixel boost changes the crop's frequency content, which is precisely the
regime where a recogniser the pipeline is already optimising toward can
mislead. The drop may be real. It may be partly an artefact. Measuring it
with a second, independently-trained recogniser is the cheapest way to find
out, and it is a prerequisite for trusting anything built on top.

**These judges must never be wired into Auto Max scoring.** A judge that
participates in selection stops being a judge. They exist to verify, after
the fact, what the in-loop metric claimed.

Currently one judge:
  sface  -- OpenCV Zoo SFace, Apache-2.0, 128-d, MobileFaceNet-style backbone
            trained with the SFace loss on a different corpus to ArcFace's
            WebFace600K. Different architecture, different objective,
            different data -- which is what makes it a second opinion rather
            than an echo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.render import alignment, sessions
from app.render.types import ModelSpec, Normalization, RenderError

# SFace consumes the SAME 5-point layout as arcface_112_v2 (OpenCV's
# FaceRecognizerSF template, verified against its published landmark
# destinations), so no new template is needed -- but it takes raw BGR 0-255
# rather than a normalised tensor, which is easy to get wrong silently.
SFACE = ModelSpec(
    name="sface_2021dec", filename="sface_2021dec.onnx", role="recognizer",
    input_size=112, template="arcface_112_v2",
    normalization=Normalization.ZERO_ONE,   # overridden below; see _blob
    license="Apache-2.0 (OpenCV Zoo; weights covered by the repository licence)",
    source_url=("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                "main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
    notes="128-d independent judge. Raw BGR 0-255 CHW input, NOT normalised.")

JUDGES: dict[str, ModelSpec] = {"sface": SFACE}


def available(name: str = "sface") -> bool:
    spec = JUDGES.get(name)
    return bool(spec) and (sessions.MODELS_DIR / spec.filename).is_file()


def _blob(crop: np.ndarray) -> np.ndarray:
    """SFace takes raw BGR 0-255 in CHW -- no scaling, no mean subtraction.

    OpenCV's own wrapper calls blobFromImage with scale 1.0 and swapRB false.
    Feeding it a /255 tensor produces plausible-looking but meaningless
    embeddings, so this is stated rather than inherited from the generic path.
    """
    return crop.transpose(2, 0, 1)[None].astype(np.float32)


def embed(frame: np.ndarray, kps: np.ndarray, name: str = "sface") -> np.ndarray:
    """Independent identity vector, L2-normalised so cosine is a dot product."""
    spec = JUDGES.get(name)
    if spec is None:
        raise RenderError(f"unknown judge '{name}'; available: {sorted(JUDGES)}")

    crop, _ = alignment.warp(frame, kps, spec.template, spec.input_size)
    sess = sessions.get(spec)
    out = sessions.run(spec, {sess.get_inputs()[0].name: _blob(crop)})[0].ravel()
    n = float(np.linalg.norm(out))
    if n < 1e-6:
        raise RenderError(f"judge '{name}' returned an empty embedding")
    return (out / n).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.ravel(), b.ravel()))


def score_pair(source_img: np.ndarray, source_kps: np.ndarray,
               rendered_img: np.ndarray, rendered_kps: np.ndarray,
               name: str = "sface") -> float:
    """Cosine between a source face and a rendered face, per the judge."""
    return similarity(embed(source_img, source_kps, name),
                      embed(rendered_img, rendered_kps, name))


def calibrate(pairs_same: list[float], pairs_diff: list[float]) -> dict[str, Any]:
    """Turn raw cosines into something interpretable.

    A judge's absolute scale is meaningless on its own -- 0.4 means different
    things to ArcFace and SFace. What matters is the separation between
    same-person and different-person pairs. This reports that separation so a
    number can be read as "comfortably the same person" rather than guessed at.
    """
    same = np.asarray([p for p in pairs_same if np.isfinite(p)], np.float64)
    diff = np.asarray([p for p in pairs_diff if np.isfinite(p)], np.float64)
    if same.size == 0 or diff.size == 0:
        return {"usable": False, "reason": "need both same- and different-person pairs"}

    # Midpoint threshold, plus d-prime as a scale-free measure of how well the
    # two populations separate.
    thr = float((same.mean() + diff.mean()) / 2)
    pooled = float(np.sqrt((same.var() + diff.var()) / 2)) or 1e-9
    return {
        "usable": True,
        "same_mean": round(float(same.mean()), 4),
        "same_min": round(float(same.min()), 4),
        "diff_mean": round(float(diff.mean()), 4),
        "diff_max": round(float(diff.max()), 4),
        "threshold": round(thr, 4),
        "separation_dprime": round(abs(same.mean() - diff.mean()) / pooled, 3),
        "clean_split": bool(same.min() > diff.max()),
        "n_same": int(same.size), "n_diff": int(diff.size),
    }
