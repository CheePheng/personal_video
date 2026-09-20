"""Face detection.

Detection quality sets a ceiling on everything downstream: a face that is
never found is never swapped, and landmarks that wobble make the swapped face
wobble with them. Two things matter here beyond calling the model:

  * **Adaptive resolution.** Detecting a 1080p or 4K frame at a fixed 640px
    throws away the pixels that make a small or distant face findable. We scale
    the detector input to the frame, within limits.

  * **A real fallback.** If the primary detector finds nothing on a hard frame
    we retry at higher resolution and a lower threshold before giving up --
    but we never invent a face, and we never hand a different person to the
    tracker just because the real one was missed.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.render import sessions
from app.render.registry import get_model
from app.render.types import Face


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Fit a frame into size x size, preserving aspect. Returns (canvas, scale)."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:rh, :rw] = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale


def adaptive_size(frame_w: int, frame_h: int, quality: str = "quality",
                  model: str = "yoloface_8n") -> int:
    """Pick a detector input size for this frame.

    Bigger input finds smaller faces but costs quadratically. Only models with
    a dynamic input axis can actually be run at another size -- yoloface_8n is
    compiled to a fixed 640x640, so asking for more is an error, not an
    optimisation. We therefore check the ONNX input shape rather than assume.
    """
    spec = get_model(model)
    if not _supports_dynamic_input(spec):
        return spec.input_size

    longest = max(frame_w, frame_h)
    if quality == "fast":
        return 640
    if longest >= 2160:
        return 1280
    if longest >= 1440:
        return 1024
    if longest >= 1080:
        return 960
    return 640


def _supports_dynamic_input(spec) -> bool:
    """True when the model's H/W axes are symbolic rather than fixed."""
    try:
        shape = sessions.get(spec).get_inputs()[0].shape
    except Exception:  # noqa: BLE001
        return False
    return len(shape) == 4 and not all(isinstance(d, int) for d in shape[2:])


def detect(frame: np.ndarray, threshold: float = 0.5,
           size: Optional[int] = None, model: str = "yoloface_8n") -> list[Face]:
    """Detect faces in a BGR frame."""
    spec = get_model(model)
    if size is None:
        size = spec.input_size

    canvas, scale = _letterbox(frame, size)
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    out = sessions.run(spec, {"input": blob})[0]
    det = np.squeeze(out).T                       # (N, 4 + 1 + 15)
    if det.ndim != 2 or det.shape[0] == 0:
        return []
    boxes_raw, scores_raw, kps_raw = np.split(det, [4, 5], axis=1)
    scores_raw = scores_raw.ravel()

    keep = np.where(scores_raw > threshold)[0]
    if keep.size == 0:
        return []
    boxes_raw, scores_raw, kps_raw = boxes_raw[keep], scores_raw[keep], kps_raw[keep]

    inv = 1.0 / scale
    faces: list[Face] = []
    for b, s, k in zip(boxes_raw, scores_raw, kps_raw):
        cx, cy, bw, bh = b
        box = np.array([(cx - bw / 2) * inv, (cy - bh / 2) * inv,
                        (cx + bw / 2) * inv, (cy + bh / 2) * inv], dtype=np.float32)
        # Landmarks arrive interleaved as (x, y, visibility).
        kps = np.stack([k[0::3] * inv, k[1::3] * inv], axis=1).astype(np.float32)
        faces.append(Face(box=box, kps=kps, score=float(s)))

    return _nms(faces, threshold)


def _nms(faces: list[Face], threshold: float, iou: float = 0.4) -> list[Face]:
    """The detector fires several times per face; keep one box each."""
    if not faces:
        return []
    rects = [[float(f.box[0]), float(f.box[1]),
              float(f.box[2] - f.box[0]), float(f.box[3] - f.box[1])] for f in faces]
    idx = cv2.dnn.NMSBoxes(rects, [f.score for f in faces], threshold, iou)
    if len(idx) == 0:
        return []
    return [faces[i] for i in np.array(idx).ravel()]


def detect_robust(frame: np.ndarray, threshold: float = 0.5,
                  quality: str = "quality", model: str = "yoloface_8n") -> list[Face]:
    """Detect with an escalating retry for difficult frames.

    Motion blur, profiles and small faces often sit just under the threshold at
    the default resolution. Rather than accept a dropped frame we retry once,
    larger and more sensitive. This is a *recall* fallback, not a licence to
    return a lower-confidence face as if it were certain -- the score travels
    with the Face so the tracker can weigh it.
    """
    h, w = frame.shape[:2]
    size = adaptive_size(w, h, quality, model)

    faces = detect(frame, threshold, size, model)
    if faces or quality == "fast":
        return faces

    # Retry more sensitively. On a fixed-input model we cannot raise the
    # resolution, so lowering the threshold is the only lever -- the score
    # still travels with each Face so the tracker can weigh it.
    bigger = size
    if _supports_dynamic_input(get_model(model)):
        bigger = min(1280, int(size * 1.5) // 32 * 32)
    return detect(frame, max(0.25, threshold * 0.6), bigger, model)
