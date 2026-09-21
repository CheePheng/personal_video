"""Anchor-based detector decoding: SCRFD and RetinaFace.

YOLOFace emits one dense tensor that decodes trivially. The InsightFace
detectors instead emit nine tensors -- three feature-map strides, each with
its own scores, box distances and landmark distances -- and every value is a
*distance from an anchor centre*, in stride units. Nothing about that is
guessable from the tensor shapes alone, which is why these two sat registered
but unwired.

Layout (both families):
    outputs[0:3]  scores  per stride 8, 16, 32   -> (N, 1)
    outputs[3:6]  bboxes  per stride             -> (N, 4)  l, t, r, b
    outputs[6:9]  kps     per stride             -> (N, 10) 5 x (dx, dy)

with N = (H/stride) * (W/stride) * anchors_per_cell.

Anchor centres are generated in row-major order and repeated per anchor, so
the flattening has to match exactly or every box lands in the wrong place.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from app.render import sessions
from app.render.registry import get_model
from app.render.types import Face, Normalization

STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2          # both SCRFD 2.5g and RetinaFace 10g


@lru_cache(maxsize=64)
def _anchor_centres(h: int, w: int, stride: int, anchors: int) -> np.ndarray:
    """Anchor centres for one feature map, in input-image pixels.

    Cached because this is pure geometry: it depends only on the input size
    and stride, never on the frame.
    """
    fh, fw = h // stride, w // stride
    ys, xs = np.mgrid[:fh, :fw]
    centres = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
    centres = centres.reshape(-1, 2)
    if anchors > 1:
        centres = np.repeat(centres, anchors, axis=0)
    return centres


def _distance_to_box(centres: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(l, t, r, b) distances from a centre -> (x1, y1, x2, y2)."""
    x1 = centres[:, 0] - dist[:, 0]
    y1 = centres[:, 1] - dist[:, 1]
    x2 = centres[:, 0] + dist[:, 2]
    y2 = centres[:, 1] + dist[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_kps(centres: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """10 alternating (dx, dy) offsets -> (N, 5, 2) absolute points."""
    pts = []
    for i in range(0, dist.shape[1], 2):
        pts.append(centres[:, 0] + dist[:, i])
        pts.append(centres[:, 1] + dist[:, i + 1])
    return np.stack(pts, axis=-1).reshape(-1, 5, 2)


def _prep(canvas: np.ndarray, norm: Normalization) -> np.ndarray:
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    if norm is Normalization.CENTRED_128:
        return (blob - 127.5) / 128.0
    if norm is Normalization.ARCFACE:
        return (blob - 127.5) / 127.5
    return blob / 255.0


def detect(frame: np.ndarray, threshold: float = 0.5, size: int = 640,
           model: str = "scrfd_2.5g") -> list[Face]:
    """Detect faces with an anchor-based InsightFace detector."""
    spec = get_model(model)

    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:rh, :rw] = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)

    sess = sessions.get(spec)
    outs = sessions.run(spec, {sess.get_inputs()[0].name: _prep(canvas, spec.normalization)})
    if len(outs) < 9:
        # Some exports omit the landmark heads; without 5 points we cannot
        # align a face, so this detector is unusable rather than degraded.
        return []

    inv = 1.0 / scale
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    kpss: list[np.ndarray] = []

    for i, stride in enumerate(STRIDES):
        sc = outs[i].reshape(-1)
        bb = outs[i + 3].reshape(-1, 4) * stride
        kp = outs[i + 6].reshape(-1, 10) * stride

        centres = _anchor_centres(size, size, stride, ANCHORS_PER_CELL)
        n = min(len(sc), len(centres))
        if n == 0:
            continue
        sc, bb, kp, centres = sc[:n], bb[:n], kp[:n], centres[:n]

        keep = np.where(sc > threshold)[0]
        if keep.size == 0:
            continue

        b = _distance_to_box(centres[keep], bb[keep]) * inv
        k = _distance_to_kps(centres[keep], kp[keep]) * inv
        boxes.extend(b)
        kpss.extend(k)
        scores.extend(sc[keep].tolist())

    if not boxes:
        return []

    rects = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
             for b in boxes]
    idx = cv2.dnn.NMSBoxes(rects, scores, threshold, 0.4)
    if len(idx) == 0:
        return []

    return [Face(box=np.asarray(boxes[i], np.float32),
                 kps=np.asarray(kpss[i], np.float32),
                 score=float(scores[i]))
            for i in np.array(idx).ravel()]
