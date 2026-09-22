"""Pose-conditioned source identity: one 512-d vector, varied per frame.

AlphaFace takes one 512-d ``source`` vector per inference. That is a
constraint on each call, not a requirement that every frame of a video use
the SAME vector. This builds a vector per target frame from the source
observations nearest that frame's pose, blended smoothly.

Two rules the construction obeys, both learned from earlier failures:

  A GLOBAL ANCHOR IS ALWAYS PRESENT. Every vector is a blend of a fixed
  identity anchor and pose-relevant evidence, never pose evidence alone.
  Without it the identity drifts as the head turns, which is a worse
  artefact than the pose mismatch it was meant to fix.

  THE VECTOR MOVES SMOOTHLY. Weights are a continuous Gaussian in pose
  distance and the result is passed through an EMA, so consecutive frames
  cannot receive unrelated identities. A hard switch between reference
  groups would show up as identity flicker, which is disqualifying however
  good the per-frame numbers look.

Expression is never transferred. Only the identity VECTOR is pose-selected;
the target keeps its own mouth, gaze, blink and timing, because none of
those come from the source vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from app.render.types import RenderError

# Width of the pose window, in degrees. Wide enough that the active set of
# references changes gradually as the head turns.
POSE_SIGMA = 20.0

# How much of the final vector is the fixed anchor. Pose evidence adjusts
# the identity; it does not replace it.
ANCHOR_SHARE = 0.55

# Temporal smoothing on the emitted vector.
VECTOR_EMA = 0.80


@dataclass
class Observation:
    """One source view: an embedding with the pose it was seen at."""

    name: str
    embedding: np.ndarray
    yaw: float
    quality: float


class PoseConditionedIdentity:
    """Emits a per-frame 512-d identity vector conditioned on target yaw."""

    def __init__(self, observations: list[Observation],
                 anchor_share: float = ANCHOR_SHARE,
                 sigma: float = POSE_SIGMA, ema: float = VECTOR_EMA):
        if not observations:
            raise RenderError("no source observations")
        self.obs = observations
        self.anchor_share = float(np.clip(anchor_share, 0.0, 1.0))
        self.sigma = sigma
        self.ema = ema
        self._state: Optional[np.ndarray] = None

        vecs = np.stack([o.embedding for o in observations])
        w = np.array([o.quality for o in observations], np.float32)
        a = (vecs * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
        self.anchor = (a / max(float(np.linalg.norm(a)), 1e-6)).astype(np.float32)
        self._yaws = np.array([o.yaw for o in observations], np.float32)
        self._q = w
        self._vecs = vecs

    def reset(self) -> None:
        self._state = None

    def weights_for(self, yaw: float) -> np.ndarray:
        d = np.abs(self._yaws - float(yaw))
        w = np.exp(-(d ** 2) / (2.0 * self.sigma ** 2)) * self._q
        s = float(w.sum())
        if s < 1e-8:
            w = self._q.copy()
            s = float(w.sum()) or 1.0
        return w / s

    def vector_for(self, yaw: Optional[float], smooth: bool = True) -> np.ndarray:
        """The identity vector to condition on for a target frame."""
        if yaw is None:
            v = self.anchor
        else:
            w = self.weights_for(float(yaw))
            pose_v = (self._vecs * w[:, None]).sum(0)
            n = float(np.linalg.norm(pose_v))
            pose_v = pose_v / n if n > 1e-6 else self.anchor
            v = self.anchor_share * self.anchor + (1.0 - self.anchor_share) * pose_v
            n = float(np.linalg.norm(v))
            v = v / n if n > 1e-6 else self.anchor

        if smooth:
            if self._state is None:
                self._state = v.astype(np.float32)
            else:
                self._state = (self.ema * self._state
                               + (1.0 - self.ema) * v).astype(np.float32)
            n = float(np.linalg.norm(self._state))
            v = self._state / n if n > 1e-6 else self.anchor
        return v.astype(np.float32)

    def coverage(self) -> dict[str, Any]:
        bins = {"left_strong": (-90, -28), "left_mild": (-28, -10),
                "frontal": (-10, 10), "right_mild": (10, 28),
                "right_strong": (28, 90)}
        out = {k: int(((self._yaws >= lo) & (self._yaws < hi)).sum())
               for k, (lo, hi) in bins.items()}
        return {"bins": out,
                "yaw_span": [float(self._yaws.min()), float(self._yaws.max())],
                "n": len(self.obs)}
