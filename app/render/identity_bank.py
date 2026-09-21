"""Pose-aware source identity: one person, many views, one vector per frame.

The existing fusion collapses every source photo into a single 512-d vector
and deliberately down-weights profiles (``w_pose = exp(-|yaw|/45)``). That is
right for building one all-purpose vector and wrong for a target that turns:
a 50-degree source photo is the BEST evidence for a 50-degree target frame,
not the worst. Averaging it away means the hardest frames are rendered from
the least relevant evidence.

This keeps every accepted reference with its measured pose and quality, then
composes a per-frame vector from the references nearest the target's pose.

What it is not:
  * not multi-person -- every reference is verified against the identity
    consensus and outliers are dropped
  * not expression transfer -- only the identity VECTOR is pose-selected. The
    target keeps its own mouth, blink, gaze and timing, because none of that
    comes from the source vector.

AlphaFace takes a single (1,512) ``source`` input, verified from the ONNX
graph, so richer conditioning is not available without retraining. The whole
opportunity is therefore producing the best possible 512-d vector per frame,
which is what this does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.render import alignment, detection, recognition
from app.render.types import Face, RenderError, SourceIdentity

# Blend width in degrees. References within roughly this distance of the
# target pose contribute; beyond it their weight decays away. Wide enough
# that the active set changes gradually as the head turns, which is what
# keeps the identity from flickering.
POSE_SIGMA = 18.0

# A reference must agree with the identity consensus by at least this much.
# Two different people score ~0.02, the same person across poses stays well
# above 0.3, so this rejects intruders without rejecting genuine profiles.
CONSENSUS_FLOOR = 0.28

# Below this the temporal smoother stops chasing; prevents per-frame jitter
# in the fused vector from becoming visible identity flicker.
VECTOR_EMA = 0.75


@dataclass
class Reference:
    """One accepted view of the source person."""

    name: str
    vec: np.ndarray            # raw ArcFace embedding (512,)
    yaw: float
    roll: float
    quality: float
    size: float
    blur: float
    score: float
    consensus: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "yaw": round(self.yaw, 2),
                "roll": round(self.roll, 2), "quality": round(self.quality, 4),
                "size": round(self.size, 1), "blur": round(self.blur, 1),
                "score": round(self.score, 3),
                "consensus": round(self.consensus, 4)}


def _quality(frame: np.ndarray, face: Face) -> dict[str, float]:
    q = recognition.assess_source_face(frame, face)
    # pose_3d, not the 2D proxy. The proxy reads a true 30-degree turn as
    # 17.9 and 45 as 31.7 -- roughly 40% low -- and cannot see pitch at all.
    # Indexing a pose bank with it collapses genuinely different views into
    # the same bin, which is how a pose-aware method measures as a no-op.
    yaw, _pitch, roll = alignment.pose_3d(face.kps, frame.shape)
    return {"size": float(q["size"]), "blur": float(q["blur"]),
            "score": float(q["detector_score"]), "yaw": float(yaw),
            "roll": float(roll)}


def _base_quality(size: float, blur: float, score: float) -> float:
    """Quality WITHOUT a pose term.

    Pose is not a defect here. It is the axis references are indexed on, so
    penalising it would re-create the problem this module exists to solve.
    """
    w_size = min(1.0, size / 160.0)
    w_conf = float(np.clip(score, 0.0, 1.0))
    w_blur = min(1.0, blur / 120.0)
    return float(max(1e-3, w_size * w_conf * w_blur))


class IdentityBank:
    """References indexed by pose, fused per target frame."""

    def __init__(self, refs: list[Reference], fallback: np.ndarray):
        self.refs = refs
        # Used when the bank is empty or a frame has no usable pose.
        self.fallback = fallback.reshape(-1).astype(np.float32)
        self._ema: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, images: list[tuple[str, np.ndarray, Face]],
              model: str = "arcface_w600k_r50") -> "IdentityBank":
        if not images:
            raise RenderError("no usable source faces")

        raw: list[Reference] = []
        for name, frame, face in images:
            try:
                vec = recognition.embed(frame, face.kps, model)
            except RenderError:
                continue
            q = _quality(frame, face)
            raw.append(Reference(
                name=name, vec=np.asarray(vec, np.float32).reshape(-1),
                yaw=q["yaw"], roll=q["roll"],
                quality=_base_quality(q["size"], q["blur"], q["score"]),
                size=q["size"], blur=q["blur"], score=q["score"]))

        if not raw:
            raise RenderError("no usable source faces after quality filtering")

        # Consensus: one identity only. Compare each reference to the
        # quality-weighted mean and drop anything that disagrees sharply --
        # that is a different person, not a different angle.
        stack = np.stack([r.vec for r in raw])
        w = np.array([r.quality for r in raw], np.float32)[:, None]
        mean = (stack * w).sum(0) / max(float(w.sum()), 1e-6)
        mean /= max(float(np.linalg.norm(mean)), 1e-6)
        kept: list[Reference] = []
        for r in raw:
            v = r.vec / max(float(np.linalg.norm(r.vec)), 1e-6)
            r.consensus = float(np.dot(v, mean))
            if r.consensus >= CONSENSUS_FLOOR:
                kept.append(r)
        if not kept:
            kept = raw            # never end up with nothing to swap from

        fb = np.stack([r.vec for r in kept])
        fw = np.array([r.quality for r in kept], np.float32)[:, None]
        fallback = (fb * fw).sum(0) / max(float(fw.sum()), 1e-6)
        return cls(kept, fallback)

    # ---------------------------------------------------------------- query
    def weights_for(self, yaw: float) -> np.ndarray:
        """Contribution of each reference at a given target yaw.

        A smooth Gaussian in pose distance times reference quality. Smooth
        so that a turning head slides between references instead of snapping,
        which is what would show up as identity flicker.
        """
        d = np.array([abs(r.yaw - yaw) for r in self.refs], np.float32)
        pose_w = np.exp(-(d ** 2) / (2.0 * POSE_SIGMA ** 2))
        qual = np.array([r.quality for r in self.refs], np.float32)
        w = pose_w * qual
        s = float(w.sum())
        if s < 1e-8:                      # nothing near this pose
            w = qual.copy()
            s = float(w.sum()) or 1.0
        return w / s

    def vector_for(self, yaw: Optional[float], smooth: bool = True) -> np.ndarray:
        """The identity vector to condition on for a frame at this pose."""
        if not self.refs or yaw is None:
            vec = self.fallback
        else:
            w = self.weights_for(float(yaw))
            vec = (np.stack([r.vec for r in self.refs]) * w[:, None]).sum(0)

        if smooth:
            if self._ema is None:
                self._ema = vec.astype(np.float32)
            else:
                self._ema = (VECTOR_EMA * self._ema
                             + (1.0 - VECTOR_EMA) * vec).astype(np.float32)
            vec = self._ema
        return vec.astype(np.float32)

    def reset_temporal(self) -> None:
        self._ema = None

    # ---------------------------------------------------------------- report
    def coverage(self) -> dict[str, Any]:
        bins = {"left_60+": (-200, -52), "left_45": (-52, -37),
                "left_30": (-37, -22), "left_15": (-22, -8),
                "frontal": (-8, 8), "right_15": (8, 22),
                "right_30": (22, 37), "right_45": (37, 52),
                "right_60+": (52, 200)}
        out: dict[str, int] = {}
        for label, (lo, hi) in bins.items():
            out[label] = sum(1 for r in self.refs if lo <= r.yaw < hi)
        return out

    def summary(self) -> dict[str, Any]:
        cov = self.coverage()
        return {"references": len(self.refs),
                "yaw_range": [round(min((r.yaw for r in self.refs), default=0.0), 1),
                              round(max((r.yaw for r in self.refs), default=0.0), 1)],
                "coverage": cov,
                "covered_bins": sum(1 for v in cov.values() if v),
                "refs": [r.as_dict() for r in self.refs]}

    def to_identity(self) -> SourceIdentity:
        """A plain SourceIdentity, for code paths that want one vector."""
        v = self.fallback / max(float(np.linalg.norm(self.fallback)), 1e-6)
        return SourceIdentity(embedding=v.reshape(1, -1),
                              per_image=[r.as_dict() for r in self.refs],
                              n_images=len(self.refs))
