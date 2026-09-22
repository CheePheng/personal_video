"""Understand each source photo before using it.

The previous fusion knew only that it had several photos. It scored each on
size, sharpness, detector confidence and frontality, then averaged. That has
three specific faults, and this module exists to fix them:

  POSE WAS TREATED AS A DEFECT. The old weight multiplied by
  ``exp(-|yaw|/45)``, so a razor-sharp 45-degree photo was given 0.37x the
  influence of a soft frontal one purely for being turned. Pose and quality
  are different axes: a sharp profile is excellent evidence about the side of
  a face, and a blurred frontal is nearly useless evidence about anything.

  OCCLUSION WAS INVISIBLE. A hand across one eye, a peace sign over a cheek
  or sunglasses changed nothing in the score, so an obstructed photo pulled
  the identity with full force.

  DUPLICATES VOTED REPEATEDLY. Five near-identical selfies counted as five
  independent observations rather than one, drowning out a single genuinely
  different view.

What this module does NOT claim: AlphaFace consumes one 512-d vector
(verified from the ONNX graph: inputs are ``target`` [b,3,256,256] and
``source`` [b,512]). There is no per-region spatial conditioning available
without retraining. Region visibility is therefore used to decide HOW MUCH a
photo contributes, not to assemble a face from parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from app.render import alignment, recognition
from app.render.types import Face, RenderError

# Facial regions, as fractions of the aligned 256x256 crop. Coarse on
# purpose: these gate how much a photo is trusted, they are not a parse map.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "forehead":    (0.22, 0.05, 0.78, 0.28),
    "left_eye":    (0.14, 0.28, 0.45, 0.46),
    "right_eye":   (0.55, 0.28, 0.86, 0.46),
    "nose":        (0.38, 0.40, 0.62, 0.62),
    "left_cheek":  (0.10, 0.46, 0.38, 0.72),
    "right_cheek": (0.62, 0.46, 0.90, 0.72),
    "mouth":       (0.32, 0.64, 0.68, 0.82),
    "jaw":         (0.22, 0.78, 0.78, 0.96),
}

# Regions weighted by how much identity they carry. Eyes and nose dominate
# recognition; the forehead is mostly hair and contributes least.
REGION_IMPORTANCE = {
    "left_eye": 0.18, "right_eye": 0.18, "nose": 0.18, "mouth": 0.12,
    "left_cheek": 0.11, "right_cheek": 0.11, "jaw": 0.08, "forehead": 0.04,
}


@dataclass
class PhotoAnalysis:
    """Everything measured about one source photo."""

    name: str
    embedding: np.ndarray
    yaw: float
    pitch: float
    roll: float
    face_px: float
    sharpness: float
    detector_conf: float
    exposure: float
    contrast: float
    clipped_high: float
    clipped_low: float
    visibility: dict[str, float] = field(default_factory=dict)
    visible_fraction: float = 1.0
    occluded_regions: list[str] = field(default_factory=list)
    consensus: float = 0.0
    duplicate_of: Optional[str] = None
    dup_penalty: float = 1.0
    weight: float = 0.0
    rejected: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1), "roll": round(self.roll, 1),
            "face_px": round(self.face_px, 1),
            "sharpness": round(self.sharpness, 1),
            "detector_conf": round(self.detector_conf, 3),
            "exposure": round(self.exposure, 1),
            "visible_fraction": round(self.visible_fraction, 3),
            "occluded_regions": self.occluded_regions,
            "consensus": round(self.consensus, 4),
            "duplicate_of": self.duplicate_of,
            "dup_penalty": round(self.dup_penalty, 3),
            "weight": round(self.weight, 4),
            "rejected": self.rejected,
        }


def _region_visibility(crop: np.ndarray) -> dict[str, float]:
    """Per-region confidence that actual skin/feature is visible.

    Occluders that matter in practice -- a hand, hair, sunglasses, deep
    shadow -- share measurable signatures against facial skin: they are
    either far from the crop's dominant skin tone, or they are flat and
    textureless, or they are crushed to black. None of those is a classifier;
    each is a property of the pixels.
    """
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Dominant skin chroma from the central face area, which is the region
    # least likely to be occluded in any usable photo.
    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    cx0, cx1 = int(w * 0.33), int(w * 0.67)
    core = lab[cy0:cy1, cx0:cx1].reshape(-1, 3)
    skin_ab = np.median(core[:, 1:], axis=0)

    out: dict[str, float] = {}
    for name, (rx0, ry0, rx1, ry1) in REGIONS.items():
        x0, y0 = int(rx0 * w), int(ry0 * h)
        x1, y1 = int(rx1 * w), int(ry1 * h)
        reg = lab[y0:y1, x0:x1]
        greg = grey[y0:y1, x0:x1]
        if reg.size == 0:
            out[name] = 0.0
            continue

        # Chroma distance from skin: hands are close, hair/sunglasses are not.
        d_ab = float(np.linalg.norm(
            np.median(reg.reshape(-1, 3)[:, 1:], axis=0) - skin_ab))
        chroma_ok = float(np.clip(1.0 - (d_ab - 6.0) / 22.0, 0.0, 1.0))

        # Texture: skin has fine detail; a flat hand or lens does not.
        tex = float(cv2.Laplacian(greg, cv2.CV_64F).var())
        tex_ok = float(np.clip(tex / 45.0, 0.0, 1.0))

        # Luminance: crushed-black (hair, sunglasses, deep shadow) or blown.
        lum = float(np.median(greg))
        lum_ok = float(np.clip(min(lum - 18.0, 246.0 - lum) / 28.0, 0.0, 1.0))

        # Any one signature failing is enough to distrust the region, so the
        # weakest term dominates rather than being averaged away.
        out[name] = float(min(chroma_ok, max(tex_ok, 0.35), lum_ok))
    return out


def analyse(name: str, frame: np.ndarray, face: Face,
            model: str = "arcface_w600k_r50") -> PhotoAnalysis:
    """Measure one source photo. Pose is recorded, never penalised."""
    q = recognition.assess_source_face(frame, face)
    yaw, pitch, roll = alignment.pose_3d(face.kps, frame.shape)
    crop, _ = alignment.warp(frame, face.kps, "arcface_128", 256)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    vis = _region_visibility(crop)
    # A turned head genuinely hides its far side. That is geometry, not
    # obstruction, so the self-occluded cheek/eye is excluded from the
    # "something is covering this face" verdict.
    hidden_by_pose: set[str] = set()
    if yaw > 22:
        hidden_by_pose |= {"left_cheek", "left_eye"}
    elif yaw < -22:
        hidden_by_pose |= {"right_cheek", "right_eye"}

    scored = {k: v for k, v in vis.items() if k not in hidden_by_pose}
    imp = {k: REGION_IMPORTANCE[k] for k in scored}
    tot = sum(imp.values()) or 1.0
    visible_fraction = float(sum(scored[k] * imp[k] for k in scored) / tot)
    occluded = sorted(k for k, v in scored.items() if v < 0.45)

    return PhotoAnalysis(
        name=name,
        embedding=np.asarray(recognition.embed(frame, face.kps, model),
                             np.float32).reshape(-1),
        yaw=yaw, pitch=pitch, roll=roll,
        face_px=float(q["size"]), sharpness=float(q["blur"]),
        detector_conf=float(q["detector_score"]),
        exposure=float(np.median(grey)), contrast=float(grey.std()),
        clipped_high=float((grey > 250).mean()),
        clipped_low=float((grey < 5).mean()),
        visibility=vis, visible_fraction=visible_fraction,
        occluded_regions=occluded)


def quality_weight(a: PhotoAnalysis) -> float:
    """Usefulness of a photo as identity evidence.

    Includes a mild pose term, which contradicts what this module originally
    argued and is kept because the measurement disagreed with the argument.
    Ablation over 21 contamination cases across 3 identities:

        variant                      mean drift    worst
        production (pose pen only)      0.02034  0.10084
        visibility, NO pose penalty     0.02639  0.07497
        visibility + pose penalty       0.01792  0.06487   <- shipped

    Dropping the pose penalty made the fusion MORE sensitive to a bad photo.
    A frontal reference is not better evidence about the side of a face, but
    it is a more stable anchor, and removing that anchoring cost more than
    the theoretical unfairness to profiles was worth. The penalty is mild
    (0.4 + 0.6*exp(-|yaw|/45)), so a 45-degree photo still carries 0.62x
    rather than being dismissed.

    Visibility is the genuinely new term and the one that pays: it is what
    takes the worst case from 0.10084 to 0.06487.
    """
    w_size = min(1.0, a.face_px / 160.0)
    w_sharp = min(1.0, a.sharpness / 120.0)
    w_conf = float(np.clip(a.detector_conf, 0.0, 1.0))
    w_exp = float(np.clip(1.0 - abs(a.exposure - 128.0) / 118.0, 0.05, 1.0))
    w_clip = float(np.clip(1.0 - (a.clipped_high + a.clipped_low) * 3.0, 0.1, 1.0))
    # Visibility enters with a floor: a partly obstructed photo still carries
    # real identity, so it is down-weighted rather than discarded.
    w_vis = 0.25 + 0.75 * float(np.clip(a.visible_fraction, 0.0, 1.0))
    w_pose = 0.4 + 0.6 * float(np.exp(-abs(a.yaw) / 45.0))
    return float(max(1e-3, w_size * w_sharp * w_conf * w_exp * w_clip
                     * w_vis * w_pose))


def mark_duplicates(items: list[PhotoAnalysis],
                    pose_deg: float = 12.0,
                    sim: float = 0.86) -> None:
    """Suppress redundant evidence so quantity is not voting power.

    Two photos are redundant only when they are BOTH close in pose and close
    in appearance. Requiring both means a genuinely different view is never
    suppressed just because an earlier photo scored higher.
    """
    order = sorted(items, key=lambda x: -quality_weight(x))
    kept: list[PhotoAnalysis] = []
    for a in order:
        va = a.embedding / (float(np.linalg.norm(a.embedding)) or 1.0)
        for k in kept:
            vk = k.embedding / (float(np.linalg.norm(k.embedding)) or 1.0)
            if abs(k.yaw - a.yaw) <= pose_deg and float(np.dot(va, vk)) >= sim:
                a.duplicate_of = k.name
                # Halve rather than drop: a near-duplicate is still a second
                # look at the same view and slightly reduces noise.
                a.dup_penalty = 0.5
                break
        if a.duplicate_of is None:
            kept.append(a)


def consensus_scores(items: list[PhotoAnalysis]) -> None:
    """Agreement with the quality-weighted centre, for outlier rejection."""
    if not items:
        return
    w = np.array([quality_weight(a) for a in items], np.float32)[:, None]
    stack = np.stack([a.embedding for a in items])
    centre = (stack * w).sum(0) / max(float(w.sum()), 1e-6)
    centre /= max(float(np.linalg.norm(centre)), 1e-6)
    for a in items:
        v = a.embedding / (float(np.linalg.norm(a.embedding)) or 1.0)
        a.consensus = float(np.dot(v, centre))
