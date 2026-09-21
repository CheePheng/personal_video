"""Persistent multi-face tracking.

"Largest face each frame" is not tracking: the moment a second person steps
nearer the camera, the swap jumps to them. This module keeps *tracks* -- each
with a running identity, a predicted position and a confidence -- and matches
detections to them with a combined score:

    identity (ArcFace cosine)  -- dominant, because it is the only signal that
                                  actually knows *who* someone is
  + IoU with the predicted box -- continuity through brief identity dips
  + centre motion consistency  -- distinguishes crossing faces
  + scale continuity           -- a face does not triple in size in one frame

Matching is done globally (best pairs first) rather than greedily per track, so
two faces crossing cannot both claim the same detection.

The governing rule: **when confidence drops, swap nobody.** A frame left
untouched is a far smaller error than a frame with the wrong person's face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.render.types import Face

# Below this cosine we do not believe a detection is our person.
IDENTITY_FLOOR = 0.28
# Below this combined score a detection is not matched to a track at all.
MATCH_FLOOR = 0.30
# Frames a track survives unmatched before it is considered gone.
MAX_MISSES = 45


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1e-6))


@dataclass
class Track:
    """One person followed through time."""

    track_id: int
    embedding: np.ndarray                 # running identity (L2-normalised)
    box: np.ndarray
    centre: np.ndarray
    size: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    misses: int = 0
    hits: int = 1
    last_score: float = 1.0
    # Diagnostics the benchmark reads back.
    switches: int = 0
    lost: int = 0

    def predict(self) -> np.ndarray:
        """Where we expect this face next, from constant-velocity motion."""
        return self.centre + self.velocity

    def update(self, face: Face, embedding: Optional[np.ndarray], score: float) -> None:
        new_centre = face.centre
        self.velocity = 0.5 * self.velocity + 0.5 * (new_centre - self.centre)
        self.centre = new_centre
        self.box = face.box
        self.size = face.size
        self.misses = 0
        self.hits += 1
        self.last_score = score
        if embedding is not None:
            # Slow EMA: adapts to lighting/pose drift without letting one bad
            # frame (or a momentary wrong match) rewrite who this track is.
            self.embedding = 0.9 * self.embedding + 0.1 * embedding
            self.embedding /= max(float(np.linalg.norm(self.embedding)), 1e-6)

    def mark_missed(self) -> None:
        self.misses += 1
        # Decay prediction so a long-lost track does not drift across frame.
        self.velocity *= 0.8


class FaceTracker:
    """Assign detections to persistent tracks across a video."""

    def __init__(self, diag: float, identity_floor: float = IDENTITY_FLOOR):
        self.diag = max(diag, 1.0)
        self.identity_floor = identity_floor
        self.tracks: list[Track] = []
        self._next_id = 1
        self.stats = {"switches": 0, "lost": 0, "created": 0, "skipped_low_conf": 0}

    def reset_motion(self) -> None:
        """Called on a scene cut: positions no longer carry over, identity does."""
        for t in self.tracks:
            t.velocity[:] = 0.0
            t.misses = 0

    def _score(self, track: Track, face: Face, emb: Optional[np.ndarray]) -> float:
        identity = 0.0
        if emb is not None:
            identity = float(np.dot(track.embedding.ravel(), emb.ravel()))

        pred = track.predict()
        dist = float(np.linalg.norm(face.centre - pred)) / self.diag
        motion = float(np.exp(-dist * 8.0))

        pbox = np.array([pred[0] - (track.box[2] - track.box[0]) / 2,
                         pred[1] - (track.box[3] - track.box[1]) / 2,
                         pred[0] + (track.box[2] - track.box[0]) / 2,
                         pred[1] + (track.box[3] - track.box[1]) / 2], np.float32)
        overlap = _iou(pbox, face.box)

        ratio = face.size / max(track.size, 1.0)
        scale = float(np.exp(-abs(np.log(max(ratio, 1e-3))) * 1.5))

        # Identity dominates; geometry only disambiguates similar-looking
        # candidates and carries a track through a brief identity dip.
        return 0.60 * identity + 0.18 * overlap + 0.14 * motion + 0.08 * scale

    def update(self, faces: list[Face],
               embeddings: list[Optional[np.ndarray]]) -> dict[int, Face]:
        """Match this frame's detections to tracks. Returns {track_id: Face}."""
        assigned: dict[int, Face] = {}
        if not faces:
            for t in self.tracks:
                t.mark_missed()
            self._retire()
            return assigned

        # Score every (track, detection) pair, then take the best pairs
        # globally so two crossing faces cannot both grab one detection.
        pairs = []
        for ti, t in enumerate(self.tracks):
            for fi, (f, e) in enumerate(zip(faces, embeddings)):
                pairs.append((self._score(t, f, e), ti, fi))
        pairs.sort(reverse=True)

        used_t: set[int] = set()
        used_f: set[int] = set()
        for score, ti, fi in pairs:
            if ti in used_t or fi in used_f or score < MATCH_FLOOR:
                continue
            track, face, emb = self.tracks[ti], faces[fi], embeddings[fi]

            # Even with good geometry, refuse a match whose identity is wrong.
            # This is what stops a face swap jumping to whoever is nearest.
            if emb is not None:
                ident = float(np.dot(track.embedding.ravel(), emb.ravel()))
                if ident < self.identity_floor:
                    self.stats["skipped_low_conf"] += 1
                    continue

            track.update(face, emb, score)
            assigned[track.track_id] = face
            used_t.add(ti)
            used_f.add(fi)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.mark_missed()

        # Unmatched detections become new tracks (a new person entered).
        for fi, (f, e) in enumerate(zip(faces, embeddings)):
            if fi in used_f or e is None:
                continue
            self.tracks.append(Track(track_id=self._next_id, embedding=e.copy(),
                                     box=f.box.copy(), centre=f.centre.copy(),
                                     size=f.size))
            assigned[self._next_id] = f
            self._next_id += 1
            self.stats["created"] += 1

        self._retire()
        return assigned

    def _retire(self) -> None:
        keep = []
        for t in self.tracks:
            if t.misses > MAX_MISSES:
                self.stats["lost"] += 1
            else:
                keep.append(t)
        self.tracks = keep


class TargetLock:
    """Follows ONE chosen person for 'reference' mode.

    Locks onto a target face, then re-acquires it by identity after gaps. It
    deliberately has no fallback to "the biggest face": if the locked identity
    is not confidently present, it returns nothing and the frame passes through
    untouched. That is the difference between a missing swap and a wrong one.
    """

    def __init__(self, diag: float, identity_floor: float = IDENTITY_FLOOR):
        self.tracker = FaceTracker(diag, identity_floor)
        self.identity_floor = identity_floor
        self.locked_id: Optional[int] = None
        self.locked_embedding: Optional[np.ndarray] = None
        self.frames_swapped = 0
        self.frames_skipped = 0
        self.reacquisitions = 0
        self.identity_switches = 0

    def reset_motion(self) -> None:
        self.tracker.reset_motion()

    def predicted_iou(self, face: Face) -> float:
        """Overlap between a detection and where the locked track should be.

        Used by the single-person fast path to decide whether a detection is
        obviously the same person continuing, or something that warrants a
        real identity check.
        """
        trk = next((t for t in self.tracker.tracks
                    if t.track_id == self.locked_id), None)
        if trk is None:
            return 0.0
        pred = trk.predict()
        hw = (trk.box[2] - trk.box[0]) / 2
        hh = (trk.box[3] - trk.box[1]) / 2
        pbox = np.array([pred[0] - hw, pred[1] - hh,
                         pred[0] + hw, pred[1] + hh], np.float32)
        return _iou(pbox, face.box)

    def select(self, faces: list[Face], embeddings: list[Optional[np.ndarray]]
               ) -> Optional[Face]:
        assigned = self.tracker.update(faces, embeddings)

        if self.locked_id is None:
            # First lock: the most prominent confidently-detected face.
            best, best_key = None, -1.0
            for tid, face in assigned.items():
                key = face.area * max(face.score, 0.01)
                if key > best_key:
                    best, best_key = tid, key
            if best is None:
                self.frames_skipped += 1
                return None
            self.locked_id = best
            trk = next(t for t in self.tracker.tracks if t.track_id == best)
            self.locked_embedding = trk.embedding.copy()
            self.frames_swapped += 1
            return assigned[best]

        # Normal case: our track was matched this frame.
        if self.locked_id in assigned:
            trk = next((t for t in self.tracker.tracks if t.track_id == self.locked_id), None)
            if trk is not None and self.locked_embedding is not None:
                # Guard against a track that has quietly drifted onto someone
                # else: compare against the identity we originally locked.
                if float(np.dot(self.locked_embedding.ravel(), trk.embedding.ravel())) < self.identity_floor:
                    self.identity_switches += 1
                    self.locked_id = None
                    self.frames_skipped += 1
                    return None
            self.frames_swapped += 1
            return assigned[self.locked_id]

        # Our track is missing. Try to re-acquire by identity alone.
        if self.locked_embedding is not None:
            best_face, best_sim = None, self.identity_floor
            for tid, face in assigned.items():
                trk = next((t for t in self.tracker.tracks if t.track_id == tid), None)
                if trk is None:
                    continue
                sim = float(np.dot(self.locked_embedding.ravel(), trk.embedding.ravel()))
                if sim > best_sim:
                    best_face, best_sim, best_id = face, sim, tid
            if best_face is not None:
                self.locked_id = best_id
                self.reacquisitions += 1
                self.frames_swapped += 1
                return best_face

        self.frames_skipped += 1
        return None

    def report(self) -> dict[str, int]:
        return {
            "frames_swapped": self.frames_swapped,
            "frames_skipped": self.frames_skipped,
            "reacquisitions": self.reacquisitions,
            "identity_switches": self.identity_switches,
            "tracks_created": self.tracker.stats["created"],
            "tracks_lost": self.tracker.stats["lost"],
            "low_confidence_rejections": self.tracker.stats["skipped_low_conf"],
        }
