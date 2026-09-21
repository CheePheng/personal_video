"""Source identity ingestion: photos + optional source video, one person.

The source person may be represented by 1-5 photos and, optionally, a video
of that same person. This turns all of it into one bank of analysed
references, each carrying its measured 3D pose and a quality score, so a
later stage can decide how to use them.

Deliberately separate from the renderer. This module only produces evidence;
whether the render path should condition on pose is a question the benchmark
answers, not this file.

Three rules the implementation follows:

  ONE PERSON. Every candidate is checked against the identity consensus of
  the photos. A video of two people contributes only the one that matches.

  QUALITY OVER QUANTITY. A five-minute video holds thousands of near-identical
  frontal frames. Left unchecked they would outvote a single excellent profile
  photo purely by count, so candidates are clustered by pose AND appearance and
  only representatives survive.

  POSE IS NOT A DEFECT. The existing photo fusion multiplies weight by
  exp(-|yaw|/45), which is correct when building one all-purpose vector and
  wrong here: a 50-degree observation is the most valuable evidence that
  exists for a 50-degree target. Quality here excludes any pose penalty.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from app.render import alignment, detection, recognition
from app.render.types import Face, RenderError

# Bumped whenever selection, scoring or pose changes, so a cached bank built
# by older logic is rebuilt instead of silently reused.
ALGORITHM_VERSION = 1

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "source_banks"

# Candidate sampling. A source video is scanned coarsely first; expensive
# analysis only runs on frames that already have a usable face.
SCAN_FPS = 3.0                 # candidate frames per second of source video
MAX_CANDIDATES = 900           # hard ceiling on frames pulled from one video

# Acceptance thresholds. Deliberately permissive on pose, strict on defects.
MIN_FACE_PX = 80.0             # below this there is little identity to read
MIN_DETECTOR_CONF = 0.55
MIN_SHARPNESS = 40.0           # Laplacian variance on the face crop
MIN_CONSENSUS = 0.28           # same-person floor; two strangers score ~0.02
EXPOSURE_LO, EXPOSURE_HI = 25.0, 235.0

# Deduplication. Two references are redundant when they sit in the same pose
# neighbourhood AND look alike; both conditions must hold.
DEDUP_POSE_DEG = 7.0
DEDUP_SIMILARITY = 0.92
TARGET_REFERENCES = 24         # representatives kept after clustering


@dataclass
class SourceRef:
    """One analysed observation of the source person."""

    origin: str                # file name
    kind: str                  # "photo" | "video"
    timestamp: Optional[float] # seconds into the source video, else None
    embedding: np.ndarray      # raw ArcFace (512,)
    yaw: float
    pitch: float
    roll: float
    face_px: float
    sharpness: float
    detector_conf: float
    exposure: float
    quality: float
    consensus: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin, "kind": self.kind,
            "timestamp": None if self.timestamp is None else round(self.timestamp, 3),
            "yaw": round(self.yaw, 2), "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "face_px": round(self.face_px, 1),
            "sharpness": round(self.sharpness, 1),
            "detector_conf": round(self.detector_conf, 3),
            "exposure": round(self.exposure, 1),
            "quality": round(self.quality, 4),
            "consensus": round(self.consensus, 4),
        }


def _face_stats(frame: np.ndarray, face: Face) -> dict[str, float]:
    x1, y1, x2, y2 = [int(v) for v in face.box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    if crop.size == 0:
        return {"sharpness": 0.0, "exposure": 0.0}
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {"sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
            "exposure": float(grey.mean())}


def _quality(face_px: float, sharpness: float, conf: float,
             exposure: float) -> float:
    """Usefulness of an observation, with NO pose term.

    Pose is the axis references are indexed on. Penalising it here would
    re-create the very bias this module exists to remove.
    """
    w_size = min(1.0, face_px / 160.0)
    w_sharp = min(1.0, sharpness / 120.0)
    w_conf = float(np.clip(conf, 0.0, 1.0))
    # Exposure: full credit in the healthy band, tapering at both ends.
    mid = 130.0
    w_exp = float(np.clip(1.0 - abs(exposure - mid) / 110.0, 0.05, 1.0))
    return float(max(1e-3, w_size * w_sharp * w_conf * w_exp))


def analyse_image(path: Path, kind: str = "photo",
                  timestamp: Optional[float] = None,
                  frame: Optional[np.ndarray] = None,
                  anchor: Optional[np.ndarray] = None) -> Optional[SourceRef]:
    """Analyse one image (or decoded frame) into a SourceRef, or reject it."""
    img = frame if frame is not None else cv2.imread(str(path))
    if img is None or img.shape[0] < 32:
        return None
    faces = detection.detect_robust(img, 0.3)
    if not faces:
        return None

    # With an anchor, pick the face that IS the source person, not the
    # biggest one -- a group photo must not contribute a bystander.
    best: Optional[tuple[float, Face, np.ndarray]] = None
    for f in faces:
        try:
            vec = recognition.embed(img, f.kps)
        except RenderError:
            continue
        key = (float(np.dot(vec.ravel() / (np.linalg.norm(vec) or 1.0),
                            anchor.ravel() / (np.linalg.norm(anchor) or 1.0)))
               if anchor is not None else float(f.area))
        if best is None or key > best[0]:
            best = (key, f, vec)
    if best is None:
        return None
    _, face, vec = best

    st = _face_stats(img, face)
    yaw, pitch, roll = alignment.pose_3d(face.kps, img.shape)
    return SourceRef(
        origin=path.name, kind=kind, timestamp=timestamp,
        embedding=np.asarray(vec, np.float32).reshape(-1),
        yaw=float(yaw), pitch=float(pitch), roll=float(roll),
        face_px=float(face.size), sharpness=st["sharpness"],
        detector_conf=float(face.score), exposure=st["exposure"],
        quality=_quality(face.size, st["sharpness"], face.score, st["exposure"]))


def _acceptable(r: SourceRef) -> tuple[bool, str]:
    if r.face_px < MIN_FACE_PX:
        return False, "face too small (%.0fpx)" % r.face_px
    if r.detector_conf < MIN_DETECTOR_CONF:
        return False, "low detector confidence (%.2f)" % r.detector_conf
    if r.sharpness < MIN_SHARPNESS:
        return False, "blurred (%.0f)" % r.sharpness
    if not (EXPOSURE_LO <= r.exposure <= EXPOSURE_HI):
        return False, "exposure %.0f" % r.exposure
    return True, ""


def scan_video(path: Path, anchor: np.ndarray,
               scan_fps: float = SCAN_FPS,
               progress=None) -> tuple[list[SourceRef], dict[str, int]]:
    """Pull candidate references from a source video."""
    import app.render.video as V

    info = V.probe(str(path))
    if info.total_frames <= 0:
        return [], {"decoded": 0}
    step = max(1, int(round(info.fps / max(scan_fps, 0.1))))
    idx = list(range(0, info.total_frames, step))[:MAX_CANDIDATES]

    stats = {"sampled": len(idx), "no_face": 0, "rejected": 0, "outlier": 0,
             "accepted": 0}
    out: list[SourceRef] = []
    # Decode in one pass rather than seeking per frame: seeking thousands of
    # times on a long video costs far more than reading it straight through.
    want = set(idx)
    dec = V.decode(str(path), info)
    try:
        for n, fr in enumerate(V.frames(dec, info)):
            if n not in want:
                continue
            r = analyse_image(path, "video", n / max(info.fps, 1e-6), fr, anchor)
            if r is None:
                stats["no_face"] += 1
                continue
            ok, _why = _acceptable(r)
            if not ok:
                stats["rejected"] += 1
                continue
            out.append(r)
            if progress and len(out) % 25 == 0:
                progress(len(out), len(idx))
    finally:
        try:
            dec.kill()
        except OSError:
            pass
    stats["accepted"] = len(out)
    return out, stats


def verify_identity(refs: list[SourceRef], anchor: np.ndarray
                    ) -> tuple[list[SourceRef], int]:
    """Drop references that are not the same person as the anchor."""
    a = anchor.ravel() / (float(np.linalg.norm(anchor)) or 1.0)
    kept, dropped = [], 0
    for r in refs:
        v = r.embedding / (float(np.linalg.norm(r.embedding)) or 1.0)
        r.consensus = float(np.dot(v, a))
        if r.consensus >= MIN_CONSENSUS:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def deduplicate(refs: list[SourceRef],
                target: int = TARGET_REFERENCES) -> list[SourceRef]:
    """Keep representative observations, not the most numerous ones.

    Two references are redundant only when they are BOTH close in pose and
    close in appearance. Requiring both means a genuinely different view is
    never discarded just because an earlier frame happened to score higher.
    """
    ordered = sorted(refs, key=lambda r: -r.quality)
    kept: list[SourceRef] = []
    for r in ordered:
        v = r.embedding / (float(np.linalg.norm(r.embedding)) or 1.0)
        redundant = False
        for k in kept:
            if abs(k.yaw - r.yaw) > DEDUP_POSE_DEG:
                continue
            kv = k.embedding / (float(np.linalg.norm(k.embedding)) or 1.0)
            if float(np.dot(v, kv)) >= DEDUP_SIMILARITY:
                redundant = True
                break
        if not redundant:
            kept.append(r)
        if len(kept) >= target:
            break
    return kept


POSE_BINS: list[tuple[str, float, float]] = [
    ("left_60+", -200.0, -52.0), ("left_45", -52.0, -37.0),
    ("left_30", -37.0, -22.0), ("left_15", -22.0, -8.0),
    ("frontal", -8.0, 8.0), ("right_15", 8.0, 22.0),
    ("right_30", 22.0, 37.0), ("right_45", 37.0, 52.0),
    ("right_60+", 52.0, 200.0),
]


def coverage(refs: list[SourceRef]) -> dict[str, Any]:
    yaw_bins = {name: 0 for name, _, _ in POSE_BINS}
    for r in refs:
        for name, lo, hi in POSE_BINS:
            if lo <= r.yaw < hi:
                yaw_bins[name] += 1
                break
    up = sum(1 for r in refs if r.pitch > 12.0)
    down = sum(1 for r in refs if r.pitch < -12.0)
    return {"yaw": yaw_bins, "looking_up": up, "looking_down": down,
            "bins_covered": sum(1 for v in yaw_bins.values() if v),
            "yaw_span": [round(min((r.yaw for r in refs), default=0.0), 1),
                         round(max((r.yaw for r in refs), default=0.0), 1)]}


# ------------------------------------------------------------------ caching
def _hash_file(p: Path, sample: int = 1 << 20) -> str:
    h = hashlib.sha256()
    st = p.stat()
    h.update(f"{p.name}:{st.st_size}".encode())
    with open(p, "rb") as f:
        h.update(f.read(sample))
        if st.st_size > sample * 2:
            f.seek(-sample, 2)
            h.update(f.read(sample))
    return h.hexdigest()


def fingerprint(photos: list[Path], video: Optional[Path]) -> str:
    h = hashlib.sha256()
    h.update(f"v{ALGORITHM_VERSION}|".encode())
    for p in sorted(photos, key=lambda x: x.name):
        h.update(_hash_file(p).encode())
    h.update(b"|video|")
    if video is not None:
        h.update(_hash_file(video).encode())
    # Model identities: a different recogniser or pose model invalidates the
    # embeddings and the poses stored alongside them.
    h.update(b"|arcface_w600k_r50|yoloface_8n|pose_3d_v1")
    return h.hexdigest()[:32]


def build(photos: list[Path], video: Optional[Path] = None,
          use_cache: bool = True, progress=None) -> dict[str, Any]:
    """Analyse photos (+ optional video) into one cached source bank."""
    photos = [Path(p) for p in photos if Path(p).is_file()]
    if not photos:
        raise RenderError("at least one source photo is required")
    video = Path(video) if video and Path(video).is_file() else None

    key = fingerprint(photos, video)
    cache_path = CACHE_DIR / f"{key}.json"
    if use_cache and cache_path.is_file():
        try:
            rec = json.loads(cache_path.read_text(encoding="utf-8"))
            if rec.get("algorithm_version") == ALGORITHM_VERSION:
                rec["cached"] = True
                return rec
        except (OSError, ValueError):
            pass

    t0 = time.time()
    # Photos first: they establish who the person is, and the anchor that
    # every video frame is checked against.
    photo_refs: list[SourceRef] = []
    for p in photos:
        r = analyse_image(p, "photo")
        if r is not None:
            photo_refs.append(r)
    if not photo_refs:
        raise RenderError("no usable face found in the source photos")

    pw = np.array([r.quality for r in photo_refs], np.float32)[:, None]
    anchor = (np.stack([r.embedding for r in photo_refs]) * pw).sum(0) / max(float(pw.sum()), 1e-6)

    photo_refs, photo_dropped = verify_identity(photo_refs, anchor)

    vid_refs: list[SourceRef] = []
    vstats: dict[str, int] = {}
    if video is not None:
        vid_refs, vstats = scan_video(video, anchor, progress=progress)
        vid_refs, vid_dropped = verify_identity(vid_refs, anchor)
        vstats["outlier"] = vid_dropped
        before = len(vid_refs)
        vid_refs = deduplicate(vid_refs)
        vstats["after_dedup"] = len(vid_refs)
        vstats["deduped_away"] = before - len(vid_refs)

    allrefs = photo_refs + vid_refs
    rec = {
        "algorithm_version": ALGORITHM_VERSION,
        "fingerprint": key,
        "photos": [p.name for p in photos],
        "video": video.name if video else None,
        "n_photos": len(photo_refs),
        "n_video_refs": len(vid_refs),
        "photo_outliers_dropped": photo_dropped,
        "video_stats": vstats,
        "coverage": coverage(allrefs),
        "refs": [r.as_dict() for r in allrefs],
        "embeddings": [r.embedding.tolist() for r in allrefs],
        "build_s": round(time.time() - t0, 2),
        "cached": False,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rec), encoding="utf-8")
    return rec


def refs_from_record(rec: dict[str, Any]) -> list[SourceRef]:
    out = []
    for d, e in zip(rec["refs"], rec["embeddings"]):
        out.append(SourceRef(
            origin=d["origin"], kind=d["kind"], timestamp=d.get("timestamp"),
            embedding=np.asarray(e, np.float32),
            yaw=d["yaw"], pitch=d["pitch"], roll=d["roll"],
            face_px=d["face_px"], sharpness=d["sharpness"],
            detector_conf=d["detector_conf"], exposure=d["exposure"],
            quality=d["quality"], consensus=d.get("consensus", 0.0)))
    return out


def summary_line(rec: dict[str, Any]) -> str:
    """The short, non-technical summary the UI shows."""
    cov = rec["coverage"]
    have = [n.replace("_", " ") for n, c in cov["yaw"].items() if c]
    bits = ["%d photos analyzed" % rec["n_photos"]]
    if rec.get("n_video_refs"):
        bits.append("%d video references selected" % rec["n_video_refs"])
    return "%s | coverage: %s" % (", ".join(bits), ", ".join(have) or "none")
