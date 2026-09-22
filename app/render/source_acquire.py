"""Pick the best <=5 real observations of one person, from photos + video.

Source video is NOT an identity-fusion input. V2.6 measured that: dozens of
video frames poured into one 512-d vector overwhelmed the photo anchors and
made identity worse. This module uses video for something else entirely --
as a way to FIND better reference images -- and then hands at most five of
them to the production fusion, unchanged.

The cap is the whole safety property. A five-minute video and a ten-second
video both yield at most five references, so frame count can never become
voting power.

Selection is an explicit acquisition rule, not a search over subsets. V2.6
showed that scoring arbitrary subsets is unstable: any criterion rewards
agreement with the others, so the most distinctive view -- a genuine profile
-- always looks like the odd one out, and switching subsets moved identity
by 0.063. Here the rule is fixed and stated:

    1. take the single best near-frontal reference as an anchor
    2. fill the remaining slots with the best available observation in each
       pose band that is not yet covered
    3. if pose bands run out, fill with the highest-quality remaining
       references that are not near-duplicates of what is already chosen

That is deterministic given the same inputs, and it cannot drop the anchor
to chase a marginally tighter average.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.render import alignment, detection, recognition
from app.render.source_analysis import PhotoAnalysis, analyse, quality_weight
from app.render.types import RenderError

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "source_acq"
ALGORITHM_VERSION = 1

MAX_REFERENCES = 5

# Coarse scan first; only promising frames get the expensive analysis.
SCAN_FPS = 2.0
MAX_SCAN_FRAMES = 1200

# A candidate must clear these to be considered at all. Deliberately about
# defects, never about pose.
MIN_FACE_PX = 90.0
MIN_DETECTOR_CONF = 0.60
MIN_SHARPNESS = 55.0
MIN_CONSENSUS = 0.30

# Pose bands, in degrees of yaw. Ordered by how much NEW information each
# adds once a frontal anchor exists.
BANDS: list[tuple[str, float, float]] = [
    ("frontal", -10.0, 10.0),
    ("left_mild", -28.0, -10.0),
    ("right_mild", 10.0, 28.0),
    ("left_strong", -80.0, -28.0),
    ("right_strong", 28.0, 80.0),
]
FILL_ORDER = ["frontal", "left_mild", "right_mild", "left_strong", "right_strong"]

# Two references are redundant when close in BOTH pose and appearance.
DUP_POSE_DEG = 10.0
DUP_SIM = 0.90


def band_of(yaw: float) -> Optional[str]:
    for name, lo, hi in BANDS:
        if lo <= yaw < hi:
            return name
    return None


def _acceptable(a: PhotoAnalysis) -> Optional[str]:
    if a.face_px < MIN_FACE_PX:
        return "face too small (%.0fpx)" % a.face_px
    if a.detector_conf < MIN_DETECTOR_CONF:
        return "low detector confidence (%.2f)" % a.detector_conf
    if a.sharpness < MIN_SHARPNESS:
        return "blurred (%.0f)" % a.sharpness
    if not (25.0 <= a.exposure <= 235.0):
        return "exposure %.0f" % a.exposure
    return None


def scan_video(path: Path, anchor_vec: Optional[np.ndarray],
               progress=None) -> tuple[list[PhotoAnalysis], dict[str, int]]:
    """Harvest candidate references from a source video.

    ``anchor_vec=None`` means there is no known identity yet (video-only
    input). Same-person filtering is then skipped on this pass and applied
    afterwards against the video's own consensus.
    """
    import app.render.video as V

    info = V.probe(str(path))
    if info.total_frames <= 0:
        return [], {"sampled": 0}
    step = max(1, int(round(info.fps / max(SCAN_FPS, 0.1))))
    wanted = set(range(0, info.total_frames, step))
    if len(wanted) > MAX_SCAN_FRAMES:
        wanted = set(sorted(wanted)[:MAX_SCAN_FRAMES])

    stats = {"sampled": len(wanted), "no_face": 0, "rejected": 0,
             "wrong_person": 0, "accepted": 0}
    out: list[PhotoAnalysis] = []
    an = (anchor_vec / (float(np.linalg.norm(anchor_vec)) or 1.0)
          if anchor_vec is not None else None)

    # One sequential decode: seeking thousands of times costs far more than
    # reading the file straight through.
    dec = V.decode(str(path), info)
    try:
        for n, fr in enumerate(V.frames(dec, info)):
            if n not in wanted:
                continue
            faces = detection.detect_robust(fr, 0.4)
            if not faces:
                stats["no_face"] += 1
                continue
            f = max(faces, key=lambda x: x.area)
            try:
                a = analyse("%s@%.2fs" % (path.name, n / max(info.fps, 1e-6)),
                            fr, f)
            except RenderError:
                stats["rejected"] += 1
                continue
            if _acceptable(a) is not None:
                stats["rejected"] += 1
                continue
            if an is not None:
                v = a.embedding / (float(np.linalg.norm(a.embedding)) or 1.0)
                a.consensus = float(np.dot(v, an))
                if a.consensus < MIN_CONSENSUS:
                    stats["wrong_person"] += 1
                    continue
            out.append(a)
            if progress and len(out) % 20 == 0:
                progress(len(out), len(wanted))
    finally:
        try:
            dec.kill()
        except OSError:
            pass
    stats["accepted"] = len(out)
    return out, stats


def _redundant(a: PhotoAnalysis, chosen: list[PhotoAnalysis]) -> bool:
    va = a.embedding / (float(np.linalg.norm(a.embedding)) or 1.0)
    for c in chosen:
        if abs(c.yaw - a.yaw) > DUP_POSE_DEG:
            continue
        vc = c.embedding / (float(np.linalg.norm(c.embedding)) or 1.0)
        if float(np.dot(va, vc)) >= DUP_SIM:
            return True
    return False


def select(candidates: list[PhotoAnalysis],
           limit: int = MAX_REFERENCES) -> tuple[list[PhotoAnalysis], dict[str, Any]]:
    """Choose <=limit references: one frontal anchor, then pose coverage.

    Photos and video frames compete purely on measured merit. Nothing is
    preferred for being a photo, and nothing is penalised for coming from
    video -- an excellent 50-degree video frame should beat a redundant
    fifth frontal photo, and a sharp uploaded photo should beat a soft
    video frame at the same angle.
    """
    if not candidates:
        raise RenderError("no usable source references")

    ranked = sorted(candidates, key=lambda a: -quality_weight(a))
    photos = [a for a in ranked if "@" not in a.name]
    chosen: list[PhotoAnalysis] = []
    notes: list[str] = []

    # 1. Anchor -- taken from the UPLOADED PHOTOS whenever any exist.
    #
    # Not because photos are inherently better: measured per-item quality is
    # comparable. It is because a 15-minute video yields ~1100 candidates
    # against 4 photos, so on pure ranking the top of every pose band is a
    # video frame by sheer count. That is the V2.6 failure -- quantity
    # becoming voting power -- returning in a new place. The first
    # measurement of this module picked 0 photos out of 5 available and
    # chose a 128px/181-sharpness anchor over a 209px/302 photo, purely
    # because it sat nearer 0 degrees yaw.
    #
    # The user's uploads are deliberate evidence. Video is opportunistic
    # evidence. When both exist, the deliberate one anchors.
    pool = photos if photos else ranked
    frontals = [a for a in pool if band_of(a.yaw) == "frontal"]
    if frontals:
        chosen.append(frontals[0])
        notes.append("anchor: %s (yaw %.1f)" % (frontals[0].name, frontals[0].yaw))
    else:
        chosen.append(pool[0])
        notes.append("no frontal available; anchor is best overall: %s (yaw %.1f)"
                     % (pool[0].name, pool[0].yaw))

    # 2. Pose coverage: best remaining observation per uncovered band.
    covered = {band_of(c.yaw) for c in chosen}
    for band in FILL_ORDER:
        if len(chosen) >= limit:
            break
        if band in covered:
            continue
        # Photos first within each band, for the same reason as the anchor:
        # a video frame should win a band because it is the only evidence
        # there, or because it is clearly better -- not because there were
        # a thousand of them.
        for src in ((photos, "photo"), (ranked, "video")):
            cands = [a for a in src[0]
                     if a not in chosen and band_of(a.yaw) == band
                     and not _redundant(a, chosen)]
            if cands:
                chosen.append(cands[0])
                covered.add(band)
                notes.append("%s: %s (yaw %.1f)" % (band, cands[0].name,
                                                    cands[0].yaw))
                break

    # 3. Remaining slots: highest quality that adds something new.
    for a in ranked:
        if len(chosen) >= limit:
            break
        if a in chosen or _redundant(a, chosen):
            continue
        chosen.append(a)
        notes.append("fill: %s (yaw %.1f)" % (a.name, a.yaw))

    report = {
        "n_candidates": len(candidates),
        "n_selected": len(chosen),
        "selection": notes,
        "bands_covered": sorted({band_of(c.yaw) for c in chosen if band_of(c.yaw)}),
        "from_photo": sum(1 for c in chosen if "@" not in c.name),
        "from_video": sum(1 for c in chosen if "@" in c.name),
        "selected": [c.as_dict() for c in chosen],
    }
    return chosen, report


# ------------------------------------------------------------------ caching
def _hash(p: Path, sample: int = 1 << 20) -> str:
    h = hashlib.sha256()
    st = p.stat()
    h.update(("%s:%d" % (p.name, st.st_size)).encode())
    with open(p, "rb") as f:
        h.update(f.read(sample))
        if st.st_size > sample * 2:
            f.seek(-sample, 2)
            h.update(f.read(sample))
    return h.hexdigest()


def fingerprint(photos: list[Path], video: Optional[Path]) -> str:
    h = hashlib.sha256()
    h.update(("v%d|" % ALGORITHM_VERSION).encode())
    for p in sorted(photos, key=lambda x: x.name):
        h.update(_hash(p).encode())
    h.update(b"|video|")
    if video is not None:
        h.update(_hash(video).encode())
    h.update(b"|arcface_w600k_r50|yoloface_8n|pose_3d")
    return h.hexdigest()[:32]


def acquire(photo_paths: list[str], video_path: Optional[str] = None,
            use_cache: bool = True, progress=None
            ) -> tuple[list[PhotoAnalysis], dict[str, Any]]:
    """Analyse photos + optional video, return the chosen <=5 references."""
    photos = [Path(p) for p in photo_paths if Path(p).is_file()]
    video = Path(video_path) if video_path and Path(video_path).is_file() else None
    if not photos and video is None:
        raise RenderError("no source photos or video supplied")

    key = fingerprint(photos, video)
    cache_path = CACHE_DIR / ("%s.json" % key)

    cand: list[PhotoAnalysis] = []
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        faces = detection.detect_robust(img, 0.3)
        if not faces:
            continue
        f = max(faces, key=lambda x: x.area)
        try:
            cand.append(analyse(p.name, img, f))
        except RenderError:
            continue

    vstats: dict[str, int] = {}
    if video is not None:
        # The anchor for same-person checks comes from the photos when they
        # exist. Video-only falls back to the video's own consensus, which is
        # weaker but is all there is.
        if cand:
            w = np.array([quality_weight(a) for a in cand], np.float32)[:, None]
            anchor = (np.stack([a.embedding for a in cand]) * w).sum(0) / max(float(w.sum()), 1e-6)
        else:
            anchor = None

        if anchor is None:
            # Video-only: there is no photo consensus to check against yet,
            # so the first pass must accept on quality alone. Passing a
            # dummy anchor rejected every frame, because consensus against
            # an arbitrary vector is meaningless rather than permissive.
            probe, vstats = scan_video(video, None, progress)
            if probe:
                w = np.array([quality_weight(a) for a in probe], np.float32)[:, None]
                anchor = (np.stack([a.embedding for a in probe]) * w).sum(0) / max(float(w.sum()), 1e-6)
                an = anchor / (float(np.linalg.norm(anchor)) or 1.0)
                for a in probe:
                    v = a.embedding / (float(np.linalg.norm(a.embedding)) or 1.0)
                    a.consensus = float(np.dot(v, an))
                vid = [a for a in probe if a.consensus >= MIN_CONSENSUS]
                vstats["wrong_person"] = len(probe) - len(vid)
            else:
                vid = []
        else:
            vid, vstats = scan_video(video, anchor, progress)
        cand += vid

    if not cand:
        raise RenderError("no usable source references found")

    chosen, report = select(cand)
    report["video_stats"] = vstats
    report["n_photos_in"] = len(photos)
    report["fingerprint"] = key
    report["cached"] = False

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            cache_path.write_text(json.dumps(
                {"algorithm_version": ALGORITHM_VERSION, "report": report},
                indent=2, default=str), encoding="utf-8")
        except OSError:
            pass
    return chosen, report


def summary(report: dict[str, Any]) -> str:
    """The short line the UI shows. No jargon, no controls."""
    bits = []
    if report.get("n_photos_in"):
        bits.append("%d photo%s analysed"
                    % (report["n_photos_in"],
                       "" if report["n_photos_in"] == 1 else "s"))
    if report.get("from_video"):
        bits.append("%d video reference%s selected"
                    % (report["from_video"],
                       "" if report["from_video"] == 1 else "s"))
    cov = [b.replace("_", " ") for b in report.get("bands_covered", [])]
    return "%s | coverage: %s" % (", ".join(bits) or "no references",
                                  ", ".join(cov) or "none")
