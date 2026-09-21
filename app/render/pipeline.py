"""The render pipeline: source identity -> tracked swap -> encoded video.

One frame's journey:

    detect faces
      -> embed each (identity vectors)
      -> tracker decides which face is OUR person  (or: none -- pass through)
      -> align, swap, (optional) pixel boost
      -> build mask: model + parsing + occlusion, temporally smoothed
      -> colour-match to the target's lighting, temporally smoothed
      -> composite
      -> (optional) restore detail
      -> encode

Progress is ``frames_done / total_frames`` -- a real count of work completed.

No content classification happens anywhere in this path. The pipeline fails
only for technical reasons, all surfaced as RenderError with the actual cause.
"""

from __future__ import annotations

import subprocess
import time
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from app.render import (alignment, color, detection, masking, metrics,
                        recognition, restoration, sessions, swapping, video)
from app.render.registry import get_model
from app.render.types import (Face, PipelineConfig, RenderError, SourceIdentity)

ProgressFn = Callable[[str, float, dict], None]

# Stage weights for the overall bar. 'processing' dominates because it is the
# only stage whose cost scales with video length.
# How long the single-person fast path may coast before re-verifying
# identity. Short enough that a wrong lock cannot persist visibly.
EMBED_EVERY = 8

STAGES = [("preparing", 0.02), ("analysing", 0.04), ("benchmarking", 0.10),
          ("processing", 0.76), ("encoding", 0.05), ("restoring audio", 0.03)]


@dataclass
class RenderOptions:
    quality: str = "quality"                  # fast | quality | auto
    swap_all_faces: bool = False
    config: Optional[PipelineConfig] = None   # explicit pipeline (auto fills it)
    detect_threshold: float = 0.5
    detector: str = "yoloface_8n"
    # Benchmarked (data/benchmarks/components/): YOLOFace reached 1.000 recall
    # at 8.2 ms against 0.829 for both SCRFD and RetinaFace on the hard-frame
    # set, so it stays primary. SCRFD is a different architecture family, which
    # is what makes it useful as a second opinion on frames YOLO drops.
    detector_fallback: Optional[str] = "scrfd_2.5g"
    identity_floor: float = 0.28
    use_parsing: bool = True
    use_occlusion: bool = True
    color_match: bool = True
    temporal: bool = True
    max_frames: Optional[int] = None          # benchmarking/tests only
    # Render only part of the target. Seconds from the start of the file;
    # end_time None means "to the end". Used both for a user-chosen range and
    # for the segmented long-video renderer.
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class RenderResult:
    frames: int = 0
    faces_swapped: int = 0
    config: Optional[PipelineConfig] = None
    tracking: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    benchmark: Optional[dict[str, Any]] = None
    mask_sources: dict[str, bool] = field(default_factory=dict)
    encoder: str = ""
    scene_cuts: int = 0

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in
             ("frames", "faces_swapped", "tracking", "video", "gpu",
              "timings", "fallbacks", "mask_sources", "encoder", "scene_cuts")}
        # PipelineConfig uses slots, so no __dict__ -- asdict() is the
        # supported way to serialise a dataclass either way.
        d["config"] = dataclasses.asdict(self.config) if self.config else None
        d["benchmark"] = self.benchmark
        return d


# ---------------------------------------------------------------- source
def load_source(paths: list[str]) -> SourceIdentity:
    """Build a fused identity from 1..5 photos of the same person."""
    if not paths:
        raise RenderError("no source photo supplied")

    collected: list[tuple[str, np.ndarray, Face]] = []
    problems: list[str] = []
    for p in paths[:5]:
        img = cv2.imread(str(p))
        if img is None:
            problems.append(f"{Path(p).name}: unreadable or unsupported image")
            continue
        faces = detection.detect_robust(img, 0.4, "quality")
        if not faces:
            problems.append(f"{Path(p).name}: no face detected")
            continue
        collected.append((Path(p).name, img, max(faces, key=lambda f: f.area)))

    if not collected:
        raise RenderError(
            "no usable face in the source photo(s). " + "; ".join(problems) +
            ". Use a clear, well-lit, front-facing photo.")
    return recognition.build_source_identity(collected)


# ---------------------------------------------------------------- one frame
class FrameRenderer:
    """Stateful per-frame renderer. State exists for temporal consistency."""

    def __init__(self, cfg: PipelineConfig, opts: RenderOptions,
                 identity: SourceIdentity, diag: float):
        self.cfg = cfg
        self.opts = opts
        self.identity = identity
        self.embedding = swapping.prepare_embedding(cfg.swapper, identity.embedding)
        self.mask_smoother = masking.MaskSmoother()
        self.color_smoother = color.ColorSmoother()
        self.transform_smoother = alignment.TransformSmoother()
        self.mask_sources: dict[str, bool] = {}
        self.last_mask_jitter = 0.0

    def reset_temporal(self) -> None:
        """Scene cut: nothing from the previous shot should carry over."""
        self.mask_smoother.reset()
        self.color_smoother.reset()
        self.transform_smoother.reset()

    def render_face(self, frame: np.ndarray, face: Face) -> tuple[np.ndarray, np.ndarray]:
        """Swap one face into ``frame``. Returns (frame, full-frame mask)."""
        patch, model_mask, matrix, size = swapping.swap(
            frame, face.kps, self.embedding, self.cfg.swapper, self.cfg.pixel_boost)

        if self.opts.temporal:
            matrix = self.transform_smoother(matrix, face.size)

        # Mask modes are genuinely different pipelines, not labels:
        #   oval    -- geometric fallback only
        #   model   -- the swapper's own mask (+ feathered box)
        #   parsing -- ... + BiSeNet face parsing (hair/glasses/hat excluded)
        #   full    -- ... + XSeg occlusion (hands and objects in front)
        # Previously "model" and "parsing" both enabled everything, so Auto
        # Max could not tell them apart and the option was decorative.
        mode = self.cfg.mask
        mask, used = masking.build(
            frame, face.kps, size, model_mask=model_mask,
            use_parsing=self.opts.use_parsing and mode in ("parsing", "full"),
            use_occlusion=self.opts.use_occlusion and mode == "full",
            face_size=face.size)
        self.mask_sources = used

        if self.opts.temporal:
            self.last_mask_jitter = self.mask_smoother.jitter(mask)
            mask = self.mask_smoother(mask)

        if self.cfg.color_match and self.opts.color_match:
            target_crop, _ = alignment.warp(
                frame, face.kps,
                get_model(self.cfg.swapper).template, size)
            # Measure, smooth, then apply ONCE. This used to apply the raw
            # correction and then apply the smoothed correction on top of the
            # already-corrected patch -- a double colour push that both cost
            # a second LAB round-trip and drove the face further from the
            # target's lighting than the smoothed parameters asked for.
            params = color.estimate(patch, target_crop, mask)
            if self.opts.temporal:
                params = self.color_smoother(params)
            patch = color.apply(patch, mask, params)

        out = alignment.paste_back(frame, patch, mask, matrix)

        if self.cfg.enhancer and self.cfg.enhancer_blend > 0:
            out = restoration.restore(out, face.kps, self.cfg.enhancer,
                                      self.cfg.enhancer_blend, face.size)

        # Full-frame mask, for seam/colour metrics.
        inv = cv2.invertAffineTransform(matrix)
        full = cv2.warpAffine(mask, inv, (frame.shape[1], frame.shape[0]))
        return out, full



# ---------------------------------------------------------------- full render
def render(source_paths: list[str], target_path: str, output_path: str,
           opts: RenderOptions, on_progress: Optional[ProgressFn] = None,
           should_cancel: Optional[Callable[[], bool]] = None,
           identity: Optional[SourceIdentity] = None) -> RenderResult:
    """Render a whole video. The only entry point the job layer needs."""
    t_start = time.time()
    res = RenderResult()

    def emit(stage: str, pct: float, **extra: Any) -> None:
        if on_progress:
            on_progress(stage, float(np.clip(pct, 0.0, 100.0)), extra)

    emit("preparing", 0.0)
    info = video.probe(target_path)
    res.video = info.as_dict()

    # Resolve the requested window against the real duration, so a range that
    # runs off the end of the file clamps instead of producing zero frames.
    src_dur = info.duration or 0.0
    start = max(0.0, float(opts.start_time or 0.0))
    end = float(opts.end_time) if opts.end_time else (src_dur or 0.0)
    if src_dur:
        end = min(end, src_dur)
    span = max(0.0, end - start) if end else 0.0
    ranged = bool(start > 0 or (span and src_dur and span < src_dur - 1e-3))
    if ranged and span <= 0:
        raise RenderError(
            f"the selected range is empty (start {start:.3f}s, end {end:.3f}s, "
            f"source is {src_dur:.3f}s long)")
    res.video["range"] = {"start": round(start, 3), "end": round(end, 3),
                          "duration": round(span, 3)} if ranged else None

    if identity is None:
        identity = load_source(source_paths)
    t_prep = time.time()

    cfg = opts.config or PipelineConfig(swapper="hyperswap_1a_256")
    res.config = cfg

    # Verify CUDA really bound before committing to a long render.
    res.gpu = sessions.verify_cuda(get_model(cfg.swapper))
    if not res.gpu.get("cuda_active"):
        raise RenderError(
            "CUDA provider did not bind -- refusing to render on CPU. "
            f"session providers: {res.gpu.get('session_providers')}")

    renderer = FrameRenderer(cfg, opts, identity, float(np.hypot(info.width, info.height)))
    from app.render.tracking import FaceTracker, TargetLock
    diag = float(np.hypot(info.width, info.height))
    lock = None if opts.swap_all_faces else TargetLock(diag, opts.identity_floor)
    tracker = FaceTracker(diag, opts.identity_floor) if opts.swap_all_faces else None

    cut_detector = metrics.SceneCutDetector()
    dec = video.decode(target_path, info,
                       start=start if ranged else None,
                       duration=span if ranged else None)
    tmp = str(Path(output_path).with_suffix(".silent.mp4"))
    enc, enc_name = video.open_encoder(tmp, info, "fast" if opts.quality == "fast" else "quality")
    res.encoder = enc_name
    if info.is_vfr:
        # Declared, not silent: the brief requires any quality-affecting
        # fallback to be recorded on the job rather than hidden.
        res.fallbacks.append(
            f"source is variable-frame-rate; encoded at the average {info.fps:.3f} fps "
            "(constant). Total duration and A/V sync are preserved; per-frame "
            "intervals are not.")
    if info.rotation:
        res.fallbacks.append(f"source carries {info.rotation}deg rotation metadata; "
                             "applied during decode.")

    # Progress must count the frames we will ACTUALLY render, not the whole
    # file, or a 2-minute range out of an hour would sit near 3% and finish.
    total = opts.max_frames or (
        int(round(span * info.fps)) if ranged else info.total_frames)
    done = 0
    prev_frame: Optional[np.ndarray] = None
    cancelled = False
    jitters: list[float] = []
    single_person = not opts.swap_all_faces
    frames_since_embed = 0
    skipped_embeds = 0

    try:
        for frame in video.frames(dec, info):
            if should_cancel and should_cancel():
                cancelled = True
                break
            if opts.max_frames and done >= opts.max_frames:
                break

            work = frame
            if prev_frame is not None and cut_detector(prev_frame, frame):
                res.scene_cuts += 1
                frames_since_embed = EMBED_EVERY      # force a real check
                renderer.reset_temporal()
                if lock:
                    lock.reset_motion()
                if tracker:
                    tracker.reset_motion()

            faces = detection.detect_with_fallback(
                frame, opts.detect_threshold, opts.quality,
                opts.detector, opts.detector_fallback)
            # Identity embedding costs ~30 ms of a ~160 ms frame. On
            # single-person footage the common case is exactly one detection
            # sitting where the tracker predicted, and re-deriving "is this
            # still them?" every frame buys nothing. So we skip it on that
            # fast path -- but only for a bounded run of frames, and never
            # when the geometry disagrees. Anything ambiguous (more than one
            # face, a jump, a re-acquisition) falls back to a real embedding,
            # because the alternative is swapping the wrong thing to save
            # 30 ms. Not used in swap_all_faces mode.
            embeds: list[Optional[np.ndarray]] = []
            fast_path = (
                single_person and len(faces) == 1 and lock is not None
                and lock.locked_id is not None
                and frames_since_embed < EMBED_EVERY
                and lock.predicted_iou(faces[0]) >= 0.55
            )
            if fast_path:
                embeds = [lock.locked_embedding]
                faces[0].embedding = lock.locked_embedding
                frames_since_embed += 1
                skipped_embeds += 1
            else:
                for f in faces:
                    try:
                        e = recognition.embed(frame, f.kps)
                    except RenderError:
                        e = None
                    f.embedding = e
                    embeds.append(e)
                frames_since_embed = 0

            chosen: list[Face] = []
            if opts.swap_all_faces and tracker is not None:
                chosen = list(tracker.update(faces, embeds).values())
            elif lock is not None:
                picked = lock.select(faces, embeds)
                chosen = [picked] if picked is not None else []

            for f in chosen:
                try:
                    work, _ = renderer.render_face(work, f)
                    res.faces_swapped += 1
                    jitters.append(renderer.last_mask_jitter)
                except RenderError:
                    # One bad face must not abort a whole render.
                    pass

            try:
                enc.stdin.write(np.ascontiguousarray(work).tobytes())
            except (BrokenPipeError, OSError) as e:
                err = enc.stderr.read().decode("utf-8", "replace")[:300] if enc.stderr else ""
                raise RenderError(f"encoder failed: {e}. {err}") from e

            prev_frame = frame
            done += 1
            if total:
                emit("processing", done / total * 100.0,
                     frames=done, total=total,
                     fps=done / max(time.time() - t_prep, 1e-6))
            elif done % 30 == 0:
                emit("processing", 0.0, frames=done, total=0)
    finally:
        try:
            if enc.stdin:
                enc.stdin.close()
        except OSError:
            pass
        try:
            enc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            enc.kill()
        try:
            dec.kill()
        except OSError:
            pass

    res.frames = done
    res.mask_sources = renderer.mask_sources
    if lock:
        res.tracking = lock.report()
    elif tracker:
        res.tracking = dict(tracker.stats)
    if jitters:
        res.tracking["mask_jitter_mean"] = round(float(np.mean(jitters)), 5)
    if skipped_embeds:
        res.tracking["identity_checks_skipped"] = skipped_embeds

    if cancelled:
        Path(tmp).unlink(missing_ok=True)
        raise RenderError("cancelled")
    if done == 0:
        Path(tmp).unlink(missing_ok=True)
        raise RenderError(
            f"no frames could be decoded from the target video (codec {info.codec}). "
            "The file may be corrupt or use an unsupported codec.")
    if not Path(tmp).exists() or Path(tmp).stat().st_size == 0:
        err = enc.stderr.read().decode("utf-8", "replace")[:400] if enc.stderr else ""
        raise RenderError(f"encoding produced no output. {err}")

    emit("encoding", 99.0, frames=done, total=total or done)
    t_render = time.time()

    emit("restoring audio", 99.5)
    res.video.update(video.finalize(
        tmp, target_path, output_path, info,
        start=start if ranged else None,
        duration=span if ranged else None))

    res.timings = {
        "prepare_s": round(t_prep - t_start, 2),
        "render_s": round(t_render - t_prep, 2),
        "total_s": round(time.time() - t_start, 2),
        "ms_per_frame": round((t_render - t_prep) * 1000.0 / max(done, 1), 2),
    }
    res.gpu.update(sessions.gpu_info())
    emit("complete", 100.0, frames=done, total=total or done)
    return res
