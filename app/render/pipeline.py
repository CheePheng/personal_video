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
STAGES = [("preparing", 0.02), ("analysing", 0.04), ("benchmarking", 0.10),
          ("processing", 0.76), ("encoding", 0.05), ("restoring audio", 0.03)]


@dataclass
class RenderOptions:
    quality: str = "quality"                  # fast | quality | auto
    swap_all_faces: bool = False
    config: Optional[PipelineConfig] = None   # explicit pipeline (auto fills it)
    detect_threshold: float = 0.5
    identity_floor: float = 0.28
    use_parsing: bool = True
    use_occlusion: bool = True
    color_match: bool = True
    temporal: bool = True
    max_frames: Optional[int] = None          # benchmarking/tests only


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
        d["config"] = self.config.__dict__ if self.config else None
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

        mask, used = masking.build(
            frame, face.kps, size, model_mask=model_mask,
            use_parsing=self.opts.use_parsing and self.cfg.mask in ("model", "parsing"),
            use_occlusion=self.opts.use_occlusion and self.cfg.mask in ("model", "parsing"),
            face_size=face.size)
        self.mask_sources = used

        if self.opts.temporal:
            self.last_mask_jitter = self.mask_smoother.jitter(mask)
            mask = self.mask_smoother(mask)

        if self.cfg.color_match and self.opts.color_match:
            target_crop, _ = alignment.warp(
                frame, face.kps,
                get_model(self.cfg.swapper).template, size)
            patch, params = color.match(patch, target_crop, mask)
            if self.opts.temporal:
                smoothed = self.color_smoother(params)
                patch, _ = color.match(patch, target_crop, mask, smoothed)

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

    dec = video.decode(target_path, info)
    tmp = str(Path(output_path).with_suffix(".silent.mp4"))
    enc, enc_name = video.open_encoder(tmp, info, "fast" if opts.quality == "fast" else "quality")
    res.encoder = enc_name

    total = opts.max_frames or info.total_frames
    done = 0
    prev_frame: Optional[np.ndarray] = None
    cancelled = False
    jitters: list[float] = []

    try:
        for frame in video.frames(dec, info):
            if should_cancel and should_cancel():
                cancelled = True
                break
            if opts.max_frames and done >= opts.max_frames:
                break

            work = frame
            if prev_frame is not None and metrics.scene_cut(prev_frame, frame):
                res.scene_cuts += 1
                renderer.reset_temporal()
                if lock:
                    lock.reset_motion()
                if tracker:
                    tracker.reset_motion()

            faces = detection.detect_robust(frame, opts.detect_threshold, opts.quality)
            embeds: list[Optional[np.ndarray]] = []
            for f in faces:
                try:
                    e = recognition.embed(frame, f.kps)
                except RenderError:
                    e = None
                f.embedding = e
                embeds.append(e)

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
    res.video.update(video.finalize(tmp, target_path, output_path, info))

    res.timings = {
        "prepare_s": round(t_prep - t_start, 2),
        "render_s": round(t_render - t_prep, 2),
        "total_s": round(time.time() - t_start, 2),
        "ms_per_frame": round((t_render - t_prep) * 1000.0 / max(done, 1), 2),
    }
    res.gpu.update(sessions.gpu_info())
    emit("complete", 100.0, frames=done, total=total or done)
    return res
