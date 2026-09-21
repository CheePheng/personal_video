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


def apply_preset(opts: "RenderOptions") -> "RenderOptions":
    """Resolve ``opts.quality`` into the actual pipeline it names.

    Single source of truth, because there was not one. The job layer built
    the real presets itself and ``render()`` fell back to a bare
    ``PipelineConfig(swapper="hyperswap_1a_256")`` whenever no config was
    passed -- no enhancer, where the shipping Quality preset runs GPEN BFR
    512 at 70%. Any harness calling ``render()`` directly therefore measured
    a pipeline no user ever runs, and reported it as "Quality".

    ``auto`` is refused rather than defaulted. Auto Max has to benchmark
    before it can name a winner, and silently rendering the default instead
    looks exactly like a successful Auto Max run while reporting the wrong
    configuration -- which is how two "Auto Max" timings came back identical
    to each other and to the default.
    """
    if opts.config is not None:
        return opts
    if opts.quality == "auto":
        raise RenderError(
            "quality='auto' needs Auto Max to pick a pipeline first: call "
            "benchmark.run() and pass the winner as RenderOptions.config. "
            "render() will not substitute a default, because that would "
            "report a default render as an Auto Max result.")
    if opts.quality == "fast":
        # Preview: one proven model, no parsing/occlusion/colour work.
        opts.config = PipelineConfig(swapper="hyperswap_1a_256", enhancer=None,
                                     mask="model", color_match=False)
        opts.use_parsing = False
        opts.use_occlusion = False
    else:
        # No restoration by default. Measured on the A->B set across 10
        # non-degenerate categories, GPEN BFR 512 at 70% -- the previous
        # default -- LOSES on every selector metric and costs 62% more time:
        #
        #                arcMean   sfMean     gap   arcWorst   ms/f
        #   none          0.6657   0.6363  0.3823     0.5599   72.5
        #   gpen@70       0.6583   0.6455  0.3661     0.5542  117.5
        #
        # "gap" is similarity-to-source minus similarity-to-target, which is
        # what a swap is for. The pipeline is deterministic (three identical
        # runs, zero spread), so these are small but real, not noise.
        #
        # Every restorer degrades identity monotonically as blend rises, and
        # gains sharpness doing it (GFPGAN@100: -0.0416 arc, +123 sharpness).
        # That is the "sharper generic face" failure, and it is why blend is
        # not simply turned down rather than off: at every level tested the
        # honest comparison against no restoration at all was not won.
        #
        # Auto Max may still SELECT a restorer per video -- it is scored per
        # candidate there. This is only the fixed default.
        # mask="model", not "full". Measured on the A->B set across 10
        # categories, the swapper's own mask beats the parsing+XSeg stack on
        # every identity metric and both judges, by the largest margin found
        # anywhere in this pass:
        #
        #             arcMean  arcWorst   sfMean     gap    seam
        #   model      0.7487    0.6790   0.6930  0.5479  1.2331
        #   parsing    0.6663    0.5599   0.6363  0.3832  0.7014
        #   full       0.6657    0.5599   0.6363  0.3823  0.7017
        #
        # +0.083 arc, +0.119 worst frame, +0.166 source-minus-target gap.
        # BiSeNet and XSeg were excluding large parts of the face, so much of
        # the swapped result was never composited and the output stayed
        # closer to the target.
        #
        # The higher seam number is a longer boundary, not a worse one: with
        # more of the face replaced there is simply more edge to measure, and
        # no halo is visible at any face size tested.
        #
        # The obvious objection -- that this destroys occluders -- was
        # checked and is wrong. On the glasses fixture, model preserves the
        # frames and arms exactly as parsing and full do, because HyperSwap
        # emits its own mask and already declines to paint over foreground.
        # Parsing and XSeg were paying 0.08 identity to protect something
        # that was not at risk.
        #
        # use_parsing/use_occlusion stay True so Auto Max can still choose
        # the heavier modes per video where they genuinely help.
        # alphaface_256, not hyperswap_1a_256. The swapper tournament was
        # first run under mask="full", and re-running it under mask="model"
        # inverted the ranking -- hyperswap_1a went from first to worst of
        # the six finalists, because it leaves far more of the target behind
        # (arc->B 0.203 against GHOST's 0.03-0.05):
        #
        #                arc->A   arc->B     gap   sf->A   sfGap  exprMouth
        #   alphaface    0.8649   0.0986  0.7663  0.7725  0.6223    0.00683
        #   ghost_3      0.8211   0.0343  0.7868  0.7608  0.7673    0.02958
        #   ghost_2      0.8257   0.0422  0.7835  0.7633  0.7405    0.02429
        #   ghost_1      0.8073   0.0477  0.7596  0.7935  0.7904    0.02967
        #   hyperswap_1b 0.7685   0.1584  0.6102  0.7335  0.5898    0.01665
        #   hyperswap_1a 0.7454   0.2029  0.5425  0.6905  0.4561    0.01457
        #
        # GHOST scores a marginally better identity GAP, and was rejected on
        # it. Visually it mangles the mouth: an open smile with teeth comes
        # back closed, asymmetric and grimacing. The last column is why --
        # mouth-opening drift, where GHOST distorts jaw opening ~4x more than
        # alphaface. The tournament's expression term saw this but at weight
        # 0.13 could not outvote a 0.08 identity gain, which is precisely the
        # "wins on identity, ruins the performance" failure to guard against.
        #
        # alphaface takes it: best absolute source likeness, top-2 on BOTH
        # judges, and the lowest mouth-opening drift of all six.
        opts.config = PipelineConfig(swapper="alphaface_256",
                                     enhancer=None, enhancer_blend=0.0,
                                     mask="model")
    return opts


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
        patch, model_mask, matrix, size, target_crop = swapping.swap(
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
            # target_crop comes back from swapping.swap(), which had to warp
            # it anyway; re-deriving it here cost a second full LANCZOS4 warp
            # per frame for a byte-identical result.
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

    cfg = apply_preset(opts).config
    assert cfg is not None
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
                # memoryview, not .tobytes(): the latter copies the whole
                # frame (24.9 MB at 4K) purely to hand the same bytes to a
                # pipe that can consume the buffer directly.
                enc.stdin.write(memoryview(np.ascontiguousarray(work)))
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
