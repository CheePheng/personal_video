"""Long-video rendering: segmented, checkpointed, resumable.

An hour of footage is roughly 108,000 frames. At the ~180-270 ms/frame this
pipeline measures, that is several hours of GPU time -- long enough that
"the machine rebooted at frame 100,000" is a realistic event rather than a
hypothetical one. Restarting from zero after that is unacceptable, so a long
render is not one long operation: it is a sequence of independent segments,
each finished and recorded before the next begins.

    [ segment 000 ] [ segment 001 ] [ segment 002 ] ...   -> concat -> master

A manifest on disk records which segments are done and the fingerprint of the
settings that produced them. Re-running the same job skips completed work; a
crash costs at most one segment. Joining is a stream copy, so splitting the
work costs no quality.

There is deliberately NO duration or file-size cap. The only hard refusal is
a concrete, measured one -- not enough disk space -- because "this file is too
long" is a judgement the user is better placed to make than the software.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from app.render import pipeline, video
from app.render.types import RenderError, SourceIdentity

# Videos shorter than this render in one pass; segmenting them would add
# concat overhead and manifest bookkeeping for no benefit.
SEGMENT_THRESHOLD_S = 180.0
# Default segment length. Long enough that per-segment overhead (process
# warm-up, concat) stays negligible; short enough that losing one to a crash
# is a couple of minutes, not an hour.
DEFAULT_SEGMENT_S = 90.0
# Rough bytes-per-second of rendered H.264 at the qualities this encodes at,
# used only to estimate whether the disk can hold the result. Deliberately
# generous: refusing a job that would have fitted is worse than warning late.
EST_BYTES_PER_SECOND = 3_500_000


def _settings_fingerprint(source_paths: list[str], target_path: str,
                          opts: pipeline.RenderOptions) -> str:
    """Identify the settings a segment was rendered with.

    If anything that affects output changes -- the source photos, the target,
    the chosen pipeline -- previously completed segments are no longer valid
    and must not be silently reused.
    """
    h = hashlib.sha256()
    for p in sorted(source_paths):
        st = Path(p).stat()
        h.update(f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}|".encode())
    tst = Path(target_path).stat()
    h.update(f"{Path(target_path).name}:{tst.st_size}:{int(tst.st_mtime)}|".encode())
    cfg = asdict(opts.config) if opts.config else {}
    h.update(json.dumps(cfg, sort_keys=True).encode())
    h.update(f"{opts.quality}|{opts.swap_all_faces}|{opts.detect_threshold}|"
             f"{opts.use_parsing}|{opts.use_occlusion}|{opts.color_match}|"
             f"{opts.temporal}|{opts.detector}|{opts.detector_fallback}".encode())
    return h.hexdigest()[:24]


def plan_segments(start: float, end: float,
                  segment_s: float = DEFAULT_SEGMENT_S) -> list[tuple[float, float]]:
    """Split a time window into consecutive segments.

    A trailing stub shorter than a third of a segment is folded into the
    previous one, so a 91-second clip is one segment rather than 90 + 1.
    """
    span = max(0.0, end - start)
    if span <= 0:
        return []
    out: list[tuple[float, float]] = []
    t = start
    while t < end - 1e-6:
        seg_end = min(t + segment_s, end)
        out.append((t, seg_end))
        t = seg_end
    if len(out) > 1 and (out[-1][1] - out[-1][0]) < segment_s / 3:
        a, _ = out[-2]
        _, b = out[-1]
        out[-2:] = [(a, b)]
    return out


class Manifest:
    """On-disk record of which segments are finished."""

    def __init__(self, path: Path, fingerprint: str):
        self.path = path
        self.fingerprint = fingerprint
        self.data: dict[str, Any] = {"fingerprint": fingerprint, "segments": {}}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                # A fingerprint mismatch means the settings changed; the old
                # segments are stale and reusing them would silently mix two
                # different renders into one file.
                if loaded.get("fingerprint") == fingerprint:
                    self.data = loaded
            except (OSError, ValueError):
                pass

    def done(self, index: int, part: Path) -> bool:
        rec = self.data["segments"].get(str(index))
        if not rec or not part.is_file():
            return False
        # Size is recorded so a half-written file from a crash mid-write is
        # not mistaken for a completed segment.
        return part.stat().st_size == rec.get("bytes") and rec.get("bytes", 0) > 0

    def record(self, index: int, part: Path, frames: int, seconds: float) -> None:
        self.data["segments"][str(index)] = {
            "bytes": part.stat().st_size, "frames": frames,
            "render_s": round(seconds, 2), "file": part.name,
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    @property
    def completed(self) -> int:
        return len(self.data["segments"])


def estimate(target_path: str, opts: pipeline.RenderOptions,
             ms_per_frame: float = 220.0) -> dict[str, Any]:
    """Estimate cost before committing to a long job.

    Returned, not enforced. The user decides whether hours of rendering is
    worth it; the software's job is to say so up front rather than discover it
    at 80%.
    """
    info = video.probe(target_path)
    src_dur = info.duration or 0.0
    start = max(0.0, float(opts.start_time or 0.0))
    end = min(float(opts.end_time), src_dur) if opts.end_time else src_dur
    span = max(0.0, end - start)

    frames = int(round(span * info.fps))
    est_s = frames * ms_per_frame / 1000.0
    need = int(span * EST_BYTES_PER_SECOND * 2)      # output + segment copies
    free = video.free_disk_bytes(target_path)

    return {
        "source_duration_s": round(src_dur, 2),
        "render_duration_s": round(span, 2),
        "frames": frames,
        "fps": info.fps,
        "resolution": f"{info.width}x{info.height}",
        "estimated_render_s": int(est_s),
        "estimated_render_human": _human_time(est_s),
        "estimated_disk_bytes": need,
        "free_disk_bytes": free,
        "disk_ok": free > need,
        "segments": len(plan_segments(start, end)) if span > SEGMENT_THRESHOLD_S else 1,
    }


def _human_time(seconds: float) -> str:
    s = int(seconds)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


def render_long(source_paths: list[str], target_path: str, output_path: str,
                opts: pipeline.RenderOptions,
                on_progress: Optional[pipeline.ProgressFn] = None,
                should_cancel: Optional[Callable[[], bool]] = None,
                identity: Optional[SourceIdentity] = None,
                segment_s: float = DEFAULT_SEGMENT_S) -> pipeline.RenderResult:
    """Render a video, segmenting and checkpointing when it is long enough.

    Short videos fall straight through to ``pipeline.render``; the segmenting
    machinery only engages where it earns its keep.
    """
    info = video.probe(target_path)
    src_dur = info.duration or 0.0
    start = max(0.0, float(opts.start_time or 0.0))
    end = min(float(opts.end_time), src_dur) if opts.end_time else src_dur
    span = max(0.0, end - start)

    if span <= 0 and src_dur:
        raise RenderError(
            f"the selected range is empty (start {start:.2f}s, end {end:.2f}s, "
            f"source is {src_dur:.2f}s)")

    # Refuse only on a concrete, measured shortage -- never on duration.
    est = estimate(target_path, opts)
    if not est["disk_ok"]:
        raise RenderError(
            f"insufficient disk space: this render needs roughly "
            f"{est['estimated_disk_bytes'] / 1e9:.1f} GB of temporary and output "
            f"space, but only {est['free_disk_bytes'] / 1e9:.1f} GB is free. "
            f"Free some space, choose a shorter range, or render to another drive.")

    if span <= SEGMENT_THRESHOLD_S or span <= 0:
        return pipeline.render(source_paths, target_path, output_path, opts,
                               on_progress, should_cancel, identity)

    segments = plan_segments(start, end, segment_s)
    out = Path(output_path)
    work = out.parent / f"{out.stem}_segments"
    work.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(work / "manifest.json",
                        _settings_fingerprint(source_paths, target_path, opts))

    # The identity is built once and reused. Re-deriving it per segment would
    # be wasted work and -- worse -- could drift between segments.
    if identity is None:
        identity = pipeline.load_source(source_paths)

    total_frames = int(round(span * info.fps))
    combined = pipeline.RenderResult()
    parts: list[str] = []
    frames_done = 0
    t0 = time.time()

    for i, (seg_start, seg_end) in enumerate(segments):
        part = work / f"seg_{i:04d}.mp4"
        seg_frames = int(round((seg_end - seg_start) * info.fps))

        if manifest.done(i, part):
            parts.append(str(part))
            frames_done += manifest.data["segments"][str(i)].get("frames", seg_frames)
            if on_progress:
                on_progress("processing", frames_done / max(total_frames, 1) * 100.0,
                            {"frames": frames_done, "total": total_frames,
                             "segment": i + 1, "segments": len(segments),
                             "resumed": True})
            continue

        if should_cancel and should_cancel():
            raise RenderError("cancelled")

        seg_opts = pipeline.RenderOptions(**{**opts.__dict__})
        seg_opts.start_time = seg_start
        seg_opts.end_time = seg_end

        base = frames_done

        def seg_progress(stage: str, pct: float, extra: dict,
                         _base: int = base, _n: int = i) -> None:
            if on_progress is None:
                return
            if stage == "processing":
                f = _base + int(extra.get("frames") or 0)
                on_progress("processing", f / max(total_frames, 1) * 100.0,
                            {**extra, "frames": f, "total": total_frames,
                             "segment": _n + 1, "segments": len(segments)})
            elif stage not in ("complete", "encoding", "restoring audio"):
                on_progress(stage, pct, {**extra, "segment": _n + 1,
                                         "segments": len(segments)})

        seg_t = time.time()
        # Segments are rendered WITHOUT audio; the soundtrack is muxed once at
        # the end from the original, which keeps A/V sync exact and avoids
        # concatenating many small audio fragments.
        r = pipeline.render(source_paths, target_path, str(part), seg_opts,
                            seg_progress, should_cancel, identity)

        manifest.record(i, part, r.frames, time.time() - seg_t)
        parts.append(str(part))
        frames_done += r.frames

        combined.faces_swapped += r.faces_swapped
        combined.scene_cuts += r.scene_cuts
        combined.config = r.config
        combined.gpu = r.gpu
        combined.encoder = r.encoder
        combined.mask_sources = r.mask_sources
        for k, v in (r.tracking or {}).items():
            if isinstance(v, (int, float)):
                combined.tracking[k] = combined.tracking.get(k, 0) + v

    if should_cancel and should_cancel():
        raise RenderError("cancelled")

    if on_progress:
        on_progress("encoding", 99.0, {"frames": frames_done, "total": total_frames,
                                       "segments": len(segments)})

    silent = str(out.with_suffix(".silent.mp4"))
    video.concat(parts, silent)

    if on_progress:
        on_progress("restoring audio", 99.5, {})
    combined.frames = frames_done
    combined.video = info.as_dict()
    combined.video["range"] = {"start": round(start, 3), "end": round(end, 3),
                               "duration": round(span, 3)}
    combined.video["segments"] = len(segments)
    combined.video.update(video.finalize(
        silent, target_path, output_path, info,
        start=start if (start > 0 or span < src_dur - 1e-3) else None,
        duration=span if (start > 0 or span < src_dur - 1e-3) else None))

    combined.timings = {
        "total_s": round(time.time() - t0, 2),
        "ms_per_frame": round((time.time() - t0) * 1000.0 / max(frames_done, 1), 2),
    }

    # Segment files are only useful for resuming; once joined they are dead
    # weight, and on an hour of 4K that is tens of gigabytes.
    for p in parts:
        Path(p).unlink(missing_ok=True)
    manifest.path.unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass

    if on_progress:
        on_progress("complete", 100.0, {"frames": frames_done, "total": total_frames})
    return combined
