"""Video I/O: probe, decode, encode, mux.

Frames move over raw pipes as BGR24 -- no JPEG/PNG round-trip anywhere, so the
only generation loss in the whole pipeline is the single final encode.

Three things that quietly break video tools, handled explicitly here:

  * **Rotation metadata.** Phone footage is often stored landscape with a 90-deg
    display matrix. Decoding raw ignores that, so a portrait video would render
    sideways. We read the side-data and let ffmpeg apply it on decode.
  * **Variable frame rate.** Decoding to a raw pipe discards per-frame
    timestamps, so the output is necessarily constant-rate. We detect VFR and
    encode at the measured AVERAGE rate, which keeps total duration and
    therefore A/V sync correct end to end -- the same conversion an NLE does on
    import. Individual inter-frame intervals are not preserved, so the
    conversion is recorded as a declared fallback on the job rather than done
    silently.
  * **Audio.** Copied rather than re-encoded when it is already browser-safe,
    which is both lossless and faster.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from app.render.types import RenderError

# Never shell=True: every argument is passed as a list element.
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


@dataclass(slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: Optional[float]
    total_frames: int
    has_audio: bool
    audio_codec: Optional[str]
    codec: str
    rotation: int
    is_vfr: bool
    pix_fmt: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height, "fps": self.fps,
            "duration": self.duration, "total_frames": self.total_frames,
            "has_audio": self.has_audio, "audio_codec": self.audio_codec,
            "codec": self.codec, "rotation": self.rotation,
            "is_vfr": self.is_vfr, "pix_fmt": self.pix_fmt,
        }


def _tools_present() -> None:
    for tool in (FFMPEG, FFPROBE):
        if shutil.which(tool) is None:
            raise RenderError(
                f"{tool} not found on PATH. Install it: winget install Gyan.FFmpeg")


def probe(path: str) -> VideoInfo:
    """Read geometry/timing. Raises with the real reason on unreadable media."""
    _tools_present()
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise RenderError(f"ffprobe failed to run: {type(e).__name__}: {e}") from e
    if r.returncode != 0:
        raise RenderError(f"unreadable or corrupt media: {r.stderr.strip()[:300]}")

    try:
        info = json.loads(r.stdout or "{}")
    except ValueError as e:
        raise RenderError(f"ffprobe returned unparseable output: {e}") from e

    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise RenderError("no video stream found in the target file")
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

    def rate(key: str) -> float:
        num, _, den = (video.get(key) or "0/1").partition("/")
        try:
            d = float(den or 1)
            return float(num) / d if d else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    r_fps = rate("r_frame_rate")        # container's nominal rate
    avg_fps = rate("avg_frame_rate")    # actual average over the file
    fps = avg_fps or r_fps or 25.0
    # A meaningful gap between nominal and average means variable timing.
    is_vfr = bool(r_fps and avg_fps and abs(r_fps - avg_fps) / max(r_fps, 1e-6) > 0.01)

    duration = None
    for src in (video.get("duration"), info.get("format", {}).get("duration")):
        try:
            duration = float(src)
            break
        except (TypeError, ValueError):
            continue

    total = 0
    for key in ("nb_frames", "nb_read_frames"):
        try:
            total = int(video.get(key) or 0)
            if total:
                break
        except (TypeError, ValueError):
            pass
    if total <= 0 and duration:
        total = int(round(fps * duration))

    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    if not rotation:
        try:
            rotation = int(float((video.get("tags") or {}).get("rotate", 0))) % 360
        except (TypeError, ValueError):
            rotation = 0

    w, h = int(video["width"]), int(video["height"])
    # A 90/270 display rotation means the decoded frames come out transposed.
    if rotation in (90, 270):
        w, h = h, w

    return VideoInfo(
        width=w, height=h, fps=round(fps, 6), duration=duration,
        total_frames=max(total, 0), has_audio=audio is not None,
        audio_codec=(audio or {}).get("codec_name"),
        codec=video.get("codec_name", "?"), rotation=rotation, is_vfr=is_vfr,
        pix_fmt=video.get("pix_fmt"))


def decode(path: str, info: VideoInfo,
           start: Optional[float] = None,
           duration: Optional[float] = None) -> subprocess.Popen:
    """Open a raw BGR24 frame pipe, optionally for a sub-range only.

    ``-ss`` goes BEFORE ``-i`` so ffmpeg seeks the input rather than decoding
    and discarding everything up to the start point -- on an hour-long file
    that is the difference between seconds and minutes. Because we decode
    rather than stream-copy, modern ffmpeg makes that seek frame-accurate.

    ``-t`` (duration) rather than ``-to``: after an input seek the output
    clock restarts at zero, so an absolute end time would mean the wrong
    thing.

    ``-autorotate`` is a BOOLEAN flag and must precede ``-i`` to bind to that
    input; passing it a value makes ffmpeg read the value as an output URL and
    abort. It is already the default, stated explicitly so a future default
    change cannot silently start rendering phone video sideways.
    """
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-autorotate"]
    if start and start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if duration and duration > 0:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        raise RenderError(f"could not start decoder: {e}") from e


def frames(proc: subprocess.Popen, info: VideoInfo) -> Iterator[np.ndarray]:
    """Yield decoded frames until the stream ends."""
    n = info.width * info.height * 3
    stdout = proc.stdout
    assert stdout is not None
    while True:
        buf = stdout.read(n)
        if not buf or len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)


def encoder_args(quality: str = "quality") -> tuple[list[str], str]:
    """Pick an encoder, preferring NVENC but tuned for quality, not speed.

    Returns (args, name). NVENC keeps the encode off the CPU while the GPU is
    already busy with inference; p7/VBR-HQ is its quality-oriented setting, not
    the fast default.
    """
    try:
        r = subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=30)
        have_nvenc = "h264_nvenc" in r.stdout
    except (OSError, subprocess.SubprocessError):
        have_nvenc = False

    if quality == "fast":
        if have_nvenc:
            return (["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                     "-cq", "25", "-b:v", "0"], "h264_nvenc/p4")
        return (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"], "libx264/veryfast")

    if have_nvenc:
        return (["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq",
                 "-rc", "vbr", "-cq", "19", "-b:v", "0",
                 "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32",
                 "-bf", "3", "-profile:v", "high"], "h264_nvenc/p7-hq")
    return (["-c:v", "libx264", "-preset", "slow", "-crf", "17",
             "-profile:v", "high"], "libx264/slow")


def open_encoder(out_path: str, info: VideoInfo, quality: str = "quality"
                 ) -> tuple[subprocess.Popen, str]:
    args, name = encoder_args(quality)
    cmd = [
        FFMPEG, "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{info.width}x{info.height}", "-r", f"{info.fps}", "-i", "-",
        # Constant rate at the measured average: the raw pipe carries no
        # timestamps, so this is what keeps total duration (and thus A/V sync)
        # correct for VFR sources.
        "-vsync", "cfr",
        *args,
        # yuv420p: the only chroma format every browser decodes reliably.
        "-pix_fmt", "yuv420p", str(out_path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE), name
    except OSError as e:
        raise RenderError(f"could not start encoder: {e}") from e


def finalize(silent_video: str, original: str, out_path: str,
             info: VideoInfo, start: Optional[float] = None,
             duration: Optional[float] = None) -> dict[str, Any]:
    """Mux audio back and make the file seekable.

    Audio is stream-copied when it is already AAC/MP3 -- lossless and fast.
    +faststart moves the index to the front so a browser can seek before the
    whole file has arrived.
    """
    result: dict[str, Any] = {"audio": "none", "faststart": False}

    if info.has_audio:
        copyable = (info.audio_codec or "").lower() in ("aac", "mp3")
        acodec = ["-c:a", "copy"] if copyable else ["-c:a", "aac", "-b:a", "192k"]
        # The audio must be cut to exactly the same window as the video, or a
        # ranged render ends up with the soundtrack from the start of the file.
        audio_in = ["-i", str(original)]
        if start and start > 0:
            audio_in = ["-ss", f"{start:.6f}"] + audio_in
        if duration and duration > 0:
            # Trim the audio to the RENDERED video's length, not the length
            # that was requested. A 6.000s request yields 144 frames, and 144
            # frames at 23.839 fps is 6.041s -- so "-t 6.0" makes the audio
            # SHORTER than the video, and -shortest then truncates the video
            # to match, discarding 22 frames. Measured: concat produced 144,
            # the final file had 122.
            #
            # The rendered frame count is authoritative; the audio follows it.
            vid_len = duration
            try:
                probed = probe(str(silent_video))
                if probed.total_frames > 0 and probed.fps > 0:
                    vid_len = max(duration, probed.total_frames / probed.fps)
            except (RenderError, OSError, ValueError):
                pass
            # A small margin so rounding can never leave audio a hair short.
            audio_in += ["-t", f"{vid_len + 0.05:.6f}"]
        # NOT -shortest. The rendered video is authoritative: it contains
        # exactly the frames we decoded and swapped. On a VFR source the
        # encoded CFR video is very slightly LONGER than the original audio
        # (580 frames at the average 23.839 fps is 24.330s against 24.163s
        # of audio), and -shortest resolves that by truncating the VIDEO --
        # silently discarding the last 4 rendered frames. Measured: mux was
        # the only lossy stage in the whole pipeline, 580 -> 576.
        #
        # -apad pads the audio with silence instead, and the video stream
        # decides the length, so every rendered frame survives. The padding
        # is bounded by the video, so it cannot run away.
        cmd = [FFMPEG, "-v", "error", "-nostdin", "-y",
               "-i", str(silent_video), *audio_in,
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *acodec,
               "-apad", "-shortest", "-shortest_buf_duration", "10",
               "-movflags", "+faststart", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, timeout=3600)
        if r.returncode == 0 and Path(out_path).exists():
            result["audio"] = "copied" if copyable else "re-encoded aac"
            result["faststart"] = True
            Path(silent_video).unlink(missing_ok=True)
            return result
        # Audio is a bonus; never lose the render over a mux failure.
        result["audio"] = f"mux failed: {r.stderr.decode('utf-8','replace')[:200]}"

    r = subprocess.run(
        [FFMPEG, "-v", "error", "-nostdin", "-y", "-i", str(silent_video),
         "-c", "copy", "-movflags", "+faststart", str(out_path)],
        capture_output=True, timeout=1800)
    if r.returncode == 0 and Path(out_path).exists():
        result["faststart"] = True
        Path(silent_video).unlink(missing_ok=True)
    else:
        Path(silent_video).replace(out_path)
    return result


def concat(parts: list[str], out_path: str) -> None:
    """Join rendered segments losslessly with the concat demuxer.

    Stream copy, so joining costs no quality and no re-encode -- which is the
    whole point of rendering a long video in resumable pieces.
    """
    listing = Path(out_path).with_suffix(".parts.txt")
    listing.write_text(
        "".join("file '" + Path(p).as_posix() + "'" + chr(10) for p in parts),
        encoding="utf-8")
    try:
        r = subprocess.run(
            [FFMPEG, "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", str(out_path)],
            capture_output=True, timeout=3600)
        if r.returncode != 0 or not Path(out_path).exists():
            raise RenderError(
                "could not join rendered segments: "
                f"{r.stderr.decode('utf-8', 'replace')[:300]}")
    finally:
        listing.unlink(missing_ok=True)


def free_disk_bytes(path: str) -> int:
    return shutil.disk_usage(Path(path).parent if Path(path).suffix else path).free


def sample_frames(path: str, info: VideoInfo, indices: list[int]) -> dict[int, np.ndarray]:
    """Pull specific frames by index, for benchmarking.

    One sequential decode rather than N seeks: seeking a long-GOP H.264 file
    repeatedly is slower and can land on the wrong frame.
    """
    want = sorted(set(i for i in indices if i >= 0))
    if not want:
        return {}
    out: dict[int, np.ndarray] = {}
    proc = decode(path, info)
    try:
        target = set(want)
        last = max(want)
        for idx, frame in enumerate(frames(proc, info)):
            if idx in target:
                out[idx] = frame.copy()
            if idx >= last:
                break
    finally:
        try:
            proc.kill()
        except OSError:
            pass
    return out
