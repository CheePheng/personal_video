"""Face swap pipeline: ONNX Runtime + OpenCV + ffmpeg. No FaceFusion.

Three ONNX models, run directly:

  yoloface_8n         detect faces          -> boxes + 5 landmarks
  arcface_w600k_r50   identity embedding    -> 512-d vector (the "who")
  hyperswap_1a_256    the swap itself       -> (embedding, face) -> new face
  gfpgan_1.4          optional restoration  -> sharpens the 256px swap to 512px

Video is decoded and re-encoded with ffmpeg via pipes, so frames never hit the
disk and progress is simply ``frames_done / total_frames`` -- a real count of
work completed, not a parsed progress bar.

Deliberately NOT here: any content/NSFW classifier. This is a local single-user
tool operating on media the user supplies. Errors are raised only for real
technical faults (no face found, unreadable media, model/CUDA failure).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parent.parent / "models"

# Landmark template hyperswap was trained against ('arcface_128'), as fractions
# of the crop. Aligning every face to this canonical layout is what lets one
# embedding drive any head pose.
ARCFACE_128 = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
], dtype=np.float32)

# arcface's own template, at 112px -- used only to build the identity vector.
ARCFACE_112_V2 = np.array([
    [0.34191607, 0.46157411],
    [0.65653393, 0.45983393],
    [0.50022500, 0.64050536],
    [0.37097589, 0.82469196],
    [0.63151696, 0.82325089],
], dtype=np.float32)

# GFPGAN was trained on FFHQ-aligned 512px crops.
FFHQ_512 = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float32)

DETECT_SIZE = 640
SWAP_SIZE = 256
EMBED_SIZE = 112
ENHANCE_SIZE = 512

# Quality presets. These change what work is actually done, not just a label:
#   detect_threshold  how faint a face still counts (lower = catches more
#                     profile/motion-blurred frames, at the cost of false hits)
#   enhance           run GFPGAN restoration over the swapped face. This is the
#                     visible difference -- the swapper outputs 256px, so on a
#                     close-up the paste-back is an upscale and looks soft.
#                     GFPGAN re-synthesises it at 512px. It roughly doubles the
#                     per-frame cost, which is the speed/quality trade.
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "fast":     {"detect_threshold": 0.60, "enhance": False},
    "balanced": {"detect_threshold": 0.50, "enhance": True},
    "best":     {"detect_threshold": 0.35, "enhance": True},
}


class SwapError(RuntimeError):
    """A real technical failure: no face, bad media, model/CUDA problem."""


# ---------------------------------------------------------------- sessions
_sessions: dict[str, ort.InferenceSession] = {}
_providers_used: list[str] = []


def _session(name: str) -> ort.InferenceSession:
    """Load a model once, preferring CUDA.

    We assert the CUDA provider actually bound rather than trusting
    ``get_available_providers()``: onnxruntime happily lists CUDA and then
    silently runs on CPU when its DLLs cannot load, turning a 2-minute render
    into an hour with no error. Surfacing that is the whole point.
    """
    if name in _sessions:
        return _sessions[name]

    path = MODELS / f"{name}.onnx"
    if not path.exists():
        raise SwapError(f"missing model: {path}")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        sess = ort.InferenceSession(
            str(path), sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as e:  # noqa: BLE001
        raise SwapError(f"could not load {name}: {type(e).__name__}: {e}") from e

    _sessions[name] = sess
    if not _providers_used:
        _providers_used.extend(sess.get_providers())
    return sess


def gpu_report() -> dict[str, Any]:
    """What the runtime can see, and what it actually bound to."""
    sess = _session("yoloface_8n")
    active = sess.get_providers()
    name = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        name = r.stdout.strip().splitlines()[0] if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return {
        "gpu_name": name,
        "available_providers": ort.get_available_providers(),
        "session_providers": active,
        "cuda_active": active and active[0] == "CUDAExecutionProvider",
    }


# ---------------------------------------------------------------- detection
def detect_faces(frame: np.ndarray, threshold: float = 0.5) -> list[dict[str, Any]]:
    """Return every face in a BGR frame as {'box','score','kps'}."""
    h, w = frame.shape[:2]
    scale = min(DETECT_SIZE / w, DETECT_SIZE / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))

    # Letterbox into a fixed 640x640 so the model's static input shape holds,
    # keeping aspect ratio so landmarks stay geometrically correct.
    canvas = np.zeros((DETECT_SIZE, DETECT_SIZE, 3), dtype=np.uint8)
    canvas[:rh, :rw] = cv2.resize(frame, (rw, rh))

    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = _session("yoloface_8n").run(None, {"input": blob})[0]
    det = np.squeeze(out).T                      # (8400, 20)
    boxes_raw, scores_raw, kps_raw = np.split(det, [4, 5], axis=1)

    keep = np.where(scores_raw.ravel() > threshold)[0]
    if keep.size == 0:
        return []
    boxes_raw, scores_raw, kps_raw = boxes_raw[keep], scores_raw[keep].ravel(), kps_raw[keep]

    inv = 1.0 / scale
    faces = []
    for b, s, k in zip(boxes_raw, scores_raw, kps_raw):
        cx, cy, bw, bh = b
        box = np.array([(cx - bw / 2) * inv, (cy - bh / 2) * inv,
                        (cx + bw / 2) * inv, (cy + bh / 2) * inv], dtype=np.float32)
        # 5 landmarks arrive interleaved (x, y, visibility).
        kps = np.stack([k[0::3] * inv, k[1::3] * inv], axis=1).astype(np.float32)
        faces.append({"box": box, "score": float(s), "kps": kps})

    # Non-max suppression: the detector fires several times per face.
    idx = cv2.dnn.NMSBoxes(
        [[float(f["box"][0]), float(f["box"][1]),
          float(f["box"][2] - f["box"][0]), float(f["box"][3] - f["box"][1])] for f in faces],
        [f["score"] for f in faces], threshold, 0.4)
    if len(idx) == 0:
        return []
    return [faces[i] for i in np.array(idx).ravel()]


def _warp(frame: np.ndarray, kps: np.ndarray, template: np.ndarray, size: int):
    """Align a face to a canonical template. Returns the crop and its matrix."""
    m, _ = cv2.estimateAffinePartial2D(kps, template * size, method=cv2.LMEDS)
    if m is None:
        raise SwapError("could not align face (degenerate landmarks)")
    return cv2.warpAffine(frame, m, (size, size), borderMode=cv2.BORDER_REPLICATE), m


def embed_face(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """512-d identity vector, L2-normalised (what hyperswap expects)."""
    crop, _ = _warp(frame, kps, ARCFACE_112_V2, EMBED_SIZE)
    blob = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    blob = (blob / 127.5) - 1.0
    vec = _session("arcface_w600k_r50").run(None, {"input": blob})[0].ravel()
    n = np.linalg.norm(vec)
    if n == 0:
        raise SwapError("face embedding was empty")
    return (vec / n).reshape(1, -1).astype(np.float32)


def swap_face(frame: np.ndarray, face: dict, embedding: np.ndarray) -> np.ndarray:
    """Paste the identity in ``embedding`` onto one face of ``frame``."""
    crop, matrix = _warp(frame, face["kps"], ARCFACE_128, SWAP_SIZE)

    blob = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    blob = (blob - 0.5) / 0.5                     # model's mean/std
    out, mask = _session("hyperswap_1a_256").run(
        None, {"source": embedding, "target": blob})

    swapped = np.clip((out[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0, 0, 255)
    swapped = swapped[:, :, ::-1].astype(np.float32)          # back to BGR

    # The model's own mask marks which pixels are face. Eroding then blurring it
    # keeps the seam off the hairline and blends it instead of cutting it.
    m = np.clip(mask[0][0], 0, 1).astype(np.float32)
    m = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=2)
    m = cv2.GaussianBlur(m, (11, 11), 0)

    inv = cv2.invertAffineTransform(matrix)
    h, w = frame.shape[:2]
    back = cv2.warpAffine(swapped, inv, (w, h), borderMode=cv2.BORDER_TRANSPARENT)
    back_m = cv2.warpAffine(m, inv, (w, h))[:, :, None]

    return (back * back_m + frame.astype(np.float32) * (1 - back_m)).astype(np.uint8)


def enhance_face(frame: np.ndarray, kps: np.ndarray, blend: float = 0.8) -> np.ndarray:
    """Restore detail in one face with GFPGAN.

    The swapper emits 256px; on a close-up that is an upscale by the time it is
    pasted back, which reads as soft/plastic. GFPGAN re-synthesises the face at
    512px. Blended rather than pasted outright so it sharpens without erasing
    the original skin texture.
    """
    crop, matrix = _warp(frame, kps, FFHQ_512, ENHANCE_SIZE)

    blob = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    blob = (blob - 0.5) / 0.5
    out = _session("gfpgan_1.4").run(None, {"input": blob})[0]

    restored = np.clip(out[0].transpose(1, 2, 0), -1, 1)
    restored = ((restored + 1) / 2 * 255.0)[:, :, ::-1].astype(np.float32)

    # Soft-edged oval so the restored patch fades into the frame rather than
    # leaving a visible square.
    mask = np.zeros((ENHANCE_SIZE, ENHANCE_SIZE), np.float32)
    cv2.ellipse(mask, (ENHANCE_SIZE // 2, ENHANCE_SIZE // 2),
                (int(ENHANCE_SIZE * 0.42), int(ENHANCE_SIZE * 0.52)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0) * blend

    inv = cv2.invertAffineTransform(matrix)
    h, w = frame.shape[:2]
    back = cv2.warpAffine(restored, inv, (w, h), borderMode=cv2.BORDER_TRANSPARENT)
    back_m = cv2.warpAffine(mask, inv, (w, h))[:, :, None]
    return (back * back_m + frame.astype(np.float32) * (1 - back_m)).astype(np.uint8)


# ---------------------------------------------------------------- video io
def probe(path: str) -> dict[str, Any]:
    """Video geometry via ffprobe. Raises on unreadable/undecodable media."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise SwapError(f"ffprobe failed: {e}") from e
    if r.returncode != 0:
        raise SwapError(f"unreadable media: {r.stderr.strip()[:300]}")

    info = json.loads(r.stdout or "{}")
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise SwapError("no video stream in target file")

    num, _, den = (video.get("r_frame_rate") or "25/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 25.0
    duration = float(info.get("format", {}).get("duration") or 0) or None

    total = int(video.get("nb_frames") or 0)
    if total <= 0:                  # many containers omit it; derive instead
        total = int(round(fps * duration)) if duration else 0

    return {
        "width": int(video["width"]), "height": int(video["height"]),
        "fps": round(fps, 4) or 25.0, "duration": duration, "total_frames": total,
        "has_audio": any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
        "codec": video.get("codec_name"),
    }


def _encoder() -> list[str]:
    """Prefer NVENC, fall back to x264 if this build/GPU lacks it."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=30)
        if "h264_nvenc" in r.stdout:
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    except (OSError, subprocess.SubprocessError):
        pass
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]


def _area(face: dict) -> float:
    b = face["box"]
    return float((b[2] - b[0]) * (b[3] - b[1]))


def _centre(face: dict) -> np.ndarray:
    b = face["box"]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=np.float32)


def _pick_tracked(
    faces: list[dict], frame: np.ndarray,
    ref_embedding: Optional[np.ndarray], ref_centre: Optional[np.ndarray],
    diag: float,
) -> Optional[dict]:
    """Choose the same person as last frame, not merely the biggest face.

    Picking the largest face per frame silently switches identity whenever
    someone else steps nearer the camera. We instead score every candidate on
    identity similarity (cosine distance against the target face we locked on
    to) and proximity to where that face was last seen, which is cheap and
    survives the face briefly shrinking or being partly occluded.
    """
    if not faces:
        return None
    if ref_embedding is None:
        return max(faces, key=_area)

    best, best_score = None, -1e9
    for f in faces:
        try:
            emb = embed_face(frame, f["kps"])
        except SwapError:
            continue
        identity = float(np.dot(ref_embedding.ravel(), emb.ravel()))   # both L2-normed
        proximity = 0.0
        if ref_centre is not None and diag > 0:
            proximity = 1.0 - min(1.0, float(np.linalg.norm(_centre(f) - ref_centre)) / diag)
        # Identity dominates; position only breaks ties between similar faces.
        score = identity + 0.35 * proximity
        if score > best_score:
            best, best_score = f, score

    # Everyone on screen is a poor match -- the tracked person has left frame.
    # Swapping the best of a bad set would put the face on a stranger.
    if best is None or best_score < 0.15:
        return None
    return best


def run_swap(
    source_image: str,
    target_video: str,
    output_path: str,
    on_progress: Optional[Callable[[str, float, dict], None]] = None,
    swap_all_faces: bool = False,
    quality: str = "balanced",
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Swap the face in ``source_image`` onto faces in ``target_video``.

    ``on_progress(stage, percent, extra)`` is called with real counts:
    percent during 'processing' is frames_done / total_frames * 100.
    """
    def emit(stage: str, pct: float, **extra: Any) -> None:
        if on_progress:
            on_progress(stage, max(0.0, min(100.0, pct)), extra)

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
    detect_threshold = preset["detect_threshold"]
    do_enhance = preset["enhance"]

    emit("preparing", 0.0)
    meta = probe(target_video)

    src = cv2.imread(source_image)
    if src is None:
        raise SwapError("could not read the source photo (unsupported or corrupt image)")
    src_faces = detect_faces(src)
    if not src_faces:
        raise SwapError("no face found in the source photo -- use a clear, front-facing photo")
    src_faces.sort(key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]),
                   reverse=True)
    embedding = embed_face(src, src_faces[0]["kps"])

    emit("detecting", 0.0, gpu=gpu_report())

    w, h, fps = meta["width"], meta["height"], meta["fps"]
    total = meta["total_frames"]
    frame_bytes = w * h * 3

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(target_video),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    tmp_video = str(Path(output_path).with_suffix(".silent.mp4"))
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         *_encoder(), "-pix_fmt", "yuv420p", tmp_video],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    done = 0
    swapped_count = 0
    cancelled = False
    # Identity lock for 'one person' mode: set from the first frame that has a
    # face, then carried forward so the same person is swapped throughout.
    ref_embedding: Optional[np.ndarray] = None
    ref_centre: Optional[np.ndarray] = None
    diag = float(np.hypot(w, h))
    try:
        while True:
            if should_cancel and should_cancel():
                cancelled = True
                break
            raw = dec.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)

            faces = detect_faces(frame, detect_threshold)
            if faces:
                if swap_all_faces:
                    chosen = faces
                else:
                    pick = _pick_tracked(faces, frame, ref_embedding, ref_centre, diag)
                    chosen = [pick] if pick else []
                    if pick is not None:
                        # Re-lock each frame so the reference drifts with pose
                        # and lighting instead of going stale on frame 1.
                        try:
                            ref_embedding = embed_face(frame, pick["kps"])
                        except SwapError:
                            pass
                        ref_centre = _centre(pick)

                out = frame
                for f in chosen:
                    try:
                        out = swap_face(out, f, embedding)
                        if do_enhance:
                            out = enhance_face(out, f["kps"])
                        swapped_count += 1
                    except SwapError:
                        pass          # one bad frame must not kill the render
                frame = out

            enc.stdin.write(frame.tobytes())
            done += 1
            if total:
                emit("processing", done / total * 100.0, frames=done, total=total)
            elif done % 30 == 0:
                emit("processing", 0.0, frames=done, total=0)
    finally:
        try:
            enc.stdin.close()
        except OSError:
            pass
        enc.wait(timeout=300)
        try:
            dec.kill()
        except OSError:
            pass

    if cancelled:
        Path(tmp_video).unlink(missing_ok=True)
        raise SwapError("cancelled")

    if done == 0:
        raise SwapError("no frames could be decoded from the target video "
                        f"(codec {meta.get('codec')})")
    if not Path(tmp_video).exists() or Path(tmp_video).stat().st_size == 0:
        raise SwapError(f"encoding produced no output: {enc.stderr.read().decode()[:300]}")

    emit("encoding", 99.0, frames=done, total=total or done)

    # Mux the original audio back in. Video is copied, so this is a container
    # operation -- no quality loss and no second encode.
    if meta["has_audio"]:
        emit("restoring audio", 99.5)
        mux = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp_video, "-i", str(target_video),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", "-movflags", "+faststart", str(output_path)],
            capture_output=True, timeout=1800)
        if mux.returncode != 0 or not Path(output_path).exists():
            # Audio is a bonus; a mux failure must not lose the render.
            Path(tmp_video).replace(output_path)
        else:
            Path(tmp_video).unlink(missing_ok=True)
    else:
        # +faststart moves the index to the front so the browser can seek
        # before the whole file has downloaded.
        fast = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp_video, "-c", "copy",
             "-movflags", "+faststart", str(output_path)],
            capture_output=True, timeout=600)
        if fast.returncode != 0 or not Path(output_path).exists():
            Path(tmp_video).replace(output_path)
        else:
            Path(tmp_video).unlink(missing_ok=True)

    emit("complete", 100.0, frames=done, total=total or done)
    return {"frames": done, "faces_swapped": swapped_count,
            "had_audio": meta["has_audio"], "fps": fps,
            "width": w, "height": h, "quality": quality,
            "enhanced": do_enhance, "detect_threshold": detect_threshold}
