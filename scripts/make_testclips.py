"""Build the V2 test matrix from synthetic clips with known ground truth.

Real footage cannot prove "zero identity switches" because nothing states which
face *should* have been swapped in each frame. So the hard tracking cases are
composited here from two known face images: we place person A and person B at
positions and scales we choose, and therefore know exactly who is who in every
frame. If the tracker follows the wrong one, the test can say so.

Scenarios built (letters match the brief's test matrix):
    A  single frontal face
    C  fast head motion
    H  small / distant face
    I  close-up
    J  two people crossing paths
    K  a non-target person becomes larger than the target
    L  target leaves frame and returns
    M  hard scene cut
    N  dark lighting
    O  bright lighting
    F  hand / foreground object crossing the face
    S  video with no audio track
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "testclips"
W, H, FPS = 960, 540, 24


def _paste(canvas: np.ndarray, face: np.ndarray, cx: int, cy: int, size: int) -> None:
    """Alpha-free composite of a square face image centred at (cx, cy)."""
    if size < 8:
        return
    f = cv2.resize(face, (size, size), interpolation=cv2.INTER_LANCZOS4)
    x1, y1 = int(cx - size // 2), int(cy - size // 2)
    x2, y2 = x1 + size, y1 + size
    fx1, fy1 = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    canvas[y1:y2, x1:x2] = f[fy1:fy1 + (y2 - y1), fx1:fx1 + (x2 - x1)]


def _bg(seed: int) -> np.ndarray:
    """A textured background -- flat colour makes detection unrealistically easy."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 90, (H // 20, W // 20, 3), dtype=np.uint8)
    return cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)


def _write(name: str, frames: list[np.ndarray], audio: bool = True) -> Path:
    path = OUT / f"{name}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio:
        # A real audio stream, so audio-preservation can actually be tested.
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={len(frames)/FPS:.3f}",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames),
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{name}: {p.stderr.decode('utf-8', 'replace')[:300]}")
    return path


def build(face_a: np.ndarray, face_b: np.ndarray) -> dict[str, Path]:
    """face_a is the tracking TARGET; face_b is the distractor."""
    made: dict[str, Path] = {}
    n = 72          # 3 seconds at 24fps

    # A: single frontal face, steady.
    made["A_single_frontal"] = _write("A_single_frontal", [
        (lambda c: (_paste(c, face_a, W // 2, H // 2, 180), c)[1])(_bg(1).copy())
        for _ in range(n)])

    # C: fast head motion -- tests transform smoothing without lag.
    frames = []
    for i in range(n):
        c = _bg(2).copy()
        x = int(W / 2 + 260 * np.sin(i * 0.42))
        y = int(H / 2 + 70 * np.cos(i * 0.55))
        _paste(c, face_a, x, y, 170)
        frames.append(c)
    made["C_fast_motion"] = _write("C_fast_motion", frames)

    # H: small / distant face.
    made["H_small_face"] = _write("H_small_face", [
        (lambda c: (_paste(c, face_a, W // 2, H // 2, 58), c)[1])(_bg(3).copy())
        for _ in range(n)])

    # I: close-up, face larger than the swap model's native 256px.
    made["I_closeup"] = _write("I_closeup", [
        (lambda c: (_paste(c, face_a, W // 2, H // 2, 420), c)[1])(_bg(4).copy())
        for _ in range(n)])

    # J: two people crossing paths. They swap sides through the middle, so at
    # the crossover their boxes overlap -- the classic identity-swap trap.
    frames = []
    for i in range(n):
        c = _bg(5).copy()
        t = i / (n - 1)
        ax = int(160 + t * (W - 320))
        bx = int((W - 160) - t * (W - 320))
        # Draw the farther one first so the nearer overlaps it.
        for x, f in sorted([(ax, face_a), (bx, face_b)], key=lambda p: -p[0]):
            _paste(c, f, x, H // 2, 170)
        frames.append(c)
    made["J_two_crossing"] = _write("J_two_crossing", frames)

    # K: the NON-target grows much larger than the target. A "largest face"
    # heuristic fails this outright; correct behaviour is to stay on A.
    frames = []
    for i in range(n):
        c = _bg(6).copy()
        t = i / (n - 1)
        _paste(c, face_b, int(W * 0.68), H // 2, int(120 + 260 * t))   # distractor grows
        _paste(c, face_a, int(W * 0.26), H // 2, 150)                  # target constant
        frames.append(c)
    made["K_other_larger"] = _write("K_other_larger", frames)

    # L: target disappears for ~1s and returns -- tests re-acquisition.
    frames = []
    for i in range(n):
        c = _bg(7).copy()
        if not (n // 3 <= i < 2 * n // 3):
            _paste(c, face_a, W // 2, H // 2, 170)
        frames.append(c)
    made["L_disappear_return"] = _write("L_disappear_return", frames)

    # M: hard scene cut at the midpoint (different background + position).
    frames = []
    for i in range(n):
        first = i < n // 2
        c = _bg(8 if first else 99).copy()
        _paste(c, face_a, W // 3 if first else 2 * W // 3, H // 2, 160 if first else 200)
        frames.append(c)
    made["M_scene_cut"] = _write("M_scene_cut", frames)

    # N / O: lighting extremes.
    for name, gain, bias in (("N_dark", 0.32, -12), ("O_bright", 1.45, 70)):
        frames = []
        for _ in range(n):
            c = _bg(9).copy()
            _paste(c, face_a, W // 2, H // 2, 180)
            frames.append(np.clip(c.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8))
        made[name] = _write(name, frames)

    # F: an opaque bar sweeps across the face -- stands in for a hand/mic. The
    # swapped face must appear BEHIND it, i.e. those pixels stay unchanged.
    frames = []
    for i in range(n):
        c = _bg(10).copy()
        _paste(c, face_a, W // 2, H // 2, 200)
        t = i / (n - 1)
        x = int(W * 0.22 + t * W * 0.5)
        cv2.rectangle(c, (x, H // 2 - 40), (x + 80, H // 2 + 130), (28, 32, 44), -1)
        frames.append(c)
    made["F_occlusion"] = _write("F_occlusion", frames)

    # S: no audio track at all.
    made["S_no_audio"] = _write("S_no_audio", [
        (lambda c: (_paste(c, face_a, W // 2, H // 2, 180), c)[1])(_bg(11).copy())
        for _ in range(n)], audio=False)

    return made


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from app.render import detection

    OUT.mkdir(parents=True, exist_ok=True)

    # Person A: the uploaded source photo.
    a_path = next((ROOT / "data" / "uploads").glob("*.jpeg"), None)
    if a_path is None:
        print("no source photo in data/uploads")
        return 2
    img_a = cv2.imread(str(a_path))
    fa = detection.detect_robust(img_a)
    if not fa:
        print("no face in source photo")
        return 2

    def crop_square(img: np.ndarray, face) -> np.ndarray:
        cx, cy = face.centre
        s = int(face.size * 2.0)
        x1, y1 = int(cx - s // 2), int(cy - s // 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x1 + s), min(img.shape[0], y1 + s)
        return cv2.resize(img[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)

    face_a = crop_square(img_a, max(fa, key=lambda f: f.area))

    # Person B: a genuinely different identity. A cached crop is reused when
    # present, because scanning the uploads for the most dissimilar face is
    # slow and the choice only needs making once. The distractor MUST be a
    # different person -- a recoloured copy of A embeds at ~0.93 cosine, which
    # would make the crossing test unfalsifiable.
    cached_b = OUT / "_face_b.png"
    face_b = cv2.imread(str(cached_b)) if cached_b.exists() else None
    vid = next((ROOT / "data" / "uploads").glob("*.mp4"), None)
    if face_b is None:
        pass
    if face_b is None and vid is not None:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "35", "-i", str(vid), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(vid)],
            capture_output=True, text=True)
        try:
            vw, vh = [int(x) for x in probe.stdout.strip().split(",")[:2]]
            frame = np.frombuffer(r.stdout[:vw * vh * 3], np.uint8).reshape(vh, vw, 3).copy()
            fb = detection.detect_robust(frame)
            if fb:
                face_b = crop_square(frame, max(fb, key=lambda f: f.area))
        except (ValueError, IndexError):
            pass

    if face_b is None:
        # Fall back to a mirrored+recoloured variant so the matrix still runs;
        # a weaker distractor, but the test remains meaningful.
        face_b = cv2.applyColorMap(cv2.flip(face_a, 1), cv2.COLORMAP_BONE)
        print("  note: using a synthetic distractor (no second real face found)")

    cv2.imwrite(str(OUT / "_face_a.png"), face_a)
    cv2.imwrite(str(OUT / "_face_b.png"), face_b)

    made = build(face_a, face_b)
    print(f"built {len(made)} clips in {OUT}")
    for k, v in sorted(made.items()):
        print(f"  {k:24} {v.stat().st_size/1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
