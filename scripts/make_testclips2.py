"""Second half of the acceptance matrix: the categories the synthetic set missed.

Adds B (profile), D (motion blur), E (glasses), G (talking/open mouth),
P (1080p), Q (short 4K), R (audio) -- plus a small REAL-footage regression set
derived from material already in data/uploads.

Synthetic and real serve different purposes and both are kept:
  synthetic -- we control ground truth, so identity/tracking claims are decidable
  real      -- actual camera motion, compression artefacts and lighting change,
               which composites do not reproduce

The real clips are cut from footage already present locally. They are used
only as technical regression fixtures (does it decode, track, hold identity,
keep audio), never redistributed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "testclips"
REAL = ROOT / "data" / "testclips" / "real"
FPS = 24


def _bg(seed: int, w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(40, 95, (max(h // 20, 2), max(w // 20, 2), 3),
                                   dtype=np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)


def _paste(canvas, face, cx, cy, size):
    if size < 8:
        return
    f = cv2.resize(face, (size, size), interpolation=cv2.INTER_LANCZOS4)
    x1, y1 = int(cx - size // 2), int(cy - size // 2)
    fx1, fy1 = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas.shape[1], x1 + size - fx1), min(canvas.shape[0], y1 + size - fy1)
    if x2 <= x1 or y2 <= y1:
        return
    canvas[y1:y2, x1:x2] = f[fy1:fy1 + (y2 - y1), fx1:fx1 + (x2 - x1)]


def _write(name: str, frames: list[np.ndarray], w: int, h: int,
           audio: bool = True, out_dir: Path = OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={len(frames)/FPS:.3f}",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{name}: {p.stderr.decode('utf-8','replace')[:300]}")
    return path


# ---------------------------------------------------------------- synthetic
def _profile(face: np.ndarray, amount: float) -> np.ndarray:
    """Approximate a yaw turn by horizontally compressing one side.

    Not a true 3D rotation -- it is a geometric stand-in that moves the
    landmarks the way a turn does, which is what the aligner and tracker see.
    """
    h, w = face.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    k = w * 0.34 * amount
    dst = np.float32([[k, 0], [w, 0], [k * 0.85, h], [w, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(face, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _glasses(face: np.ndarray) -> np.ndarray:
    """Draw plausible spectacles over the eye region."""
    f = face.copy()
    h, w = f.shape[:2]
    ey = int(h * 0.42)
    r = int(w * 0.14)
    for cx in (int(w * 0.33), int(w * 0.67)):
        cv2.circle(f, (cx, ey), r, (28, 28, 32), 3)
        overlay = f.copy()
        cv2.circle(overlay, (cx, ey), r - 2, (120, 140, 160), -1)
        f = cv2.addWeighted(overlay, 0.18, f, 0.82, 0)
    cv2.line(f, (int(w * 0.33) + r, ey), (int(w * 0.67) - r, ey), (28, 28, 32), 3)
    cv2.line(f, (int(w * 0.33) - r, ey), (0, int(h * 0.36)), (28, 28, 32), 3)
    cv2.line(f, (int(w * 0.67) + r, ey), (w, int(h * 0.36)), (28, 28, 32), 3)
    return f


def _open_mouth(face: np.ndarray, amount: float) -> np.ndarray:
    """Stretch the lower face downward to mimic a jaw opening."""
    h, w = face.shape[:2]
    out = face.copy()
    top = int(h * 0.62)
    lower = face[top:, :]
    nh = int(lower.shape[0] * (1.0 + 0.22 * amount))
    lower = cv2.resize(lower, (w, nh), interpolation=cv2.INTER_LINEAR)
    out[top:, :] = lower[:h - top, :]
    return out


def build_synth(face: np.ndarray) -> dict[str, Path]:
    made: dict[str, Path] = {}
    n = 72
    W, H = 960, 540

    # B: side / profile turn, sweeping through the range.
    frames = []
    for i in range(n):
        c = _bg(21, W, H).copy()
        amt = abs(np.sin(i * 0.045)) * 0.95
        _paste(c, _profile(face, amt), W // 2, H // 2, 190)
        frames.append(c)
    made["B_profile"] = _write("B_profile", frames, W, H)

    # D: motion blur -- directional blur that grows and shrinks.
    frames = []
    for i in range(n):
        c = _bg(22, W, H).copy()
        _paste(c, face, int(W / 2 + 200 * np.sin(i * 0.3)), H // 2, 180)
        k = 1 + 2 * int(abs(np.sin(i * 0.3)) * 7)
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        frames.append(cv2.filter2D(c, -1, kern))
    made["D_motion_blur"] = _write("D_motion_blur", frames, W, H)

    # E: glasses.
    g = _glasses(face)
    made["E_glasses"] = _write("E_glasses", [
        (lambda c: (_paste(c, g, W // 2, H // 2, 200), c)[1])(_bg(23, W, H).copy())
        for _ in range(n)], W, H)

    # G: talking -- the jaw opens and closes.
    frames = []
    for i in range(n):
        c = _bg(24, W, H).copy()
        _paste(c, _open_mouth(face, abs(np.sin(i * 0.38))), W // 2, H // 2, 200)
        frames.append(c)
    made["G_talking"] = _write("G_talking", frames, W, H)

    # P: 1080p.
    W2, H2 = 1920, 1080
    made["P_1080p"] = _write("P_1080p", [
        (lambda c: (_paste(c, face, W2 // 2, H2 // 2, 360), c)[1])(_bg(25, W2, H2).copy())
        for _ in range(48)], W2, H2)

    # Q: short 4K -- deliberately brief; this is a VRAM/throughput check.
    W3, H3 = 3840, 2160
    made["Q_4k"] = _write("Q_4k", [
        (lambda c: (_paste(c, face, W3 // 2, H3 // 2, 700), c)[1])(_bg(26, W3, H3).copy())
        for _ in range(24)], W3, H3)

    # R: an explicit with-audio case (S_no_audio is its counterpart).
    made["R_with_audio"] = _write("R_with_audio", [
        (lambda c: (_paste(c, face, W // 2, H // 2, 180), c)[1])(_bg(27, W, H).copy())
        for _ in range(n)], W, H, audio=True)
    return made


# ---------------------------------------------------------------- real
def build_real() -> dict[str, Path]:
    """Cut short regression fixtures from a CURATED source clip.

    Real footage contributes what composites cannot: sensor noise, codec
    artefacts, rolling shutter, genuine motion blur and lighting that drifts.

    The source must be placed deliberately at ``data/testclips/source.mp4``.
    This used to take ``sorted(data/uploads/*.mp4)[0]`` -- whatever the user
    happened to have uploaded, chosen by filename order. A benchmark fixture
    is something you should be able to point at and justify, and picking it
    by accident makes every number measured against it unexplainable.
    """
    REAL.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    src = OUT / "source.mp4"
    if not src.is_file():
        print("  no curated source clip at %s -- skipping real fixtures.\n"
              "  Place a single-person clip there to enable them." % src)
        return made
    # Different offsets give genuinely different motion/lighting conditions.
    cuts = [("real_A_motion", "20", "4"), ("real_B_talking", "48", "4"),
            ("real_C_lighting", "75", "4")]
    for name, start, dur in cuts:
        out = REAL / f"{name}.mp4"
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", start, "-i", str(src),
             "-t", dur, "-vf", "scale=960:-2", "-c:v", "libx264", "-crf", "20",
             "-preset", "medium", "-c:a", "aac", str(out)],
            capture_output=True, timeout=300)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 1000:
            made[name] = out
    return made


def main() -> int:
    from app.render import detection

    face_path = OUT / "_face_a.png"
    if not face_path.exists():
        print("run scripts/make_testclips.py first")
        return 2
    face = cv2.imread(str(face_path))

    made = build_synth(face)
    print(f"synthetic: {len(made)} clips")
    for k, v in sorted(made.items()):
        print(f"  {k:22} {v.stat().st_size/1e6:7.2f} MB")

    real = build_real()
    print(f"real footage: {len(real)} clips")
    for k, v in sorted(real.items()):
        print(f"  {k:22} {v.stat().st_size/1e6:7.2f} MB")

    # A clip with no findable face is a broken fixture, not a test result.
    print("sanity: face detectable in each new synthetic clip")
    import app.render.video as V
    for k, v in sorted(made.items()):
        info = V.probe(str(v))
        got = V.sample_frames(str(v), info, [info.total_frames // 2])
        fr = next(iter(got.values()), None)
        n = len(detection.detect_robust(fr, 0.4)) if fr is not None else 0
        print(f"  {k:22} {info.width}x{info.height}  faces={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
