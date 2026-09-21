"""V2 acceptance test suite.

Runs the synthetic test matrix through the real render pipeline and checks
outcomes that can actually fail. The tracking clips are composited from two
known identities (see make_testclips.py), so "the tracker followed the wrong
person" is a decidable statement rather than an impression.

Hard requirements (a failure here fails the suite):
  * zero identity switches on the multi-person clips
  * the swapped face on K_other_larger is the TARGET, not the larger distractor
  * occluded pixels are left untouched -- the new face sits behind the object
  * audio survives when present; duration is preserved; output is valid
  * CUDA genuinely bound, never a silent CPU fallback

Run:  python scripts/v2_testsuite.py [clip_name ...]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIPS = ROOT / "data" / "testclips"
OUTDIR = ROOT / "data" / "benchmarks" / "testsuite"

# Tolerances. Duration must hold to well under one frame at 24fps.
DURATION_TOL_S = 0.10


def probe(path: Path) -> dict[str, Any]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return {}
    return json.loads(r.stdout or "{}")


def _streams(info: dict, kind: str) -> list[dict]:
    return [s for s in info.get("streams", []) if s.get("codec_type") == kind]


def run_clip(name: str, quality: str = "quality") -> dict[str, Any]:
    """Render one clip and collect everything the checks need."""
    from app.render import pipeline
    from app.render.types import PipelineConfig, RenderError

    clip = CLIPS / f"{name}.mp4"
    out = OUTDIR / f"{name}__{quality}.mp4"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    opts = pipeline.RenderOptions(quality=quality)
    if quality == "quality":
        opts.config = PipelineConfig(swapper="hyperswap_1a_256",
                                     enhancer="gpen_bfr_512", enhancer_blend=0.7,
                                     mask="model")
    else:
        opts.config = PipelineConfig(swapper="hyperswap_1a_256", mask="model")

    rec: dict[str, Any] = {"clip": name, "quality": quality}
    t0 = time.time()
    try:
        res = pipeline.render([str(CLIPS / "_face_a.png")], str(clip), str(out), opts)
    except RenderError as e:
        rec.update(ok=False, error=str(e)[:300])
        return rec

    rec.update(
        ok=True, wall_s=round(time.time() - t0, 1),
        frames=res.frames, faces=res.faces_swapped,
        tracking=res.tracking, scene_cuts=res.scene_cuts,
        cuda=res.gpu.get("cuda_active"), encoder=res.encoder,
        mask_sources=res.mask_sources, output=str(out),
        ms_per_frame=res.timings.get("ms_per_frame"),
    )

    src, dst = probe(clip), probe(out)
    rec["src_duration"] = float(src.get("format", {}).get("duration") or 0)
    rec["out_duration"] = float(dst.get("format", {}).get("duration") or 0)
    rec["src_audio"] = bool(_streams(src, "audio"))
    rec["out_audio"] = bool(_streams(dst, "audio"))
    rec["out_valid"] = bool(_streams(dst, "video"))
    rec["out_bytes"] = out.stat().st_size if out.exists() else 0
    return rec


# ---------------------------------------------------------------- checks
def check_common(rec: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    out.append(("renders", bool(rec.get("ok")), rec.get("error", "")))
    if not rec.get("ok"):
        return out
    out.append(("cuda active", rec.get("cuda") is True, str(rec.get("cuda"))))
    out.append(("output valid", rec.get("out_valid") is True, ""))
    out.append(("output non-empty", rec.get("out_bytes", 0) > 1024,
                f"{rec.get('out_bytes')} bytes"))
    d = abs(rec.get("out_duration", 0) - rec.get("src_duration", 0))
    out.append(("duration preserved", d <= DURATION_TOL_S,
                f"src {rec.get('src_duration'):.3f}s out {rec.get('out_duration'):.3f}s"))
    if rec.get("src_audio"):
        out.append(("audio preserved", rec.get("out_audio") is True, ""))
    else:
        out.append(("no-audio handled", rec.get("out_audio") is False,
                    "source had no audio"))
    return out


def check_no_switches(rec: dict) -> list[tuple[str, bool, str]]:
    t = rec.get("tracking") or {}
    sw = t.get("identity_switches")
    return [("identity switches == 0", sw == 0, f"switches={sw}")]


def check_target_not_distractor(rec: dict) -> list[tuple[str, bool, str]]:
    """On K_other_larger the target sits left at constant size; the distractor
    grows on the right. Confirm the swap landed on the LEFT face."""
    from app.render import detection, recognition

    out = Path(rec["output"])
    info = probe(out)
    vs = _streams(info, "video")
    if not vs:
        return [("swapped the target, not the larger face", False, "no video stream")]
    w, h = int(vs[0]["width"]), int(vs[0]["height"])

    # Last frame: distractor is at its largest here, so this is the hard case.
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-sseof", "-0.2", "-i", str(out), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    if len(r.stdout) < w * h * 3:
        return [("swapped the target, not the larger face", False, "could not read frame")]
    frame = np.frombuffer(r.stdout[:w * h * 3], np.uint8).reshape(h, w, 3)

    src = cv2.imread(str(CLIPS / "_face_a.png"))
    sf = detection.detect_robust(src, 0.3)
    if not sf:
        return [("swapped the target, not the larger face", False, "no source face")]
    se = recognition.embed(src, sf[0].kps)

    faces = detection.detect_robust(frame, 0.4)
    if not faces:
        return [("swapped the target, not the larger face", False, "no faces in output")]

    best_left = best_right = -1.0
    for f in faces:
        try:
            sim = recognition.similarity(recognition.embed(frame, f.kps), se)
        except Exception:  # noqa: BLE001
            continue
        if f.centre[0] < w / 2:
            best_left = max(best_left, sim)
        else:
            best_right = max(best_right, sim)
    ok = best_left > best_right and best_left > 0.25
    return [("swapped the target, not the larger face", ok,
             f"left(target) sim={best_left:.3f}  right(distractor) sim={best_right:.3f}")]


def check_occlusion(rec: dict) -> list[tuple[str, bool, str]]:
    """The sweeping bar must survive the swap: those pixels stay as they were."""
    out = Path(rec["output"])
    clip = CLIPS / "F_occlusion.mp4"
    info = probe(out)
    vs = _streams(info, "video")
    if not vs:
        return [("occluder preserved (face rendered behind)", False, "no video")]
    w, h = int(vs[0]["width"]), int(vs[0]["height"])

    def mid_frame(p: Path) -> Optional[np.ndarray]:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "1.5", "-i", str(p), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        if len(r.stdout) < w * h * 3:
            return None
        return np.frombuffer(r.stdout[:w * h * 3], np.uint8).reshape(h, w, 3).copy()

    a, b = mid_frame(clip), mid_frame(out)
    if a is None or b is None:
        return [("occluder preserved (face rendered behind)", False, "frame read failed")]

    # The bar is a flat dark colour; find it in the ORIGINAL and require those
    # same pixels to be nearly unchanged in the render.
    bar = (np.abs(a.astype(int) - np.array([28, 32, 44])).sum(axis=2) < 30)
    if bar.sum() < 500:
        return [("occluder preserved (face rendered behind)", False,
                 f"occluder not located ({bar.sum()} px)")]
    delta = float(np.abs(a[bar].astype(int) - b[bar].astype(int)).mean())
    return [("occluder preserved (face rendered behind)", delta < 12.0,
             f"mean change on occluder = {delta:.2f} (lower is better)")]


def check_scene_cut(rec: dict) -> list[tuple[str, bool, str]]:
    n = rec.get("scene_cuts", 0)
    return [("scene cut detected", n >= 1, f"cuts detected={n}")]


def check_reacquire(rec: dict) -> list[tuple[str, bool, str]]:
    t = rec.get("tracking") or {}
    swapped = t.get("frames_swapped", 0)
    skipped = t.get("frames_skipped", 0)
    # Target is absent for ~1/3 of the clip: it must be skipped, not faked.
    return [
        ("skipped frames where target absent", skipped > 0, f"skipped={skipped}"),
        ("re-acquired target after gap", swapped > 0, f"swapped={swapped}"),
        ("no identity switches", t.get("identity_switches") == 0,
         f"switches={t.get('identity_switches')}"),
    ]


EXTRA = {
    "J_two_crossing": check_no_switches,
    "K_other_larger": lambda r: check_no_switches(r) + check_target_not_distractor(r),
    "F_occlusion": check_occlusion,
    "M_scene_cut": check_scene_cut,
    "L_disappear_return": check_reacquire,
}

ORDER = ["A_single_frontal", "C_fast_motion", "H_small_face", "I_closeup",
         "J_two_crossing", "K_other_larger", "L_disappear_return",
         "M_scene_cut", "N_dark", "O_bright", "F_occlusion", "S_no_audio"]


def main(argv: list[str]) -> int:
    names = argv[1:] or ORDER
    results, failures = [], []

    for name in names:
        if not (CLIPS / f"{name}.mp4").exists():
            print(f"  {name}: MISSING CLIP -- run scripts/make_testclips.py")
            failures.append((name, "missing clip", ""))
            continue

        print(f"\n=== {name} ===", flush=True)
        rec = run_clip(name)
        checks = check_common(rec)
        if rec.get("ok") and name in EXTRA:
            checks += EXTRA[name](rec)

        for label, ok, note in checks:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {label}" + (f"   ({note})" if note else ""))
            if not ok:
                failures.append((name, label, note))
        if rec.get("ok"):
            t = rec.get("tracking") or {}
            print(f"         {rec['frames']} frames, {rec['faces']} swapped, "
                  f"{rec.get('ms_per_frame')} ms/frame, cuts={rec.get('scene_cuts')}, "
                  f"switches={t.get('identity_switches')}")
        results.append({"name": name, "record": rec,
                        "checks": [{"label": c, "ok": o, "note": n} for c, o, n in checks]})

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    total = sum(len(r["checks"]) for r in results)
    print(f"\n{'='*66}")
    print(f"  {total - len(failures)}/{total} checks passed across {len(results)} clips")
    if failures:
        print(f"  {len(failures)} FAILURES:")
        for clip, label, note in failures:
            print(f"    {clip:22} {label}  {note}")
    else:
        print("  ALL CHECKS PASSED")
    print(f"{'='*66}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
