"""Component benchmarks: pixel boost, detectors, and TensorRT vs CUDA.

Three questions the V2 report left open, each answered by measurement rather
than assertion:

  pixel-boost  Does tiling the aligned crop at 512/768/1024 actually recover
               detail, or is it just (boost/256)^2 more compute for nothing?
  detectors    YOLOFace vs SCRFD vs RetinaFace on genuinely hard frames.
  tensorrt     Is the TensorRT provider numerically equivalent to CUDA, and
               is it worth the build cost?

Run:  python scripts/bench_components.py [pixel|detector|trt|all]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIPS = ROOT / "data" / "testclips"
OUT = ROOT / "data" / "benchmarks" / "components"


def _vram() -> int:
    from app.render import sessions
    return int(sessions.gpu_info().get("vram_used_mb") or 0)


# ---------------------------------------------------------------- pixel boost
def bench_pixel_boost(clips: Optional[list[str]] = None) -> dict[str, Any]:
    """Measure whether pixel boost buys real detail.

    Tested on clips with LARGE faces, because that is the only case where it
    could help: if the aligned face is already smaller than 256px, the extra
    tiles are interpolating pixels that were never captured.
    """
    from app.render import (alignment, detection, masking, metrics,
                            recognition, sessions, swapping)
    from app.render.types import RenderError
    import app.render.video as V

    clips = clips or ["I_closeup", "P_1080p", "Q_4k"]
    src = cv2.imread(str(CLIPS / "_face_a.png"))
    se = recognition.embed(src, detection.detect_robust(src, 0.3)[0].kps)
    emb = swapping.prepare_embedding("hyperswap_1a_256", se)

    rows: list[dict[str, Any]] = []
    for clip in clips:
        path = CLIPS / f"{clip}.mp4"
        if not path.exists():
            continue
        info = V.probe(str(path))
        idx = info.total_frames // 2
        got = V.sample_frames(str(path), info, [idx])
        if idx not in got:
            continue
        frame = got[idx]
        faces = detection.detect_robust(frame, 0.4, "quality")
        if not faces:
            continue
        face = max(faces, key=lambda f: f.area)

        for boost in (0, 512, 768, 1024):
            sessions.release()          # measure each level from a clean slate
            try:
                t = time.time()
                patch, mm, mtx, size = swapping.swap(frame, face.kps, emb,
                                                     "hyperswap_1a_256", boost)
                ms = (time.time() - t) * 1000.0
                mask, _ = masking.build(frame, face.kps, size, model_mask=mm,
                                        face_size=face.size)
                out = alignment.paste_back(frame, patch, mask, mtx)
            except RenderError as e:
                rows.append({"clip": clip, "boost": boost or 256,
                             "error": str(e)[:160]})
                continue

            fo = detection.detect_robust(out, 0.3, "quality")
            ident = None
            if fo:
                try:
                    ident = recognition.similarity(
                        recognition.embed(out, max(fo, key=lambda f: f.area).kps), se)
                except RenderError:
                    pass
            inv = cv2.invertAffineTransform(mtx)
            full = cv2.warpAffine(mask, inv, (frame.shape[1], frame.shape[0]))

            rows.append({
                "clip": clip, "face_px": round(float(face.size), 1),
                "boost": boost or 256,
                "identity": None if ident is None else round(ident, 4),
                "sharpness": round(metrics.sharpness(out, face.box), 1),
                "texture": round(metrics.texture_retention(frame, out, face.box), 4),
                "seam": round(metrics.seam_score(out, full), 4),
                "expression": round(metrics.expression_delta(
                    face.kps, max(fo, key=lambda f: f.area).kps), 5) if fo else None,
                "ms": round(ms, 1),
                "vram_mb": _vram(),
            })

    print(f"\n{'clip':<12}{'face px':>8}{'boost':>7}{'identity':>10}"
          f"{'sharp':>9}{'texture':>9}{'seam':>7}{'ms':>8}{'vram':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['clip']:<12}{'':>8}{r['boost']:>7}  ERROR {r['error'][:44]}")
            continue
        print(f"{r['clip']:<12}{r['face_px']:>8}{r['boost']:>7}"
              f"{(r['identity'] or 0):>10.4f}{r['sharpness']:>9.1f}{r['texture']:>9.3f}"
              f"{r['seam']:>7.3f}{r['ms']:>8.0f}{r['vram_mb']:>7}")
    return {"rows": rows}


# ---------------------------------------------------------------- detectors
def bench_detectors(clips: Optional[list[str]] = None) -> dict[str, Any]:
    """Recall, landmark agreement and speed on the hard categories."""
    from app.render import detection, sessions
    from app.render.registry import DETECTORS
    from app.render.sessions import MODELS_DIR
    from app.render.registry import ALL
    import app.render.video as V

    clips = clips or ["H_small_face", "B_profile", "D_motion_blur",
                      "F_occlusion", "N_dark", "J_two_crossing", "I_closeup"]
    models = [m for m in DETECTORS if (MODELS_DIR / ALL[m].filename).exists()]

    rows: list[dict[str, Any]] = []
    for clip in clips:
        path = CLIPS / f"{clip}.mp4"
        if not path.exists():
            continue
        info = V.probe(str(path))
        idxs = [int(info.total_frames * f) for f in (0.15, 0.35, 0.55, 0.75, 0.95)]
        frames = V.sample_frames(str(path), info, idxs)
        if not frames:
            continue

        # YOLO is the reference for landmark agreement, not for truth.
        ref: dict[int, Any] = {}
        for i, fr in frames.items():
            fs = detection.detect(fr, 0.5, None, "yoloface_8n")
            if fs:
                ref[i] = max(fs, key=lambda f: f.area)

        for m in models:
            found = 0
            total_boxes = 0
            lm_errors: list[float] = []
            times: list[float] = []
            for i, fr in frames.items():
                t = time.time()
                try:
                    fs = detection.detect(fr, 0.5, None, m)
                except Exception:  # noqa: BLE001
                    fs = []
                times.append((time.time() - t) * 1000.0)
                total_boxes += len(fs)
                if fs:
                    found += 1
                    if i in ref:
                        best = min(fs, key=lambda f: float(
                            np.linalg.norm(f.centre - ref[i].centre)))
                        lm_errors.append(float(np.linalg.norm(
                            best.kps - ref[i].kps, axis=1).mean()))
            rows.append({
                "clip": clip, "detector": m,
                "frames": len(frames), "frames_with_face": found,
                "recall": round(found / max(len(frames), 1), 3),
                "boxes_total": total_boxes,
                "landmark_err_px": round(float(np.mean(lm_errors)), 2) if lm_errors else None,
                "ms": round(float(np.mean(times)), 1),
            })

    print(f"\n{'clip':<16}{'detector':<16}{'recall':>8}{'boxes':>7}"
          f"{'lm err px':>11}{'ms':>7}")
    for r in rows:
        le = "-" if r["landmark_err_px"] is None else f"{r['landmark_err_px']:.2f}"
        print(f"{r['clip']:<16}{r['detector']:<16}{r['recall']:>8.2f}"
              f"{r['boxes_total']:>7}{le:>11}{r['ms']:>7.1f}")

    print(f"\n{'detector':<16}{'mean recall':>12}{'mean ms':>9}")
    for m in models:
        sub = [r for r in rows if r["detector"] == m]
        if sub:
            print(f"{m:<16}{np.mean([r['recall'] for r in sub]):>12.3f}"
                  f"{np.mean([r['ms'] for r in sub]):>9.1f}")
    return {"rows": rows, "models": models}


# ---------------------------------------------------------------- tensorrt
def bench_tensorrt(model: str = "hyperswap_1a_256", runs: int = 6) -> dict[str, Any]:
    """CUDA vs TensorRT: identical output, and is it worth the build cost?

    TensorRT compiles an engine on first use, which can take minutes. That
    cost is reported separately from steady-state inference, because they
    matter to different things (a one-off render vs a long batch).
    """
    import onnxruntime as ort
    from app.render.registry import get_model
    from app.render.sessions import MODELS_DIR

    spec = get_model(model)
    path = MODELS_DIR / spec.filename
    avail = ort.get_available_providers()
    result: dict[str, Any] = {"model": model, "available_providers": avail}

    if "TensorrtExecutionProvider" not in avail:
        result["verdict"] = "TensorRT provider not available"
        print(result["verdict"])
        return result

    rng = np.random.default_rng(0)
    feeds_shapes = {}
    tmp = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for inp in tmp.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        feeds_shapes[inp.name] = np.asarray(
            rng.standard_normal(shape), dtype=np.float32)
    del tmp

    def run_with(providers: list, label: str) -> Optional[dict[str, Any]]:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            t = time.time()
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
            build_s = time.time() - t
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: session failed -- {type(e).__name__}: {str(e)[:120]}")
            return None
        active = sess.get_providers()

        try:
            t = time.time()
            first = sess.run(None, feeds_shapes)
            warm_s = time.time() - t
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: inference failed -- {type(e).__name__}: {str(e)[:120]}")
            return None

        times = []
        for _ in range(runs):
            t = time.time()
            sess.run(None, feeds_shapes)
            times.append((time.time() - t) * 1000.0)
        info = {"label": label, "providers": active,
                "session_build_s": round(build_s, 2),
                "first_infer_s": round(warm_s, 2),
                "steady_ms": round(float(np.median(times)), 1),
                "vram_mb": _vram()}
        print(f"  {label:<10} build {info['session_build_s']:>7.2f}s  "
              f"first {info['first_infer_s']:>7.2f}s  "
              f"steady {info['steady_ms']:>7.1f}ms  providers={active[0]}")
        return {**info, "_out": first}

    print(f"\nTensorRT vs CUDA on {model}")
    cuda = run_with(["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA")
    trt = run_with([("TensorrtExecutionProvider", {"trt_fp16_enable": False}),
                    "CUDAExecutionProvider", "CPUExecutionProvider"], "TensorRT")

    if cuda and trt:
        diffs = [float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
                 for a, b in zip(cuda["_out"], trt["_out"])]
        result["max_abs_diff"] = max(diffs) if diffs else None
        result["speedup"] = round(cuda["steady_ms"] / max(trt["steady_ms"], 1e-6), 3)
        equivalent = result["max_abs_diff"] is not None and result["max_abs_diff"] < 1e-2
        faster = result["speedup"] > 1.10
        result["equivalent"] = equivalent
        result["verdict"] = (
            "adopt: equivalent output and materially faster" if equivalent and faster
            else "keep CUDA: output equivalent but not materially faster" if equivalent
            else "keep CUDA: output differs beyond tolerance")
        print(f"  max abs output difference: {result['max_abs_diff']:.2e}")
        print(f"  steady-state speedup:      {result['speedup']:.2f}x")
        print(f"  VERDICT: {result['verdict']}")
    for d in (cuda, trt):
        if d:
            d.pop("_out", None)
    result["cuda"], result["tensorrt"] = cuda, trt
    return result


def main(argv: list[str]) -> int:
    what = (argv[1] if len(argv) > 1 else "all").lower()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    if what in ("pixel", "all"):
        print("=" * 74 + "\nPIXEL BOOST\n" + "=" * 74)
        report["pixel_boost"] = bench_pixel_boost()
    if what in ("detector", "all"):
        print("\n" + "=" * 74 + "\nDETECTORS\n" + "=" * 74)
        report["detectors"] = bench_detectors()
    if what in ("trt", "tensorrt", "all"):
        print("\n" + "=" * 74 + "\nTENSORRT\n" + "=" * 74)
        report["tensorrt"] = bench_tensorrt()

    (OUT / "components.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'components.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
