"""V1 vs V2 head-to-head on identical inputs.

V2 does not get to replace V1 for being newer. It has to win on the metrics
that matter, measured the same way for both, on the same clips, with the same
source photo. This script renders each clip through both engines and prints the
comparison plus a contact sheet for visual inspection.

Metrics are computed on the OUTPUT files, independently of either renderer, so
neither engine can mark its own homework.
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
OUT = ROOT / "data" / "benchmarks" / "v1_vs_v2"

# Clips chosen to exercise the things V2 claims to have fixed.
DEFAULT_CLIPS = ["A_single_frontal", "K_other_larger", "J_two_crossing",
                 "F_occlusion", "I_closeup", "N_dark"]


def render_v1(source: str, target: str, out: str) -> dict[str, Any]:
    """Render with the V1 engine, then free its models.

    V1 keeps its OWN session cache, separate from app.render.sessions. Left
    resident it holds several GB of VRAM for the rest of the process, which
    silently starves whatever runs next -- in this harness, V2.1's nine-swapper
    sweep, turning a ~50 second benchmark into tens of minutes.
    """
    import gc

    from app import swapper
    t0 = time.time()
    res = swapper.run_swap(source_image=source, target_video=target,
                           output_path=out, quality="balanced")
    res["wall_s"] = round(time.time() - t0, 1)
    try:
        swapper._sessions.clear()
    except AttributeError:
        pass
    gc.collect()
    return res


def render_v2(source: str, target: str, out: str) -> dict[str, Any]:
    from app.render import pipeline
    from app.render.types import PipelineConfig
    opts = pipeline.RenderOptions(quality="quality")
    opts.config = PipelineConfig(swapper="hyperswap_1a_256", enhancer="gpen_bfr_512",
                                 enhancer_blend=0.7, mask="full")
    t0 = time.time()
    res = pipeline.render([source], target, out, opts)
    d = res.as_dict()
    d["wall_s"] = round(time.time() - t0, 1)
    return d


def measure(out_path: str, source_embedding: np.ndarray,
            original: str) -> dict[str, Any]:
    """Score a finished video. Identical code for both engines."""
    from app.render import detection, metrics, recognition, video as V

    info = V.probe(out_path)
    src_info = V.probe(original)
    # 20 sampled frames, not 48. Scoring runs two detections, an embedding per
    # detected face and an optical-flow pass per frame, for every engine --
    # several hundred model calls per clip, in the same process that is also
    # driving the renders. Halving the sample barely moves the means (these
    # are per-frame metrics averaged over a clip) and removes the dominant
    # cost of the comparison harness.
    n = min(info.total_frames, 20)
    idxs = sorted(set(int(round(i)) for i in np.linspace(0, max(info.total_frames - 1, 0), n)))

    rendered = V.sample_frames(out_path, info, idxs)
    originals = V.sample_frames(original, src_info, idxs)

    ident, stab, sharp, expr, flicker = [], [], [], [], []
    prev_emb: Optional[np.ndarray] = None
    prev_frame: Optional[np.ndarray] = None
    switches = 0

    for i in sorted(rendered):
        frame = rendered[i]
        faces = detection.detect_robust(frame, 0.4, "quality")
        if not faces:
            continue

        # Measure the SWAPPED face, not the biggest one. On a multi-person clip
        # the largest face is often the distractor that was deliberately left
        # alone -- scoring it reports the renderer's correct restraint as an
        # identity failure, and makes the swap appear to jump between people.
        best = None
        for f in faces:
            try:
                e = recognition.embed(frame, f.kps)
            except Exception:  # noqa: BLE001
                continue
            sim = recognition.similarity(e, source_embedding)
            if best is None or sim > best[0]:
                best = (sim, e, f)
        if best is None:
            continue
        sim, emb, face = best

        # A face that resembles nobody we swapped in is an untouched bystander.
        if sim < 0.15:
            continue

        ident.append(sim)
        if prev_emb is not None:
            s = recognition.similarity(prev_emb, emb)
            stab.append(s)
            if s < 0.45:
                switches += 1
        sharp.append(metrics.sharpness(frame, face.box))

        orig = originals.get(i)
        if orig is not None:
            of = detection.detect_robust(orig, 0.4, "quality")
            if of:
                expr.append(metrics.expression_delta(
                    max(of, key=lambda f: f.area).kps, face.kps))
        if prev_frame is not None:
            flicker.append(metrics.flow_warped_difference(prev_frame, frame, face.box))

        prev_emb, prev_frame = emb, frame

    def agg(xs: list[float]) -> Optional[float]:
        return round(float(np.mean(xs)), 4) if xs else None

    return {
        "identity_mean": agg(ident),
        "identity_min": round(float(np.min(ident)), 4) if ident else None,
        "identity_stability_mean": agg(stab),
        "identity_switches": switches,
        "sharpness_mean": agg(sharp),
        "expression_delta_mean": agg(expr),
        "flow_flicker_mean": agg(flicker),
        "duration": info.duration,
        "has_audio": info.has_audio,
        "frames_scored": len(ident),
    }


def contact_sheet(rows: list[dict], path: Path) -> None:
    """Source | original frame | V1 output | V2 output, one row per clip."""
    from app.render import detection, video as V
    cell = 220
    tiles: list[list[np.ndarray]] = []

    src_img = cv2.imread(str(CLIPS / "_face_a.png"))

    def face_crop(video_path: str, at: float = 0.5) -> np.ndarray:
        blank = np.zeros((cell, cell, 3), np.uint8)
        try:
            info = V.probe(video_path)
            idx = int(info.total_frames * at)
            got = V.sample_frames(video_path, info, [idx])
            if idx not in got:
                return blank
            frame = got[idx]
            faces = detection.detect_robust(frame, 0.4, "quality")
            if not faces:
                return cv2.resize(frame, (cell, cell))
            f = max(faces, key=lambda x: x.area)
            cx, cy = f.centre
            s = int(f.size * 2.2)
            x1, y1 = max(0, int(cx - s // 2)), max(0, int(cy - s // 2))
            x2 = min(frame.shape[1], x1 + s)
            y2 = min(frame.shape[0], y1 + s)
            return cv2.resize(frame[y1:y2, x1:x2], (cell, cell))
        except Exception:  # noqa: BLE001
            return blank

    for r in rows:
        tiles.append([
            cv2.resize(src_img, (cell, cell)),
            face_crop(str(CLIPS / f"{r['clip']}.mp4")),
            face_crop(r["v1_output"]),
            face_crop(r["v2_output"]),
        ])

    if not tiles:
        return
    header = 26
    sheet = np.full((header + cell * len(tiles), cell * 4, 3), 22, np.uint8)
    for x, label in enumerate(["SOURCE", "TARGET", "V1", "V2"]):
        cv2.putText(sheet, label, (x * cell + 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    for y, row in enumerate(tiles):
        for x, img in enumerate(row):
            sheet[header + y * cell:header + (y + 1) * cell, x * cell:(x + 1) * cell] = img
        cv2.putText(sheet, rows[y]["clip"][:22], (6, header + y * cell + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 220, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def main(argv: list[str]) -> int:
    from app.render import detection, recognition

    clips = argv[1:] or DEFAULT_CLIPS
    OUT.mkdir(parents=True, exist_ok=True)

    source = str(CLIPS / "_face_a.png")
    src_img = cv2.imread(source)
    sf = detection.detect_robust(src_img, 0.3)
    if not sf:
        print("no face in the source image")
        return 2
    src_emb = recognition.embed(src_img, max(sf, key=lambda f: f.area).kps)

    rows = []
    for name in clips:
        clip = CLIPS / f"{name}.mp4"
        if not clip.exists():
            print(f"  skip {name}: no clip")
            continue
        print(f"\n=== {name} ===", flush=True)

        v1_out = str(OUT / f"{name}__v1.mp4")
        v2_out = str(OUT / f"{name}__v2.mp4")
        row: dict[str, Any] = {"clip": name, "v1_output": v1_out, "v2_output": v2_out}

        try:
            r1 = render_v1(source, str(clip), v1_out)
            row["v1_wall_s"] = r1["wall_s"]
            row["v1"] = measure(v1_out, src_emb, str(clip))
            print(f"  v1 done in {r1['wall_s']}s")
        except Exception as e:  # noqa: BLE001
            row["v1_error"] = f"{type(e).__name__}: {e}"[:200]
            print(f"  v1 FAILED: {row['v1_error']}")

        try:
            r2 = render_v2(source, str(clip), v2_out)
            row["v2_wall_s"] = r2["wall_s"]
            row["v2_tracking"] = r2.get("tracking")
            row["v2"] = measure(v2_out, src_emb, str(clip))
            print(f"  v2 done in {r2['wall_s']}s")
        except Exception as e:  # noqa: BLE001
            row["v2_error"] = f"{type(e).__name__}: {e}"[:200]
            print(f"  v2 FAILED: {row['v2_error']}")

        rows.append(row)

    # ---- report
    hdr = (f"\n{'clip':<20} {'metric':<24} {'V1':>10} {'V2':>10}  winner")
    print(hdr)
    print("-" * len(hdr))
    keys = [("identity_mean", "identity (higher)", 1),
            ("identity_min", "identity worst-case", 1),
            ("identity_stability_mean", "temporal stability", 1),
            ("identity_switches", "identity switches", -1),
            ("flow_flicker_mean", "flicker (lower)", -1),
            ("expression_delta_mean", "expression drift", -1),
            ("sharpness_mean", "sharpness (higher)", 1)]
    tally = {"V1": 0, "V2": 0, "tie": 0}
    for r in rows:
        a, b = r.get("v1"), r.get("v2")
        if not a or not b:
            continue
        for key, label, direction in keys:
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                continue
            if abs(va - vb) < 1e-9:
                win = "tie"
            elif (vb > va) == (direction > 0):
                win = "V2"
            else:
                win = "V1"
            tally[win] += 1
            print(f"{r['clip']:<20} {label:<24} {va:>10.4f} {vb:>10.4f}  {win}")
    print("-" * len(hdr))
    print(f"  metric wins -> V2: {tally['V2']}   V1: {tally['V1']}   tie: {tally['tie']}")

    (OUT / "comparison.json").write_text(
        json.dumps({"rows": rows, "tally": tally}, indent=2, default=str), encoding="utf-8")
    contact_sheet(rows, OUT / "contact_sheet.jpg")
    print(f"\n  wrote {OUT/'comparison.json'}")
    print(f"  wrote {OUT/'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
