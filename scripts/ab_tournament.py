"""Phase 1/2: swapper and restoration tournaments on the A->B set ONLY.

Every previous ranking in this project was measured at least partly on the
synthetic suite, which pasted the SOURCE's own face into the clip. That is a
self-swap: the face underneath already matched, so identity scores mostly
measured "did not wreck a correct face". Rankings built on it are void.

This re-runs the competition against data/testclips/ab/, where the source (A)
is a genuinely different person from the subject (B), confirmed distinct by
both recognisers before the fixtures were emitted.

Two recognisers, kept strictly apart:

  ArcFace  SELECTOR. The swap is conditioned on it, so it is inside the loop.
  SFace    HOLDOUT. Never used to choose anything; reported alongside so a
           winner picked by ArcFace can be checked by something independent.

Run:  python scripts/ab_tournament.py swappers
      python scripts/ab_tournament.py restoration [swapper ...]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
SOURCE = AB / "_source_a.png"
OUT = ROOT / "data" / "benchmarks" / "ab_tournament"

# Category -> clip. Chosen to cover the robustness axes; single-subject only,
# because this measures identity transfer rather than tracking.
CATEGORIES: list[tuple[str, str]] = [
    ("frontal", "A_single_frontal"),
    ("profile", "B_profile"),
    ("talking", "G_talking"),
    ("closeup", "I_closeup"),
    ("dark", "N_dark"),
    ("bright", "O_bright"),
    ("motion_blur", "D_motion_blur"),
    ("glasses", "E_glasses"),
    ("occlusion", "F_occlusion"),
    ("small_face", "H_small_face"),
    ("1080p", "P_1080p"),
    ("4k", "Q_4k"),
]

FRAMES_PER_CLIP = 6


def load_samples(stem: str) -> list[tuple[int, np.ndarray]]:
    import app.render.video as V
    p = AB / (stem + ".mp4")
    if not p.is_file():
        return []
    info = V.probe(str(p))
    n = info.total_frames or 0
    if n <= 0:
        return []
    idx = [int(i * (n - 1) / max(FRAMES_PER_CLIP - 1, 1)) for i in range(FRAMES_PER_CLIP)]
    got = V.sample_frames(str(p), info, idx)
    return sorted(got.items())


def evaluate(cfg, identity, opts, src_sface) -> dict[str, Any]:
    """Score one pipeline config across every category clip."""
    from app.render import detection, judges, metrics, recognition, sessions
    from app.render.pipeline import FrameRenderer
    from app.render.types import RenderError

    per_cat: dict[str, dict[str, Any]] = {}
    arc_all: list[float] = []
    sfc_all: list[float] = []
    agg_keys = ["expression_delta", "landmark_delta", "seam",
                "color_discontinuity", "sharpness", "texture_retention",
                "mask_jitter", "flow_flicker"]
    pooled: dict[str, list[float]] = {k: [] for k in agg_keys}
    total_ms: list[float] = []
    failures = 0
    t_start = time.time()

    for cat, stem in CATEGORIES:
        samples = load_samples(stem)
        if not samples:
            continue
        diag = float(np.hypot(samples[0][1].shape[1], samples[0][1].shape[0]))
        renderer = FrameRenderer(cfg, opts, identity, diag)
        arc_c: list[float] = []
        sfc_c: list[float] = []
        prev_out = None

        for idx, frame in samples:
            faces = detection.detect_robust(frame, opts.detect_threshold, "quality")
            if not faces:
                continue
            face = max(faces, key=lambda f: f.area)
            try:
                t = time.time()
                out, full_mask = renderer.render_face(frame, face)
                total_ms.append((time.time() - t) * 1000.0)
            except RenderError:
                failures += 1
                continue

            out_faces = detection.detect_robust(out, 0.3, "quality")
            of = max(out_faces, key=lambda f: f.area) if out_faces else None
            if of is not None:
                try:
                    arc = recognition.similarity(
                        recognition.embed(out, of.kps), identity.embedding)
                    arc_c.append(arc)
                except RenderError:
                    pass
                if src_sface is not None:
                    try:
                        sfc_c.append(judges.similarity(
                            src_sface, judges.embed(out, of.kps)))
                    except RenderError:
                        pass
                pooled["expression_delta"].append(
                    metrics.expression_delta(face.kps, of.kps))
                pooled["landmark_delta"].append(
                    metrics.landmark_delta(face.kps, of.kps, face.size))
            pooled["seam"].append(metrics.seam_score(out, full_mask))
            pooled["color_discontinuity"].append(
                metrics.color_discontinuity(out, full_mask))
            pooled["sharpness"].append(metrics.sharpness(out, face.box))
            pooled["texture_retention"].append(
                metrics.texture_retention(frame, out, face.box))
            pooled["mask_jitter"].append(renderer.last_mask_jitter)
            if prev_out is not None:
                pooled["flow_flicker"].append(
                    metrics.flow_warped_difference(prev_out, out, face.box))
            prev_out = out

        arc_all += arc_c
        sfc_all += sfc_c
        per_cat[cat] = {
            "arc_mean": round(float(np.mean(arc_c)), 4) if arc_c else None,
            "arc_worst": round(float(np.min(arc_c)), 4) if arc_c else None,
            "sface_mean": round(float(np.mean(sfc_c)), 4) if sfc_c else None,
            "n": len(arc_c),
        }

    def m(key: str) -> Optional[float]:
        v = pooled[key]
        return round(float(np.mean(v)), 5) if v else None

    vram = int(sessions.gpu_info().get("vram_used_mb") or 0)
    agg = {
        "identity_mean": round(float(np.mean(arc_all)), 4) if arc_all else None,
        "identity_min": round(float(np.min(arc_all)), 4) if arc_all else None,
        "sface_mean": round(float(np.mean(sfc_all)), 4) if sfc_all else None,
        "sface_min": round(float(np.min(sfc_all)), 4) if sfc_all else None,
        "expression_delta_mean": m("expression_delta"),
        "landmark_delta_mean": m("landmark_delta"),
        "seam_mean": m("seam"),
        "color_discontinuity_mean": m("color_discontinuity"),
        "sharpness_mean": m("sharpness"),
        "texture_retention_mean": m("texture_retention"),
        "mask_jitter_mean": m("mask_jitter"),
        "flow_flicker_mean": m("flow_flicker"),
        "ms_per_frame": round(float(np.mean(total_ms)), 1) if total_ms else None,
        "identity_switches": 0,          # single-subject clips by construction
        "frames_measured": len(arc_all),
        "frames_failed": failures,
        "vram_peak_mb": vram,
        "wall_s": round(time.time() - t_start, 1),
    }
    # composite_score is the SELECTOR. It must never see an SFace term.
    selector_view = {k: v for k, v in agg.items()
                     if not k.startswith("sface")}
    score, contrib = metrics.composite_score(selector_view)
    return {"label": cfg.label or cfg.describe(), "config": asdict(cfg),
            "selector_score": score, "contributions": contrib,
            "aggregate": agg, "per_category": per_cat}


def setup():
    from app.render import detection, judges, pipeline, recognition
    from app.render.pipeline import RenderOptions
    if not SOURCE.is_file():
        raise SystemExit("no A->B fixtures; run scripts/make_ab_fixtures.py")
    identity = pipeline.load_source([str(SOURCE)])
    src = cv2.imread(str(SOURCE))
    sf = detection.detect_robust(src, 0.3)
    src_sface = (judges.embed(src, sf[0].kps)
                 if sf and judges.available() else None)
    opts = RenderOptions(quality="quality")
    opts.use_parsing = False
    opts.use_occlusion = False
    return identity, src_sface, opts


def report(rows: list[dict[str, Any]], name: str) -> None:
    rows.sort(key=lambda r: -(r["selector_score"] or 0))
    print("\n%-26s %7s %7s %7s | %7s %7s | %6s %6s %6s %6s | %6s"
          % ("pipeline", "score", "arcMean", "arcWrst", "sfMean", "sfWrst",
             "expr", "seam", "flick", "sharp", "ms/f"))
    print("-" * 118)
    for r in rows:
        a = r["aggregate"]
        print("%-26s %7s %7s %7s | %7s %7s | %6s %6s %6s %6s | %6s"
              % (r["label"][:26], round(r["selector_score"], 4),
                 a["identity_mean"], a["identity_min"],
                 a["sface_mean"], a["sface_min"],
                 a["expression_delta_mean"], a["seam_mean"],
                 a["flow_flicker_mean"], a["sharpness_mean"],
                 a["ms_per_frame"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name + ".json")).write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print("\nwritten: %s" % (OUT / (name + ".json")))


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "swappers"
    from app.render.registry import SWAPPERS
    from app.render.types import PipelineConfig
    identity, src_sface, opts = setup()

    if mode == "swappers":
        print("Phase 1: swapper tournament on A->B (no pixel boost)")
        rows = []
        for name in SWAPPERS:
            # mask="model": measured +0.083 identity over the parsing/XSeg
            # stack, so ranking swappers under "full" compares them through
            # a mask that suppresses the very thing being measured.
            cfg = PipelineConfig(swapper=name, mask="model", label=name)
            try:
                rows.append(evaluate(cfg, identity, opts, src_sface))
                r = rows[-1]["aggregate"]
                print("  %-26s arc %s/%s  sface %s/%s  %s ms/f"
                      % (name, r["identity_mean"], r["identity_min"],
                         r["sface_mean"], r["sface_min"], r["ms_per_frame"]),
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print("  %-26s FAILED: %s" % (name, str(e)[:70]), flush=True)
        report(rows, "swappers")
        return 0

    if mode == "restoration":
        winners = sys.argv[2:] or ["hyperswap_1a_256"]
        enhancers = [None, "gpen_bfr_512", "gfpgan_1.4", "codeformer",
                     "restoreformer_plus_plus"]
        blends = [0.2, 0.4, 0.6, 0.8, 1.0]
        print("Phase 2: restoration tournament on A->B")
        rows = []
        for sw in winners:
            for enh in enhancers:
                for bl in ([0.0] if enh is None else blends):
                    lab = "%s+%s@%d" % (sw.replace("_256", ""),
                                        enh or "none", int(bl * 100))
                    cfg = PipelineConfig(swapper=sw, enhancer=enh,
                                         enhancer_blend=bl, mask="full",
                                         label=lab)
                    try:
                        rows.append(evaluate(cfg, identity, opts, src_sface))
                        r = rows[-1]["aggregate"]
                        print("  %-30s arc %s/%s  sface %s  sharp %s  %s ms/f"
                              % (lab, r["identity_mean"], r["identity_min"],
                                 r["sface_mean"], r["sharpness_mean"],
                                 r["ms_per_frame"]), flush=True)
                    except Exception as e:  # noqa: BLE001
                        print("  %-30s FAILED: %s" % (lab, str(e)[:60]), flush=True)
        report(rows, "restoration")
        return 0

    raise SystemExit("unknown mode %r" % mode)


if __name__ == "__main__":
    raise SystemExit(main())
