"""Regression tests for the Auto Max scoring function.

A weighted score is only meaningful if every weighted term actually moves when
the thing it measures gets worse. One term was previously hard-coded, which
silently awarded 18% of the score to every candidate equally -- the score
looked fine and discriminated on 82% of what it claimed to.

So for each metric this asserts four properties:

  1. RESPONDS   -- deliberately degrading the input changes the metric
  2. DIRECTION  -- it moves the way the name implies
  3. NON-ZERO   -- its contribution to the composite is not always constant
  4. PENALISED  -- a worse value yields a worse composite score

Run:  python scripts/test_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.render import metrics  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, note: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({note})" if note else ""))


# ---------------------------------------------------------------- helpers
def _good() -> dict:
    """A plausibly excellent candidate."""
    return {
        "identity_mean": 0.62, "identity_min": 0.58,
        "identity_stability_mean": 0.985,
        "identity_switches": 0, "flow_flicker_mean": 3.0,
        "mask_jitter_mean": 0.02, "seam_mean": 0.65,
        "color_discontinuity_mean": 4.0, "expression_delta_mean": 0.012,
        "sharpness_mean": 380.0, "texture_retention_mean": 1.05,
        "ms_per_frame": 150.0,
    }


def degrade(base: dict, key: str, worse: float) -> dict:
    d = dict(base)
    d[key] = worse
    return d


# ---------------------------------------------------------------- metric behaviour
def test_metric_responses() -> None:
    print("\n=== metrics respond to deliberate degradation ===")
    rng = np.random.default_rng(0)
    img = (rng.integers(60, 190, (256, 256, 3))).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # sharpness: blurring must lower it
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    box = np.array([20, 20, 236, 236], np.float32)
    s_ok, s_bad = metrics.sharpness(img, box), metrics.sharpness(blurred, box)
    check("sharpness drops when blurred", s_bad < s_ok, f"{s_ok:.1f} -> {s_bad:.1f}")

    # texture retention: blurring the render must lower it
    t_ok = metrics.texture_retention(img, img, box)
    t_bad = metrics.texture_retention(img, blurred, box)
    check("texture retention drops when smoothed", t_bad < t_ok, f"{t_ok:.2f} -> {t_bad:.2f}")

    # seam: a hard-edged mask must score worse than a feathered one
    soft = np.zeros((256, 256), np.float32)
    cv2.circle(soft, (128, 128), 70, 1.0, -1)
    hard_mask = soft.copy()
    soft = cv2.GaussianBlur(soft, (31, 31), 0)
    patch = img.copy()
    patch[:] = cv2.addWeighted(img, 0.5, np.full_like(img, 200), 0.5, 0)
    hard_img = np.where(hard_mask[:, :, None] > 0.5, patch, img).astype(np.uint8)
    soft_img = (patch * soft[:, :, None] + img * (1 - soft[:, :, None])).astype(np.uint8)
    seam_hard = metrics.seam_score(hard_img, hard_mask)
    seam_soft = metrics.seam_score(soft_img, hard_mask)
    check("seam score worse for a hard edge", seam_hard > seam_soft,
          f"hard {seam_hard:.3f} vs soft {seam_soft:.3f}")

    # colour discontinuity: tinting inside the mask must raise it
    tinted = img.copy()
    tinted[hard_mask > 0.5] = np.clip(
        tinted[hard_mask > 0.5].astype(int) + np.array([70, -40, 40]), 0, 255).astype(np.uint8)
    c_ok = metrics.color_discontinuity(img, hard_mask)
    c_bad = metrics.color_discontinuity(tinted, hard_mask)
    check("colour discontinuity rises when tinted", c_bad > c_ok, f"{c_ok:.1f} -> {c_bad:.1f}")

    # flow flicker: adding noise between frames must raise it
    nxt = np.clip(img.astype(int) + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8)
    f_ok = metrics.flow_warped_difference(img, img, box)
    f_bad = metrics.flow_warped_difference(img, nxt, box)
    check("flow flicker rises with frame noise", f_bad > f_ok, f"{f_ok:.2f} -> {f_bad:.2f}")

    # expression delta: moving the mouth points must register
    kps = np.array([[80, 100], [170, 100], [125, 150], [90, 195], [160, 195]], np.float32)
    moved = kps.copy()
    moved[3:] += np.array([0, 22], np.float32)
    e_same = metrics.expression_delta(kps, kps)
    e_diff = metrics.expression_delta(kps, moved)
    check("expression delta rises when mouth moves", e_diff > e_same,
          f"{e_same:.4f} -> {e_diff:.4f}")

    # Scene cut. Structured scenes, not pure noise: two noise images have
    # similar histograms AND large pixel differences, which no real footage
    # does, so testing on noise tests nothing real.
    def scene(seed: int, shift: int = 0) -> np.ndarray:
        r = np.random.default_rng(seed)
        f = np.full((180, 320, 3), r.integers(30, 90, 3), np.uint8)
        for _ in range(12):
            x, y = int(r.integers(0, 300)), int(r.integers(0, 160))
            cv2.rectangle(f, (x, y), (x + 20, y + 18),
                          tuple(int(v) for v in r.integers(80, 240, 3)), -1)
        return np.roll(f, shift, axis=1)

    det = metrics.SceneCutDetector()
    shot_a = [scene(7, i) for i in range(0, 10, 2)]      # one shot, panning
    fired = [det(shot_a[i], shot_a[i + 1]) for i in range(len(shot_a) - 1)]
    check("scene cut: no false cut while panning within one shot", not any(fired),
          f"fired={fired}")
    check("scene cut: fires on a genuine change of scene", det(shot_a[-1], scene(99)))


# ---------------------------------------------------------------- composite
def test_composite_behaviour() -> None:
    print("\n=== composite score: every weighted term must matter ===")
    base = _good()
    base_score, base_parts = metrics.composite_score(base)
    print(f"  baseline score = {base_score:.5f}")

    worse_values = {
        "identity": ("identity_mean", 0.22),
        "identity_worst": ("identity_min", 0.26),
        "identity_switches": ("identity_switches", 4),
        "temporal": ("identity_stability_mean", 0.80),
        "blending": ("seam_mean", 1.45),
        "expression": ("expression_delta_mean", 0.075),
        "detail": ("sharpness_mean", 60.0),
        "speed": ("ms_per_frame", 1400.0),
    }

    for term, (key, bad) in worse_values.items():
        score, parts = metrics.composite_score(degrade(base, key, bad))
        moved = abs(parts[term] - base_parts[term]) > 1e-6
        lower = score < base_score
        check(f"{term:18} contribution changes when degraded", moved,
              f"{base_parts[term]:.4f} -> {parts[term]:.4f}")
        check(f"{term:18} degraded => lower total score", lower,
              f"{base_score:.4f} -> {score:.4f}")

    # Weights must sum to 1 so the score stays interpretable.
    total = sum(metrics.WEIGHTS.values())
    check("weights sum to 1.0", abs(total - 1.0) < 1e-9, f"sum={total}")

    # Single-person weighting: the two identity terms must dominate together,
    # identity_mean must still be the single largest, and multi-person switch
    # detection must be demoted to a defensive minimum.
    ident_total = metrics.WEIGHTS["identity"] + metrics.WEIGHTS["identity_worst"]
    check("identity mean + worst-case are the dominant pair", ident_total >= 0.40,
          f"{ident_total:.2f}")
    check("identity mean is the single largest weight",
          metrics.WEIGHTS["identity"] == max(metrics.WEIGHTS.values()))
    check("identity_switches demoted for single-person use",
          metrics.WEIGHTS["identity_switches"] <= 0.03,
          f"{metrics.WEIGHTS['identity_switches']:.2f}")
    check("identity_switches retained as a defensive guard, not deleted",
          metrics.WEIGHTS["identity_switches"] > 0.0)
    check("speed is among the smallest weights",
          metrics.WEIGHTS["speed"] <= 0.03)
    # A quality term must never be outranked by speed.
    for q in ("identity", "identity_worst", "temporal", "expression", "blending", "detail"):
        check(f"{q:16} outranks speed", metrics.WEIGHTS[q] > metrics.WEIGHTS["speed"])

    # Worst-case identity must move the score independently of the mean:
    # two pipelines with the same average but different worst frames must
    # not tie, or the term is decorative.
    same_mean_good = metrics.composite_score({**base, "identity_min": 0.60})[0]
    same_mean_bad = metrics.composite_score({**base, "identity_min": 0.30})[0]
    check("worst-case identity separates equal-mean candidates",
          same_mean_good - same_mean_bad > 0.02,
          f"{same_mean_bad:.4f} -> {same_mean_good:.4f}")

    # A uniformly bad candidate must lose to a uniformly good one.
    bad = {k: v for k, v in base.items()}
    for key, val in [("identity_mean", 0.20), ("identity_switches", 6),
                     ("identity_stability_mean", 0.78), ("seam_mean", 1.5),
                     ("expression_delta_mean", 0.08), ("sharpness_mean", 50.0),
                     ("flow_flicker_mean", 14.0), ("mask_jitter_mean", 0.12)]:
        bad[key] = val
    bad_score, _ = metrics.composite_score(bad)
    check("bad candidate scores far below good", bad_score < base_score * 0.6,
          f"{bad_score:.4f} vs {base_score:.4f}")

    # Identity must DISCRIMINATE across the range real models produce, not
    # saturate. This guards the exact failure that let a 0.77-identity
    # pipeline outscore a 0.96 one.
    good_id, _ = metrics.composite_score(degrade(base, "identity_mean", 0.96))
    mid_id, _ = metrics.composite_score(degrade(base, "identity_mean", 0.77))
    check("identity discriminates between 0.77 and 0.96", good_id - mid_id > 0.04,
          f"{mid_id:.4f} -> {good_id:.4f} (gap {good_id - mid_id:.4f})")
    for lo, hi in ((0.45, 0.60), (0.60, 0.75), (0.75, 0.90)):
        a, _ = metrics.composite_score(degrade(base, "identity_mean", lo))
        b, _ = metrics.composite_score(degrade(base, "identity_mean", hi))
        check(f"identity gradient across {lo}-{hi}", b > a, f"{a:.4f} -> {b:.4f}")

    # A missing metric must not crash or silently win.
    partial = {"identity_mean": 0.5}
    ps, _ = metrics.composite_score(partial)
    check("missing metrics degrade gracefully", 0.0 <= ps <= 1.0, f"score={ps:.4f}")


def main() -> int:
    print("Auto Max scoring regression tests")
    test_metric_responses()
    test_composite_behaviour()
    print(f"\n{'='*62}")
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
    print(f"{'='*62}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
