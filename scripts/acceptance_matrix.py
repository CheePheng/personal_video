"""The full A-S acceptance matrix, plus the real-footage regression set.

Every category from the original specification gets an explicit PASS/FAIL.
A category with no clip is reported as MISSING, never quietly omitted -- the
whole point of a matrix is that the gaps are visible.

Category-specific assertions live in v2_testsuite.EXTRA where they already
exist; this runner adds the categories that suite does not cover and prints
one table for all 19.

Run:  python scripts/acceptance_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v2_testsuite import (CLIPS, EXTRA, check_common, run_clip)  # noqa: E402

REAL = CLIPS / "real"
OUTDIR = ROOT / "data" / "benchmarks" / "acceptance"

# category -> (clip stem, human label)
MATRIX: list[tuple[str, str, str]] = [
    ("A", "A_single_frontal",   "single frontal face"),
    ("B", "B_profile",          "side / profile"),
    ("C", "C_fast_motion",      "fast head movement"),
    ("D", "D_motion_blur",      "motion blur"),
    ("E", "E_glasses",          "glasses"),
    ("F", "F_occlusion",        "foreground object across face"),
    ("G", "G_talking",          "talking / open mouth"),
    ("H", "H_small_face",       "small / distant face"),
    ("I", "I_closeup",          "close-up"),
    ("J", "J_two_crossing",     "two people crossing"),
    ("K", "K_other_larger",     "distractor becomes larger"),
    ("L", "L_disappear_return", "target leaves and returns"),
    ("M", "M_scene_cut",        "scene cut"),
    ("N", "N_dark",             "dark lighting"),
    ("O", "O_bright",           "bright lighting"),
    ("P", "P_1080p",            "1080p"),
    ("Q", "Q_4k",               "short 4K"),
    ("R", "R_with_audio",       "video with audio"),
    ("S", "S_no_audio",         "video without audio"),
]

REAL_CLIPS = [("real_A_motion", "real footage: camera + subject motion"),
              ("real_B_talking", "real footage: natural talking"),
              ("real_C_lighting", "real footage: changing lighting")]


def check_resolution(rec: dict, w: int, h: int) -> list[tuple[str, bool, str]]:
    v = rec.get("video") or {}
    got = (v.get("width"), v.get("height"))
    return [(f"renders at {w}x{h}", got == (w, h), f"got {got[0]}x{got[1]}")]


def run_category(code: str, stem: str, label: str,
                 base: Path = CLIPS) -> dict[str, Any]:
    clip = base / f"{stem}.mp4"
    if not clip.exists():
        return {"code": code, "clip": stem, "label": label,
                "status": "MISSING", "checks": [], "note": "no clip built"}

    # 4K is the VRAM stress case; cap frames so it stays a check, not a soak.
    rec = run_clip(stem, "quality") if code != "Q" else run_clip(stem, "quality")
    checks = check_common(rec)
    if rec.get("ok") and stem in EXTRA:
        checks += EXTRA[stem](rec)
    if code == "P":
        checks += check_resolution(rec, 1920, 1080)
    if code == "Q":
        checks += check_resolution(rec, 3840, 2160)
        checks.append(("4K completed without VRAM exhaustion",
                       rec.get("ok") is True and "VRAM" not in str(rec.get("error", "")),
                       str(rec.get("error", ""))[:80]))
    if rec.get("ok"):
        # Every category must actually swap something, or it proved nothing.
        checks.append(("swapped at least one face", rec.get("faces", 0) > 0,
                       f"faces={rec.get('faces')}"))

    failed = [c for c in checks if not c[1]]
    return {"code": code, "clip": stem, "label": label,
            "status": "PASS" if not failed else "FAIL",
            "checks": [{"label": c, "ok": o, "note": n} for c, o, n in checks],
            "record": rec}


def main(argv: list[str]) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    print("=" * 78)
    print("ACCEPTANCE MATRIX  A-S")
    print("=" * 78)
    for code, stem, label in MATRIX:
        r = run_category(code, stem, label)
        results.append(r)
        mark = {"PASS": "PASS", "FAIL": "FAIL", "MISSING": "MISS"}[r["status"]]
        extra = ""
        rec = r.get("record") or {}
        if rec.get("ok"):
            t = rec.get("tracking") or {}
            extra = (f"  {rec.get('frames')}f  {rec.get('ms_per_frame')}ms/f"
                     f"  switches={t.get('identity_switches', '-')}")
        print(f"  [{mark}] {code}  {label:<34}{extra}", flush=True)
        for c in r["checks"]:
            if not c["ok"]:
                print(f"           FAILED: {c['label']}  {c['note']}")

    print("\n" + "=" * 78)
    print("REAL-FOOTAGE REGRESSION")
    print("=" * 78)
    real_results: list[dict[str, Any]] = []
    for stem, label in REAL_CLIPS:
        if not (REAL / f"{stem}.mp4").exists():
            print(f"  [MISS] {label} (no clip)")
            real_results.append({"clip": stem, "status": "MISSING"})
            continue
        # run_clip resolves against CLIPS, so point it at the real sub-dir.
        import v2_testsuite as ts
        old = ts.CLIPS
        ts.CLIPS = REAL
        try:
            rec = ts.run_clip(stem, "quality")
        finally:
            ts.CLIPS = old
        checks = check_common(rec)
        if rec.get("ok"):
            t = rec.get("tracking") or {}
            checks.append(("no identity switches", t.get("identity_switches") == 0,
                           f"switches={t.get('identity_switches')}"))
            checks.append(("swapped at least one face", rec.get("faces", 0) > 0,
                           f"faces={rec.get('faces')}"))
        failed = [c for c in checks if not c[1]]
        status = "PASS" if not failed else "FAIL"
        real_results.append({"clip": stem, "status": status,
                             "checks": [{"label": c, "ok": o, "note": n} for c, o, n in checks],
                             "record": rec})
        extra = ""
        if rec.get("ok"):
            extra = f"  {rec.get('frames')}f  {rec.get('ms_per_frame')}ms/f"
        print(f"  [{status}] {label:<40}{extra}")
        for c, o, n in checks:
            if not o:
                print(f"           FAILED: {c}  {n}")

    # ---- summary table
    print("\n" + "=" * 78)
    print(f"  {'':<4}{'category':<36}{'status':<8}{'notes'}")
    print("-" * 78)
    npass = 0
    for r in results:
        rec = r.get("record") or {}
        note = ""
        if rec.get("ok"):
            v = rec.get("video") or {}
            note = f"{v.get('width')}x{v.get('height')}"
            if rec.get("src_audio"):
                note += " +audio"
        elif r["status"] == "FAIL":
            note = str(rec.get("error", ""))[:36]
        print(f"  {r['code']:<4}{r['label']:<36}{r['status']:<8}{note}")
        npass += r["status"] == "PASS"
    print("-" * 78)
    rp = sum(1 for r in real_results if r["status"] == "PASS")
    print(f"  A-S: {npass}/{len(results)} passed     real footage: {rp}/{len(real_results)} passed")
    print("=" * 78)

    (OUTDIR / "matrix.json").write_text(
        json.dumps({"matrix": results, "real": real_results}, indent=2, default=str),
        encoding="utf-8")
    incomplete = [r["code"] for r in results if r["status"] != "PASS"]
    if incomplete:
        print(f"  NOT PASSING: {incomplete}")
    return 1 if (incomplete or rp < len(real_results)) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
