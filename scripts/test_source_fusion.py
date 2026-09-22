"""Does intelligent source fusion beat the current production fusion?

The decisive question is not "is the clean set slightly better" -- it is
"when the user uploads one bad photo among good ones, how far does the
identity move?" A user's five photos will include a peace sign, a hand, a
blurry one. The fusion that survives that wins.

Method: build a CLEAN baseline identity from good photos, then replace one
photo with a contaminated version and measure how far the fused identity
drifts from the clean baseline. Lower drift is better. Both fusions see
exactly the same inputs.

Contaminations are applied to the IMAGE, so the analysis sees what a real
bad upload looks like rather than a synthetic embedding perturbation.

Run:  python scripts/test_source_fusion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIX = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/fixtures")
AB = ROOT / "data" / "testclips" / "ab"
OUT = ROOT / "data" / "benchmarks" / "source_fusion"


def identity_sets() -> dict[str, list[Path]]:
    sets: dict[str, list[Path]] = {}
    a = [p for p in sorted(FIX.glob("a[0-9]_*.jpg")) if p.stat().st_size > 5000]
    a = [AB / "_source_a.png"] + a
    if len(a) >= 3:
        sets["A_chawla"] = a[:5]
    w = sorted((FIX / "wsrc").glob("w_photo*.png"))
    if len(w) >= 3:
        sets["W_walker"] = w[:5]
    m = sorted((FIX / "msrc").glob("m_*.jpg"))
    if len(m) >= 3:
        sets["M_melvin"] = m[:5]
    return sets


# ---------------------------------------------------------------- contaminate
def _face_of(img: np.ndarray):
    from app.render import detection
    fs = detection.detect_robust(img, 0.3)
    return max(fs, key=lambda x: x.area) if fs else None


def contaminate(img: np.ndarray, kind: str) -> Optional[np.ndarray]:
    f = _face_of(img)
    if f is None:
        return None
    o = img.copy()
    cx, cy = f.centre
    sz = f.size
    le, re = f.kps[0], f.kps[1]

    if kind == "hand_cheek":
        # Peace sign over the left cheek: two fingers plus a palm, skin-toned,
        # soft-edged -- the classic "still a good photo of me" upload.
        m = np.zeros(img.shape[:2], np.float32)
        hx, hy = int(cx - sz * 0.55), int(cy + sz * 0.25)
        for dx in (-int(sz * 0.10), int(sz * 0.10)):
            cv2.ellipse(m, (hx + dx, hy - int(sz * 0.22)),
                        (int(sz * 0.075), int(sz * 0.30)), 12, 0, 360, 1.0, -1)
        cv2.ellipse(m, (hx, hy + int(sz * 0.18)),
                    (int(sz * 0.26), int(sz * 0.22)), 8, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), sz * 0.02)[:, :, None]
        skin = np.full_like(o, (118, 150, 195), np.uint8).astype(np.float32)
        return np.clip(o * (1 - m) + skin * m, 0, 255).astype(np.uint8)

    if kind == "hand_eye":
        m = np.zeros(img.shape[:2], np.float32)
        cv2.ellipse(m, (int(le[0]), int(le[1])),
                    (int(sz * 0.30), int(sz * 0.22)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), sz * 0.025)[:, :, None]
        skin = np.full_like(o, (116, 148, 192), np.uint8).astype(np.float32)
        return np.clip(o * (1 - m) + skin * m, 0, 255).astype(np.uint8)

    if kind == "sunglasses":
        m = np.zeros(img.shape[:2], np.float32)
        for c in (le, re):
            cv2.ellipse(m, (int(c[0]), int(c[1])),
                        (int(sz * 0.26), int(sz * 0.17)), 0, 0, 360, 1.0, -1)
        cv2.line(m, (int(le[0]), int(le[1])), (int(re[0]), int(re[1])),
                 1.0, max(2, int(sz * 0.05)))
        m = cv2.GaussianBlur(m, (0, 0), 2.0)[:, :, None]
        dark = np.full_like(o, (18, 16, 20), np.uint8).astype(np.float32)
        return np.clip(o * (1 - m) + dark * m, 0, 255).astype(np.uint8)

    if kind == "blur":
        k = max(3, int(sz * 0.14) | 1)
        return cv2.GaussianBlur(o, (k, k), 0)

    if kind == "overexposed":
        return np.clip(o.astype(np.float32) * 1.75 + 46, 0, 255).astype(np.uint8)

    if kind == "underexposed":
        return np.clip(o.astype(np.float32) * 0.32, 0, 255).astype(np.uint8)

    if kind == "hair":
        m = np.zeros(img.shape[:2], np.float32)
        for i in range(8):
            x0 = int(cx - sz * 0.5 + i * sz * 0.14)
            pts = np.array([[x0, int(cy - sz * 0.95)],
                            [x0 + int(sz * 0.12), int(cy)],
                            [x0 - int(sz * 0.04), int(cy + sz * 0.85)]], np.int32)
            cv2.polylines(m, [pts], False, 1.0, max(2, int(sz * 0.045)))
        m = cv2.GaussianBlur(m, (0, 0), sz * 0.012)[:, :, None]
        hair = np.full_like(o, (26, 24, 30), np.uint8).astype(np.float32)
        return np.clip(o * (1 - m) + hair * m, 0, 255).astype(np.uint8)
    return None


# ---------------------------------------------------------------- fusions
def load(paths: list[Path]):
    from app.render import detection
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        f = _face_of(img)
        if f is not None:
            out.append((p.name, img, f))
    return out


def fuse_current(items) -> np.ndarray:
    from app.render import recognition
    ident = recognition.build_source_identity(items)
    return np.asarray(ident.embedding, np.float32).ravel()


def fuse_new(items) -> tuple[np.ndarray, dict]:
    from app.render import source_analysis as SA, source_fusion as SF
    an = [SA.analyse(n, img, f) for n, img, f in items]
    v, rep = SF.build(an)
    return v, rep


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (float(np.linalg.norm(a)) or 1.0)
    b = b / (float(np.linalg.norm(b)) or 1.0)
    return float(np.dot(a, b))


def main() -> int:
    sets = identity_sets()
    if not sets:
        raise SystemExit("no source identity sets found")
    print("identity sets: %s\n" % ", ".join(
        "%s(%d)" % (k, len(v)) for k, v in sets.items()))

    kinds = ["hand_cheek", "hand_eye", "sunglasses", "blur",
             "overexposed", "underexposed", "hair"]
    rows: list[dict[str, Any]] = []

    for sname, paths in sets.items():
        items = load(paths)
        if len(items) < 3:
            print("%s: too few usable photos" % sname)
            continue

        clean_cur = fuse_current(items)
        clean_new, rep = fuse_new(items)
        print("=== %s: %d photos | new fusion: %s" % (sname, len(items), rep["subset"]))
        for p in rep["photos"]:
            print("      %-22s yaw %6.1f  vis %.2f  w %.3f  %s%s"
                  % (p["name"][:22], p["yaw"], p["visible_fraction"], p["weight"],
                     ",".join(p["occluded_regions"]) or "-",
                     "  DUP" if p["duplicate_of"] else ""))

        print("    %-14s %10s %10s   (drift from that fusion's own clean baseline)"
              % ("contamination", "current", "new"))
        for kind in kinds:
            # Contaminate the LAST photo; the others still show the anatomy.
            worst = None
            bad_items = list(items)
            c = contaminate(items[-1][1], kind)
            if c is None:
                continue
            f = _face_of(c)
            if f is None:
                # A contamination that destroys detection is itself a result.
                print("    %-14s %10s %10s   (face lost)" % (kind, "-", "-"))
                continue
            bad_items[-1] = (items[-1][0] + "#" + kind, c, f)

            cur = fuse_current(bad_items)
            new, _ = fuse_new(bad_items)
            d_cur = 1.0 - cos(cur, clean_cur)
            d_new = 1.0 - cos(new, clean_new)
            rows.append({"set": sname, "kind": kind,
                         "drift_current": d_cur, "drift_new": d_new})
            flag = "  <-- better" if d_new < d_cur * 0.85 else (
                "  worse" if d_new > d_cur * 1.15 else "")
            print("    %-14s %10.5f %10.5f%s" % (kind, d_cur, d_new, flag))
        print()

    if rows:
        dc = np.array([r["drift_current"] for r in rows])
        dn = np.array([r["drift_new"] for r in rows])
        print("=" * 66)
        print("CONTAMINATION DRIFT, %d cases across %d identities"
              % (len(rows), len(sets)))
        print("  current fusion  mean %.5f  worst %.5f" % (dc.mean(), dc.max()))
        print("  new fusion      mean %.5f  worst %.5f" % (dn.mean(), dn.max()))
        print("  improvement     mean %.1f%%  worst %.1f%%"
              % (100 * (1 - dn.mean() / max(dc.mean(), 1e-9)),
                 100 * (1 - dn.max() / max(dc.max(), 1e-9))))
        better = int((dn < dc * 0.85).sum())
        same = int(((dn >= dc * 0.85) & (dn <= dc * 1.15)).sum())
        print("  new is better in %d/%d cases, equivalent in %d, worse in %d"
              % (better, len(rows), same, len(rows) - better - same))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "drift.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
