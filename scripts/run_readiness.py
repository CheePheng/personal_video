"""End-to-end run-readiness check on one normal A->B render.

Not a unit test: this does what a user does. One source face, one target
video with a different person in it, the shipping Balanced preset, then it
inspects the file that came out -- CUDA actually bound, audio actually
present, dimensions and duration actually right, and the output actually
decodable by something other than the code that wrote it.

Run:  python scripts/run_readiness.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
SOURCE = AB / "_source_a.png"
TARGET = AB / "R_with_audio.mp4"          # has a real audio stream
OUTPUT = ROOT / "data" / "outputs" / "_readiness_check.mp4"

OK: list[str] = []
BAD: list[str] = []


def check(label: str, cond: bool, note: str = "") -> None:
    (OK if cond else BAD).append(label)
    print("  [%s] %-46s %s" % ("PASS" if cond else "FAIL", label, note))


def ffprobe(path: Path) -> dict[str, Any]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout) if r.returncode == 0 else {}


def main() -> int:
    from app.render import detection, judges, pipeline, recognition
    from app.render.pipeline import RenderOptions
    from app.render.registry import get_model

    print("Run-readiness check\n")

    print("Pixel boost")
    for m in ("hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256"):
        pb = get_model(m).pixel_boost
        check("%s boost disabled" % m, pb == (), "pixel_boost=%r" % (pb,))

    print("\nFixtures")
    check("source photo present", SOURCE.is_file(), str(SOURCE.name))
    check("A->B target present", TARGET.is_file(), str(TARGET.name))
    if BAD:
        print("\nmissing fixtures; run scripts/make_ab_fixtures.py")
        return 1

    print("\nRender (Balanced preset, 256px, A->B)")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    opts = RenderOptions(quality="quality")
    t0 = time.time()
    res = pipeline.render([str(SOURCE)], str(TARGET), str(OUTPUT), opts)
    wall = time.time() - t0

    cfg = res.config.describe() if res.config else "?"
    check("render completed", res.frames > 0, "%d frames in %.1fs" % (res.frames, wall))
    check("preset is Balanced (with restoration)", "gpen_bfr_512" in cfg, cfg)
    check("no pixel boost in effect", "boost" not in cfg, cfg)
    check("CUDA provider bound", bool(res.gpu.get("cuda_active")),
          str(res.gpu.get("session_providers")))
    check("faces were swapped", res.faces_swapped > 0,
          "%d of %d frames" % (res.faces_swapped, res.frames))
    t = res.tracking or {}
    check("no identity switches", t.get("identity_switches", 0) == 0,
          "switches=%s" % t.get("identity_switches"))

    print("\nOutput file")
    check("output exists", OUTPUT.is_file())
    size = OUTPUT.stat().st_size if OUTPUT.is_file() else 0
    check("output non-empty", size > 10000, "%d bytes" % size)

    probe = ffprobe(OUTPUT)
    streams = probe.get("streams", [])
    vid = next((s for s in streams if s.get("codec_type") == "video"), None)
    aud = next((s for s in streams if s.get("codec_type") == "audio"), None)
    check("decodable by ffprobe", bool(vid), "codec=%s" % (vid or {}).get("codec_name"))

    src_probe = ffprobe(TARGET)
    src_v = next((s for s in src_probe.get("streams", [])
                  if s.get("codec_type") == "video"), None)
    if vid and src_v:
        check("dimensions preserved",
              (vid.get("width"), vid.get("height"))
              == (src_v.get("width"), src_v.get("height")),
              "%sx%s" % (vid.get("width"), vid.get("height")))
    check("audio preserved", bool(aud),
          "codec=%s" % (aud or {}).get("codec_name") if aud else "NO AUDIO STREAM")

    sd = float(src_probe.get("format", {}).get("duration") or 0)
    od = float(probe.get("format", {}).get("duration") or 0)
    check("duration preserved", abs(sd - od) < 0.25,
          "src %.3fs out %.3fs" % (sd, od))

    # +faststart: the moov atom must precede mdat or the browser cannot
    # start playing until the whole file has downloaded.
    head = OUTPUT.read_bytes()[:200000] if OUTPUT.is_file() else b""
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    check("faststart (moov before mdat)",
          moov != -1 and (mdat == -1 or moov < mdat),
          "moov@%d mdat@%d" % (moov, mdat))

    print("\nIdentity (A->B, source is NOT the person in the clip)")
    src = cv2.imread(str(SOURCE))
    sf = detection.detect_robust(src, 0.3)
    arc_vals: list[float] = []
    sfc_vals: list[float] = []
    if sf:
        src_arc = recognition.embed(src, sf[0].kps)
        src_sfc = judges.embed(src, sf[0].kps) if judges.available() else None
        import app.render.video as V
        info = V.probe(str(OUTPUT))
        idx = [int(i * (info.total_frames - 1) / 7) for i in range(8)]
        for _, fr in sorted(V.sample_frames(str(OUTPUT), info, idx).items()):
            fs = detection.detect_robust(fr, 0.4, "quality")
            if not fs:
                continue
            f = max(fs, key=lambda x: x.area)
            arc_vals.append(recognition.similarity(src_arc, recognition.embed(fr, f.kps)))
            if src_sfc is not None:
                sfc_vals.append(judges.similarity(src_sfc, judges.embed(fr, f.kps)))
    am = float(np.mean(arc_vals)) if arc_vals else 0.0
    sm = float(np.mean(sfc_vals)) if sfc_vals else 0.0
    # 0.4 is well above the ~0.02 two strangers score and below the 0.66 the
    # A->B ladder measured, so it catches a collapse without pinning a target.
    check("identity transferred (ArcFace selector)", am > 0.40, "mean %.4f" % am)
    check("identity transferred (SFace holdout)", sm > 0.40, "mean %.4f" % sm)

    print("\n" + "=" * 66)
    print("  %d passed, %d failed" % (len(OK), len(BAD)))
    if BAD:
        for b in BAD:
            print("    FAILED: %s" % b)
    print("=" * 66)
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
