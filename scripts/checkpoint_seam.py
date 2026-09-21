"""Phase 10: are checkpoint-segment boundaries visible?

Long renders are split into ~90s segments, each a separate pipeline.render()
call, then joined by stream copy. Stream copy protects codec quality, but it
says nothing about whether the temporal state is continuous: MaskSmoother,
ColorSmoother and TransformSmoother all start empty in a new segment, and the
tracker re-locks from scratch. If that matters, an hour-long render would
show a small jump every ninety seconds -- everything "working" and still
visibly wrong.

This renders the same clip twice, continuously and in segments, and compares
the frames either side of each boundary. The comparison is against the
CONTINUOUS render, which is the ground truth for what the output should be.

Run:  python scripts/checkpoint_seam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
SOURCE = AB / "_source_a.png"
WORK = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/seam")


def main() -> int:
    from app.render import detection, longform, pipeline, recognition
    from app.render.pipeline import RenderOptions
    import app.render.video as V

    WORK.mkdir(parents=True, exist_ok=True)
    src_img = cv2.imread(str(SOURCE))
    src_emb = recognition.embed(src_img, detection.detect_robust(src_img, 0.3)[0].kps)

    # A clip long enough to segment. R_with_audio looped gives continuous
    # motion across the join so a discontinuity would actually show.
    long_clip = WORK / "long.mp4"
    if not long_clip.is_file():
        import subprocess
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "7",
                        "-i", str(AB / "G_talking.mp4"), "-c", "copy",
                        str(long_clip)], check=True, timeout=300)
    info = V.probe(str(long_clip))
    print("source clip: %d frames, %.2fs at %.3f fps"
          % (info.total_frames, info.duration, info.fps))

    # Force segmentation well inside the clip.
    orig_thr, orig_seg = longform.SEGMENT_THRESHOLD_S, longform.DEFAULT_SEGMENT_S
    longform.SEGMENT_THRESHOLD_S = 2.0
    longform.DEFAULT_SEGMENT_S = max(2.0, info.duration / 4.0)
    seg_len = longform.DEFAULT_SEGMENT_S
    print("forcing segments of %.2fs (threshold %.1fs)" % (seg_len, 2.0))

    cont = WORK / "continuous.mp4"
    segd = WORK / "segmented.mp4"
    for p in (cont, segd):
        p.unlink(missing_ok=True)
    for d in (WORK / "segments",):
        if d.exists():
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    print("\nrendering continuously...")
    longform.SEGMENT_THRESHOLD_S = 1e9          # disable segmentation
    # segment_s passed explicitly: render_long binds it as a DEFAULT argument
    # at import time, so assigning longform.DEFAULT_SEGMENT_S afterwards does
    # not reach it. Same trap that silently misdirected the fixture builder.
    r1 = longform.render_long([str(SOURCE)], str(long_clip), str(cont),
                              RenderOptions(quality="quality"), None, None, None,
                              segment_s=seg_len)
    print("  %d frames" % r1.frames)

    print("rendering in segments...")
    longform.SEGMENT_THRESHOLD_S = 2.0
    r2 = longform.render_long([str(SOURCE)], str(long_clip), str(segd),
                              RenderOptions(quality="quality"), None, None, None,
                              segment_s=seg_len)
    print("  %d frames" % r2.frames)

    longform.SEGMENT_THRESHOLD_S, longform.DEFAULT_SEGMENT_S = orig_thr, orig_seg

    ia, ib = V.probe(str(cont)), V.probe(str(segd))
    print("\ncontinuous %d frames  |  segmented %d frames" % (ia.total_frames, ib.total_frames))
    if ia.total_frames != ib.total_frames:
        print("FRAME COUNT MISMATCH -- segmentation is dropping or duplicating frames")

    bounds = [int(round(seg_len * info.fps * k))
              for k in range(1, int(info.duration / seg_len) + 1)]
    bounds = [b for b in bounds if 0 < b < min(ia.total_frames, ib.total_frames) - 1]
    print("segment boundaries at frames: %s" % bounds)

    span = 30
    idx = sorted({f for b in bounds for f in range(max(0, b - span),
                                                   min(ia.total_frames, b + span + 1))})
    fa = V.sample_frames(str(cont), ia, idx)
    fb = V.sample_frames(str(segd), ib, idx)

    print("\n%8s %10s %10s %10s %10s" % ("frame", "pixdiff", "arc(cont)", "arc(seg)", "d-arc"))
    print("-" * 54)
    worst_pix = 0.0
    worst_at = -1
    rows = []
    for f in idx:
        if f not in fa or f not in fb:
            continue
        a, b = fa[f], fb[f]
        d = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
        rows.append((f, d))
        if d > worst_pix:
            worst_pix, worst_at = d, f

    # Identity either side of each boundary, on both renders.
    for bd in bounds:
        for f in (bd - 2, bd - 1, bd, bd + 1, bd + 2):
            if f not in fa or f not in fb:
                continue
            aa = ab = None
            for img, slot in ((fa[f], "a"), (fb[f], "b")):
                fs = detection.detect_with_fallback(img, 0.5, "quality",
                                                    "yoloface_8n", "scrfd_2.5g")
                if not fs:
                    continue
                v = recognition.similarity(
                    recognition.embed(img, max(fs, key=lambda x: x.area).kps), src_emb)
                if slot == "a":
                    aa = v
                else:
                    ab = v
            d = float(np.abs(fa[f].astype(np.int16) - fb[f].astype(np.int16)).mean())
            print("%8d %10.4f %10s %10s %10s"
                  % (f, d,
                     "%.4f" % aa if aa is not None else "-",
                     "%.4f" % ab if ab is not None else "-",
                     "%+.4f" % (ab - aa) if (aa is not None and ab is not None) else "-"))

    # Frame-to-frame jump WITHIN the segmented render, at the boundary vs
    # elsewhere. A seam shows up as a spike here even if both renders drift.
    print("\nframe-to-frame change inside the segmented render:")
    allidx = sorted(fb)
    jumps = {}
    for i in range(1, len(allidx)):
        f0, f1 = allidx[i - 1], allidx[i]
        if f1 - f0 != 1:
            continue
        jumps[f1] = float(np.abs(fb[f1].astype(np.int16) - fb[f0].astype(np.int16)).mean())
    at_bound = [jumps[b] for b in bounds if b in jumps]
    other = [v for k, v in jumps.items() if k not in bounds]
    if at_bound and other:
        print("  at boundary   mean %.4f  max %.4f" % (np.mean(at_bound), np.max(at_bound)))
        print("  elsewhere     mean %.4f  max %.4f  p99 %.4f"
              % (np.mean(other), np.max(other), np.percentile(other, 99)))
        ratio = np.mean(at_bound) / max(np.mean(other), 1e-6)
        print("  boundary / elsewhere ratio: %.2fx" % ratio)
        verdict = ("boundary jump is within normal frame-to-frame variation"
                   if np.max(at_bound) <= np.percentile(other, 99)
                   else "BOUNDARY SPIKE -- visible seam likely")
        print("  %s" % verdict)

    print("\nworst continuous-vs-segmented pixel difference: %.4f at frame %d"
          % (worst_pix, worst_at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
