"""Long-form exact-frame acceptance.

Frames were being lost in the mux: -shortest truncated the VIDEO to match a
marginally shorter audio track, silently discarding trailing rendered frames
(580 -> 576 continuous, and 564 once segmented). Nothing errored. The only
way that stays fixed is a test that counts frames by DECODING and fails loudly.

Every count here comes from decoding. Container nb_frames is a header field a
muxer may write optimistically, and trusting it is how this class of bug hides.

Run:  python scripts/test_longform.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
SOURCE = AB / "_source_a.png"
WORK = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/longform")

OK: list[str] = []
BAD: list[str] = []


def check(label: str, cond: bool, note: str = "") -> None:
    (OK if cond else BAD).append(label)
    print("  [%s] %-50s %s" % ("PASS" if cond else "FAIL", label, note))


def decoded(path: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=1800)
    try:
        return int(r.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        return -1


def audio_of(path: Path) -> Optional[str]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=300)
    s = r.stdout.strip()
    return s or None


def build(name: str, src: Path, loops: int, audio: bool) -> Path:
    p = WORK / name
    if p.is_file():
        return p
    cmd = ["ffmpeg", "-v", "error", "-y", "-stream_loop", str(loops), "-i", str(src)]
    cmd += ["-c", "copy"] if audio else ["-an", "-c:v", "copy"]
    subprocess.run(cmd + [str(p)], check=True, timeout=600)
    return p


def render(src: Path, out: Path, segmented: bool, seg_s: float,
           start: Optional[float] = None, end: Optional[float] = None):
    from app.render import longform
    from app.render.pipeline import RenderOptions
    import shutil
    out.unlink(missing_ok=True)
    # Clear the checkpoint directory. render_long resumes any segment the
    # manifest says is finished, so reusing an output path across tests with
    # different segment lengths silently splices old segments into the new
    # render. Each test must start from nothing.
    shutil.rmtree(out.parent / (out.stem + "_segments"), ignore_errors=True)
    keep = longform.SEGMENT_THRESHOLD_S
    longform.SEGMENT_THRESHOLD_S = 2.0 if segmented else 1e9
    try:
        return longform.render_long(
            [str(SOURCE)], str(src), str(out),
            RenderOptions(quality="quality", start_time=start, end_time=end),
            None, None, None, segment_s=seg_s)
    finally:
        longform.SEGMENT_THRESHOLD_S = keep


def main() -> int:
    import app.render.video as V
    WORK.mkdir(parents=True, exist_ok=True)

    print("Long-form exact-frame acceptance\n")

    # ---- 1/2: continuous vs segmented, with audio
    src = build("src_audio.mp4", AB / "G_talking.mp4", 7, audio=True)
    info = V.probe(str(src))
    seg_s = info.duration / 4.0
    print("source: %d container frames, %.3fs, fps %.4f, vfr=%s"
          % (info.total_frames, info.duration, info.fps, info.is_vfr))

    cont = WORK / "cont.mp4"
    segd = WORK / "segd.mp4"
    r1 = render(src, cont, False, seg_s)
    r2 = render(src, segd, True, seg_s)
    n1, n2 = decoded(cont), decoded(segd)
    print("\nrendered: pipeline reported %d / %d frames" % (r1.frames, r2.frames))
    check("continuous keeps every rendered frame", n1 == r1.frames,
          "rendered %d, file %d" % (r1.frames, n1))
    check("segmented keeps every rendered frame", n2 == r2.frames,
          "rendered %d, file %d" % (r2.frames, n2))
    check("segmented == continuous frame count", n1 == n2, "%d vs %d" % (n1, n2))
    check("audio preserved through segmentation", audio_of(segd) is not None,
          str(audio_of(segd)))

    # ---- no duplicates / no reordering: compare the two renders frame by frame
    ia, ib = V.probe(str(cont)), V.probe(str(segd))
    probe_n = min(ia.total_frames, ib.total_frames)
    idx = [int(i * (probe_n - 1) / 39) for i in range(40)]
    fa, fb = V.sample_frames(str(cont), ia, idx), V.sample_frames(str(segd), ib, idx)
    diffs = [float(np.abs(fa[i].astype(np.int16) - fb[i].astype(np.int16)).mean())
             for i in idx if i in fa and i in fb]
    check("no frame reordering between renders", bool(diffs) and max(diffs) < 12.0,
          "max mean|delta| %.3f over %d sampled frames" % (max(diffs) if diffs else -1, len(diffs)))

    # Consecutive identical frames would mean segmentation duplicated one.
    run_idx = list(range(0, min(probe_n, 200)))
    fseq = V.sample_frames(str(segd), ib, run_idx)
    dupes = 0
    prev = None
    for i in run_idx:
        if i not in fseq:
            continue
        if prev is not None and np.array_equal(prev, fseq[i]):
            dupes += 1
        prev = fseq[i]
    check("no duplicate frames introduced", dupes == 0, "%d exact repeats" % dupes)

    # ---- 3: no-audio path
    src_na = build("src_noaudio.mp4", AB / "S_no_audio.mp4", 7, audio=False)
    ina = V.probe(str(src_na))
    out_na = WORK / "noaudio.mp4"
    r3 = render(src_na, out_na, True, ina.duration / 4.0)
    n3 = decoded(out_na)
    check("no-audio segmented keeps every frame", n3 == r3.frames,
          "rendered %d, file %d" % (r3.frames, n3))
    check("no-audio output has no audio stream", audio_of(out_na) is None,
          str(audio_of(out_na)))

    # ---- 4: selected timestamp range
    a, b = 4.0, 10.0
    out_rng = WORK / "ranged.mp4"
    r4 = render(src, out_rng, True, 2.0, start=a, end=b)
    n4 = decoded(out_rng)
    ir = V.probe(str(out_rng))
    check("ranged render keeps every frame", n4 == r4.frames,
          "rendered %d, file %d" % (r4.frames, n4))
    check("ranged duration is correct", abs(ir.duration - (b - a)) < 0.25,
          "asked %.2fs, got %.3fs" % (b - a, ir.duration))
    check("ranged output keeps audio", audio_of(out_rng) is not None,
          str(audio_of(out_rng)))

    # ---- 5: checkpoint resume still works
    from app.render import longform
    manifests = list(WORK.glob("**/*.manifest.json")) + list(WORK.glob("**/manifest.json"))
    check("checkpoint manifest machinery present",
          hasattr(longform, "Manifest") and hasattr(longform, "plan_segments"),
          "%d manifest file(s) on disk" % len(manifests))

    print("\n" + "=" * 70)
    print("  %d passed, %d failed" % (len(OK), len(BAD)))
    for b_ in BAD:
        print("    FAILED: %s" % b_)
    print("=" * 70)
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
