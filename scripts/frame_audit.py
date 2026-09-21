"""Priority 1: count frames at every stage to find where they are lost.

A 576-frame source produced 573 frames continuous and 564 segmented. The
previous pass established that rendering is NOT at fault -- each -ss/-t
window decodes exactly the right count, and the segment plan is arithmetically
exact -- so the loss is downstream. This narrows it to a single stage by
counting at every boundary instead of inferring.

Counts are taken by DECODING, never from container metadata: nb_frames is a
header field that a muxer may write optimistically, and trusting it is how a
frame-loss bug hides.

Run:  python scripts/frame_audit.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"
WORK = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/audit")


def decoded_count(path: Path) -> int:
    """Real decoded frame count. Never trust nb_frames."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=900)
    try:
        return int(r.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        return -1


def stream_facts(path: Path) -> dict[str, Any]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames",
         "-show_entries", "format=duration,start_time", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=300)
    try:
        d = json.loads(r.stdout)
        s = (d.get("streams") or [{}])[0]
        f = d.get("format") or {}
        return {**s, "format_duration": f.get("duration"),
                "format_start": f.get("start_time")}
    except ValueError:
        return {}


def pts_list(path: Path, limit: int = 100000) -> list[int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=900)
    out = []
    for line in r.stdout.splitlines()[:limit]:
        line = line.strip().rstrip(",")
        if line and line != "N/A":
            try:
                out.append(int(line))
            except ValueError:
                pass
    return out


def main() -> int:
    from app.render import longform, pipeline
    from app.render.pipeline import RenderOptions
    import app.render.video as V

    WORK.mkdir(parents=True, exist_ok=True)
    src = WORK / "src.mp4"
    if not src.is_file():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "7",
                        "-i", str(AB / "G_talking.mp4"), "-c", "copy", str(src)],
                       check=True, timeout=300)

    print("=" * 74)
    print("STAGE A/I/J -- SOURCE")
    print("=" * 74)
    facts = stream_facts(src)
    info = V.probe(str(src))
    n_src = decoded_count(src)
    print("  decoded frames        : %d" % n_src)
    print("  container nb_frames   : %s" % facts.get("nb_frames"))
    print("  r_frame_rate          : %s" % facts.get("r_frame_rate"))
    print("  avg_frame_rate        : %s" % facts.get("avg_frame_rate"))
    print("  time_base             : %s" % facts.get("time_base"))
    print("  stream start_time     : %s" % facts.get("start_time"))
    print("  stream duration       : %s" % facts.get("duration"))
    print("  probe fps / is_vfr    : %.6f / %s" % (info.fps, info.is_vfr))
    pts = pts_list(src)
    if len(pts) > 2:
        d = np.diff(pts)
        print("  PTS step: min %d  max %d  mode %d  unique %d"
              % (d.min(), d.max(), int(np.bincount(d - d.min()).argmax() + d.min()),
                 len(set(d.tolist()))))

    print()
    print("=" * 74)
    print("STAGE C -- PER-SEGMENT DECODE (does -ss/-t lose frames?)")
    print("=" * 74)
    seg_len = info.duration / 4.0
    tot = 0
    for i in range(4):
        a = i * seg_len
        dec = V.decode(str(src), info, start=a, duration=seg_len)
        n = sum(1 for _ in V.frames(dec, info))
        try:
            dec.kill()
        except OSError:
            pass
        tot += n
        print("  seg %d  start %8.4f  dur %8.4f  ->  %4d frames" % (i, a, seg_len, n))
    print("  segmented decode total: %d" % tot)
    dec = V.decode(str(src), info)
    n_cont_dec = sum(1 for _ in V.frames(dec, info))
    try:
        dec.kill()
    except OSError:
        pass
    print("  continuous decode     : %d" % n_cont_dec)

    print()
    print("=" * 74)
    print("STAGE E/F -- ENCODE ONLY (feed N frames, how many come out?)")
    print("=" * 74)
    for n_in in (100, n_cont_dec):
        tmp = WORK / ("enc_%d.mp4" % n_in)
        tmp.unlink(missing_ok=True)
        enc, name = V.open_encoder(str(tmp), info, "quality")
        blank = np.zeros((info.height, info.width, 3), np.uint8)
        for k in range(n_in):
            blank[:] = (k % 255)
            enc.stdin.write(memoryview(np.ascontiguousarray(blank)))
        enc.stdin.close()
        enc.wait(timeout=600)
        got = decoded_count(tmp)
        print("  wrote %5d frames -> encoder produced %5d   (%s)  delta %+d"
              % (n_in, got, name, got - n_in))

    print()
    print("=" * 74)
    print("STAGE G -- AUDIO MUX (finalize)")
    print("=" * 74)
    silent = WORK / ("enc_%d.mp4" % n_cont_dec)
    muxed = WORK / "muxed.mp4"
    muxed.unlink(missing_ok=True)
    before = decoded_count(silent)
    V.finalize(str(silent), str(src), str(muxed), info)
    after = decoded_count(muxed)
    print("  before mux %d  ->  after mux %d   delta %+d" % (before, after, after - before))
    mf = stream_facts(muxed)
    print("  muxed r_frame_rate %s  avg %s  time_base %s  start %s"
          % (mf.get("r_frame_rate"), mf.get("avg_frame_rate"),
             mf.get("time_base"), mf.get("start_time")))

    print()
    print("=" * 74)
    print("STAGE H -- CONCAT")
    print("=" * 74)
    parts = []
    for i in range(4):
        p = WORK / ("part_%d.mp4" % i)
        p.unlink(missing_ok=True)
        enc, _ = V.open_encoder(str(p), info, "quality")
        blank = np.zeros((info.height, info.width, 3), np.uint8)
        for k in range(145):
            blank[:] = ((i * 40 + k) % 255)
            enc.stdin.write(memoryview(np.ascontiguousarray(blank)))
        enc.stdin.close()
        enc.wait(timeout=600)
        c = decoded_count(p)
        print("  part %d: %d frames" % (i, c))
        parts.append((str(p), c))
    joined = WORK / "joined.mp4"
    joined.unlink(missing_ok=True)
    V.concat([p for p, _ in parts], str(joined))
    total_parts = sum(c for _, c in parts)
    got = decoded_count(joined)
    print("  sum of parts %d  ->  concat produced %d   delta %+d"
          % (total_parts, got, got - total_parts))
    jf = stream_facts(joined)
    print("  joined r_frame_rate %s  avg %s  duration %s"
          % (jf.get("r_frame_rate"), jf.get("avg_frame_rate"), jf.get("duration")))

    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print("  source decoded            : %d" % n_src)
    print("  continuous decode         : %d" % n_cont_dec)
    print("  segmented decode (4x)     : %d" % tot)
    print("  encoder round-trip delta  : see STAGE E")
    print("  mux delta                 : %+d" % (after - before))
    print("  concat delta              : %+d" % (got - total_parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
