# Face Swap

Local face swap for video. Runs entirely on your RTX 5070 Ti. No API keys, no
accounts, no per-use cost, no limits.

## How to use it

**Double-click `start.bat`.**

It prints a link like:

    https://joan-thumbnail-albert-scores.trycloudflare.com

Open that on any device — phone, laptop, anywhere. Then:

1. Pick the **video** you want to edit
2. Pick a **photo of the face** you want to put in it
3. Press **Swap face**
4. Wait, then download the result

Leave the black window open while it works. Closing it stops the site.

The link is different every time you start it. That's normal.

## How long it takes

Measured on this machine at 720p, Balanced (swap + enhance): **~3.5 frames/sec**,
roughly **7x the length of your video**.

| Your video | Render time |
|---|---|
| 1 min | ~7 min |
| 5 min | ~34 min |
| 10 min | ~68 min |

1080p is comfortably supported but slower — budget roughly 10–12x.

You can close the browser while it renders; the job keeps going and the page
reconnects to it when you come back.

## Don't render while gaming

A Balanced render peaks at **15.6 GB of your 16.3 GB of VRAM**. NBA 2K27 alone
holds about 8.4 GB. They do not fit together — starting a render mid-game will
either run out of memory or tank your framerate. Render when you're done playing.

## Quality settings

| Setting | Speed | Use when |
|---|---|---|
| Fast | fastest | quick preview |
| **Balanced** | default | most of the time |
| Best | slowest | final version you care about |

"Which face in the video" — pick **One person** for a single subject, or
**Everyone in frame** if several people should all be swapped.

## Tips for good results

- Use a **clear, front-facing, well-lit** photo of the face. This matters more
  than any setting.
- Avoid sunglasses, heavy shadow, or extreme angles in the face photo.
- Short clips render much faster. Test on a 10-second cut before a long video.

## If something goes wrong

**The link doesn't open.**
Wait ~20 seconds and retry — the tunnel takes a moment to route. If it still
fails, close the window and run `start.bat` again for a fresh link.

**"No face detected."**
The face photo isn't clear enough, or no face was found in the video. Try a
better-lit, front-facing photo.

**It's very slow.**
Check the black window for `CUDAExecutionProvider`. If it says CPU instead, the
GPU isn't being used — run `bash scripts/02_fix_ort_cuda.sh`.
The render also refuses to start silently on CPU: check the job error text.

## What's installed where

| Thing | Location |
|---|---|
| Python 3.12 + GPU libraries | `C:swenv` (kept off your system Python) |
| ONNX models | `models/` |
| Web app + render pipeline | `app/` |
| Your uploads | `data/uploads/` |
| Finished videos | `data/outputs/` |

Nothing in `data/` or `models/` is committed to git.

## How a render works

Everything runs in-process; there is no external CLI and no subprocess output to
parse. The pipeline lives in `app/render/` -- see [docs/V2_ARCHITECTURE.md](docs/V2_ARCHITECTURE.md).

    upload 1-5 face photos + a target video
      -> ffprobe the target (fps, frame count, rotation, VFR, audio, codec)
      -> fuse the photos into one identity, weighted by face quality
      -> decode frames with ffmpeg (raw BGR over a pipe)
      -> detect faces            yoloface_8n
      -> embed each face         arcface_w600k_r50
      -> track: which face is OUR person, by identity + motion + overlap
      -> swap                    hyperswap_1a/1b/1c
      -> mask: model + BiSeNet parsing + XSeg occlusion, temporally smoothed
      -> colour-match to the target's lighting, inside the mask only
      -> restore detail (optional, blend chosen by benchmark)
      -> encode with NVENC p7-hq, mux the original audio, +faststart
      -> Video Library

Progress is `frames_done / total_frames` -- a real count, not an estimate.

### Scope: one face, one person

This tool is built for a single target person. One source photo, one video with
one person in it, one swapped face out. That assumption is what the quality
work is tuned for.

Multi-face swapping still exists behind **Advanced**, and the tracker that
prevents identity switching is still there and still tested -- but it is not
the supported path, and Auto Max no longer spends its scoring budget on
"which person should I swap?" when there is only ever one.

### Long videos, and picking a range

**There is no duration limit and no file-size limit.** None is imposed in the
code, and none is added. An hour-long, multi-gigabyte video is a supported
input; it is simply a long job.

The only hard refusal is a measured one:

    insufficient disk space: this render needs roughly 19.4 GB of temporary
    and output space, but only 11.2 GB is free.

Everything else is reported rather than blocked. `POST /api/estimate` returns
frame count, a wall-clock estimate and disk headroom before you commit.

**Picking a range.** When you choose a video the browser reads its duration
and resolution from the *local file*, before uploading a byte — so on a 5 GB
source you can select minutes 35–42 up front instead of waiting out an upload
to discover you only wanted seven minutes. The renderer then decodes only that
window (`-ss` before `-i`, so seeking an hour-long file costs seconds), and
trims the audio to exactly the same window.

**Checkpoints.** Anything over three minutes is rendered in ~90-second
segments. Each completed segment is recorded in a manifest fingerprinted by
source, target and settings. If the machine reboots at frame 100,000, re-running
the job skips everything already done and resumes at the segment that was in
flight — a crash costs at most ninety seconds of work, not seven hours.
Segments are joined by stream copy, so splitting the work costs no quality.
Changing the source photo, the target or the pipeline invalidates the manifest
rather than silently mixing two different renders into one file.

Rough guide at the speeds this measures (~180–270 ms/frame at 1080p,
~660 ms/frame at 4K):

| Source | Approx. render time |
|---|---|
| 1 h @ 30 fps, 1080p | 5–8 h |
| 1 h @ 60 fps, 1080p | 11–16 h |
| 1 h @ 30 fps, 4K | ~20 h |

Which is exactly why the range picker and the checkpoints exist.

### Quality modes

| Mode | What it does |
|---|---|
| **Auto Max** | **Recommended.** Samples the hard frames of *your* video, benchmarks all nine swappers and every restoration blend on them, and picks the winner by measurement. |
| **Quality** | Fixed high-quality pipeline. Faster, no benchmarking step. |
| **Fast** | One model, no parsing/occlusion masks, no restoration. For previews. |

Auto Max is the default because a fixed preset genuinely loses sometimes. On a
dark test clip the fixed Quality preset scored 0.783 identity -- *worse* than
the old V1 engine's 0.825 -- while Auto Max changed pipeline and reached
**0.906**. No single model is best for every video, which is the whole reason
the competition is run per render.

Results are cached: the key covers the video, the fused identity, every
installed model's SHA256 and the scoring algorithm version, so re-rendering
the same job reuses the verdict (~4 min cold, instant warm) while changing a
model or a metric invalidates it.

Auto Max searches three stages: every installed swapper bare, then restoration
model and blend on the leader, then **mask mode and colour matching** on
whatever is winning. In testing all three sample videos chose a stage-three
variant, so that last stage earns its cost.

Auto Max writes its full report to `data/benchmarks/<job-id>.json`, and the
library records exactly which pipeline won.

### How the score is weighted

Tuned for the single-person case -- "how good does this one face look?", not
"which person is this?":

| Term | Weight | Why |
|---|---|---|
| identity (mean) | 0.30 | does it look like you |
| temporal stability | 0.18 | no flicker or crawl |
| **identity worst-case** | 0.14 | the eye lands on the worst frame, not the average |
| expression | 0.13 | the target keeps their own performance |
| blending | 0.13 | seam, colour continuity, occlusion |
| detail | 0.08 | sharpness and texture |
| identity switches | 0.02 | defensive only -- see below |
| speed | 0.02 | never traded against quality |

`identity_switches` was 0.18 during the multi-person work. On single-person
footage it should never fire, and a weighted term that is always constant is
dead weight in the score -- the same defect as the hard-coded version of it
found earlier. It is demoted rather than deleted, because it still catches a
detector rescue or a reflection pulling the swap off-subject.

`scripts/test_scoring.py` asserts, for every term, that degrading it changes
the metric, in the right direction, and lowers the total. 43 checks.

### Masks

Four modes, and they are genuinely different pipelines (they used to be
aliases, which meant Auto Max could not tell them apart):

| Mode | What it uses |
|---|---|
| `oval` | geometric fallback only |
| `model` | the swapper's own mask + feathered box |
| `parsing` | + BiSeNet face parsing (hair, glasses, hat excluded) |
| `full` | + XSeg occlusion (hands and objects in front of the face) |

### Measured decisions

Two features were built, benchmarked, and then switched off because the
numbers did not support them:

| Feature | Measurement | Decision |
|---|---|---|
| **Pixel boost** (512/768/1024) | Identity fell at every level (0.978 -> 0.959 on a 212 px face); sharpness moved under 2%; render time rose | **Disabled.** Code retained, option withdrawn. |
| **TensorRT** | Provider is listed by onnxruntime but `nvinfer_10.dll` is absent, so sessions silently fall back to CUDA | **Not evaluated.** Reported as such rather than as a bogus 1.00x tie. |

Detector choice was also settled by measurement: YOLOFace reached 1.000 recall
at 8.2 ms against 0.829 for both SCRFD and RetinaFace, so it stays primary,
with SCRFD wired as a recall-only fallback for frames YOLO drops. A fallback
can rescue a missed face but never decides *who* gets swapped -- that stays
with the identity tracker.

### Tracking

"One person" mode locks onto a face by identity, not by size. If another person
moves closer to the camera, gets larger, or crosses in front, the swap stays on
the locked person. If the locked person is not confidently present, **no frame
is swapped** -- a missing swap is a smaller error than the wrong face.

### The models

| File | Size | Job |
|---|---|---|
| `yoloface_8n.onnx` | 12 MB | find faces; box + 5 landmarks |
| `arcface_w600k_r50.onnx` | 174 MB | 512-d identity vector |
| `hyperswap_1a/1b/1c_256.onnx` | 403 MB each | the swap (all three benchmarked) |
| `bisenet_resnet_34.onnx` | 94 MB | face parsing -- hair/glasses/hat masks |
| `xseg_1.onnx` | 70 MB | occlusion -- hands and objects in front of the face |
| `gfpgan_1.4` / `codeformer` / `gpen_bfr_512` / `restoreformer_plus_plus` | 284-377 MB | restoration candidates |

Licensing for every one of these is recorded in [docs/MODELS.md](docs/MODELS.md).
Short version: this stack is **research / personal use only**.

Models are gitignored and fetched on demand:

    python -m app.render.download

Each download is SHA256-recorded in `models/hashes.json` and verified on every
later load.

### Testing

    python scripts/make_testclips.py     # build the synthetic test matrix
    python scripts/v2_testsuite.py       # 80 checks across 12 clips
    python scripts/v1_vs_v2.py           # head-to-head against V1

The tracking clips are composited from two known identities, so "zero identity
switches" is a decidable claim rather than an impression.

## Housekeeping

Uploads and outputs are kept forever. Delete old ones when disk gets tight:

    rm -rf data/uploads/* data/outputs/*

## Notes

- Swap model is `hyperswap_1a_256` rather than `inswapper_128`: higher
  resolution (256 vs 128), and it avoids InsightFace's non-commercial licence.
- There is no content/NSFW classifier in this pipeline. It is a local,
  single-user tool; you are responsible for having the rights to the media you
  feed it, including the consent of anyone appearing in it.
