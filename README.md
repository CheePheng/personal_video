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

Measured with the GPU dedicated to rendering, on the **Balanced** preset that
the app actually runs (HyperSwap 1a + GPEN BFR 512 at 70% + full mask):

| Resolution | ms/frame | Roughly | Renderer VRAM | GPU busy |
|---|---|---|---|---|
| 720p | ~140 | 7 fps | 4.6 GB | 43% |
| 1080p | ~148 | 7 fps | 4.7 GB | 42% |
| 4K | ~212 | 5 fps | 5.2 GB | 29% |

Without the restoration pass (swap only) the same clips run at ~87 / ~97 /
~151 ms/frame using 2.5–3.2 GB. Quote whichever you mean: for a while these
numbers were measured on the swap-only path and reported as "Balanced",
because `render()` quietly fell back to a bare config when none was passed
while the job layer built the real one. Both now come from
`pipeline.apply_preset()`, so there is one definition of each preset.

1080p costs only a little more than 720p: most of the per-frame work happens
on a fixed-size aligned face crop, not on the whole frame, so frame area
matters far less than you would expect.

You can close the browser while it renders; the job keeps going and the page
reconnects to it when you come back.

## Don't render while gaming

Not because of memory — because of compute. The renderer needs about
**4.6 GB at 720p and 5.2 GB at 4K** on Balanced, against roughly 14.7 GB free
on an idle machine. It is nowhere near VRAM-limited. But a game will take the
GPU's compute and power budget, and render times climb accordingly. Render
when you're done playing.

### Measuring VRAM honestly

This section used to claim a render "peaks at 15.6 GB of your 16.3 GB". That
was wrong, and the way it was wrong is worth recording. The VRAM probe reads
`nvidia-smi --query-gpu=memory.used`, which reports **the whole card** — every
process on it. The number was captured with a game (~8.4 GB), a desktop, a
browser and two stale copies of this app's own server (~2.2 GB between them)
all resident, and then attributed entirely to the renderer.

Anything measuring VRAM must therefore report two figures: whole-card usage,
and usage minus the idle floor. `scripts/clean_baseline.py` does exactly that
(`peak_card_mb` and `peak_attrib_mb`) so the two can never be confused again.

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

Rough guide at the speeds this now measures on a dedicated GPU, Balanced
(~148 ms/frame at 1080p, ~212 ms/frame at 4K):

| Source | Approx. render time |
|---|---|
| 1 h @ 30 fps, 1080p | ~4.4 h |
| 1 h @ 60 fps, 1080p | ~8.9 h |
| 1 h @ 30 fps, 4K | ~6.4 h |

These replace earlier figures of 5–8 h / 11–16 h / ~20 h, which were measured
on a contended GPU and before the compositing path was fixed.

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
| **Pixel boost** | On a true A->B swap, identity collapses to 0.03-0.04 at 768 and 1024 -- what two strangers score -- while costing 2.7-4.1x more | **Disabled entirely.** See below. |
| **TensorRT** | Provider is listed by onnxruntime but `nvinfer_10.dll` is absent, so sessions silently fall back to CUDA | **Not evaluated.** Reported as such rather than as a bogus 1.00x tie. |

#### A correction worth reading

Pixel boost was withdrawn, then re-enabled, then withdrawn again. The round
trip is the point.

Boost tiles a high-resolution aligned crop and runs the 256px model on each
tile independently. The first study said it hurt; the second, scored with
SFace (an independently trained recogniser held out of the loop), said 768
and 1024 slightly *helped*, so they were re-enabled.

Both studies ran on the synthetic suite -- which pasted the **source's own
face** into every clip. Swapping a face onto itself leaves a correct face
underneath no matter what the tiling does, so those runs were really scoring
tile seams, and 768's smoother seams read as "better identity".

Re-measured on the A->B set, where the source is a genuinely different person
from the subject (8 clips, 30 samples per level):

| boost | ArcFace (selector) | SFace (holdout) | ms/frame |
|---|---|---|---|
| **256** | **0.6566** | **0.6345** | 83.3 |
| 768 | 0.0361 | 0.0294 | 223.0 |
| 1024 | 0.0160 | −0.0271 | 342.7 |

Two unrelated people score about 0.02 on these recognisers. So at 768 and
above the swap is not producing a worse likeness -- it is producing none.
Each tile receives a fragment with no global facial structure, leaving the
identity conditioning nothing coherent to act on. Both judges agree, by a
margin of 0.6, and the worst frame at 256 beats the best frame at 1024.

`pixel_boost=()` on every swapper. Do not re-enable it on evidence from the
self-swap clips.

Note also the honest ceiling: **0.6566, not 0.97.** Every "0.96 identity"
figure this project used to quote was a face swapped onto itself. 0.65 is
comfortably "same person" for ArcFace and is the real number to improve on.

The holdout judge lives in `app/render/judges.py` and is deliberately **not**
wired into Auto Max -- a judge that participates in selection stops being a
judge. `scripts/test_scoring.py` asserts that.

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
