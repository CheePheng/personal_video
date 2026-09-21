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

### Quality modes

| Mode | What it does |
|---|---|
| **Fast** | One model, no parsing/occlusion masks, no restoration. For previews. |
| **Quality** | Full masks, colour match, temporal smoothing, GPEN restoration. Default. |
| **Auto Max** | Samples the hard frames of *your* video, benchmarks every model and restoration blend on them, and picks the winner by measurement. |

Auto Max writes its full report to `data/benchmarks/<job-id>.json`, and the
library records exactly which pipeline won.

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
