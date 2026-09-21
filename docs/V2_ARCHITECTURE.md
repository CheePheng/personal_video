# V2 Render Pipeline — Architecture

Reference for whoever maintains `app/render/`. Everything below was read out of
the code as it stands; figures are quoted from the source, not remembered.

---

## 1. Overall architecture

### 1.1 Module map

The pipeline lives in `app/render/`. Each module owns exactly one concern and
imports downward only — nothing in the leaf modules imports `pipeline.py`.

| Module | Lines | Owns |
|---|---:|---|
| `types.py` | 129 | Shared dataclasses and enums: `Face`, `SourceIdentity`, `ModelSpec`, `Normalization`, `FrameMetrics`, `PipelineConfig`, and the `RenderError` exception. Deliberately free of heavy imports (no `onnxruntime`, no `cv2` at module scope) so any module can import it without dragging in the CUDA stack. |
| `registry.py` | 218 | Every model the pipeline can run, declared as data. Detectors, recogniser, swappers, enhancers, parsers, plus `EXCLUDED` — models considered and rejected, with the reason. Also `export_registry()`, which writes `models/registry.json` as the download manifest and licence record. |
| `sessions.py` | 141 | ONNX Runtime session construction, caching, VRAM-aware error translation, and the CUDA binding assertion. Also `gpu_info()` / `verify_cuda()` / `free_vram_mb()`. |
| `download.py` | 170 | Resumable, hash-verified model fetch driven by the registry. Records a SHA-256 on first download and verifies against it thereafter. |
| `detection.py` | 143 | Face detection, letterboxing, NMS, adaptive input resolution, and the escalating retry for hard frames. |
| `alignment.py` | 155 | The 5-point landmark templates, the similarity transform onto them, `paste_back()` compositing, rough pose estimation, and `TransformSmoother`. |
| `recognition.py` | 132 | ArcFace embedding, cosine similarity, per-photo quality assessment, and multi-photo identity fusion. |
| `tracking.py` | 279 | `Track`, `FaceTracker` (global best-pair matching) and `TargetLock` (single-person follow with no largest-face fallback). |
| `swapping.py` | 132 | The swap call itself, model-agnostic; input-dict construction by shape inspection; real tiled pixel boost. |
| `masking.py` | 190 | Box / BiSeNet parsing / XSeg occlusion masks, their combination, adaptive feathering, and `MaskSmoother`. |
| `color.py` | 91 | LAB mean/std matching confined to the mask, and `ColorSmoother`. |
| `restoration.py` | 101 | Enhancer invocation, `adaptive_blend()` by face size, soft-oval composite of the restored crop. |
| `metrics.py` | 228 | Every quality measurement, the scene-cut detector, the selection `WEIGHTS`, and `composite_score()`. |
| `benchmark.py` | 322 | Hard-frame sampling, the candidate matrix, per-config scoring, the enhancer identity guard, and the JSON report. |
| `video.py` | 280 | `ffprobe` metadata, raw BGR24 decode/encode pipes, encoder selection, audio mux and `+faststart`, and indexed frame sampling. |
| `pipeline.py` | 330 | `RenderOptions`, `RenderResult`, `load_source()`, the stateful `FrameRenderer`, and `render()` — the only entry point the job layer needs. |

Outside `app/render/`:

| File | Role |
|---|---|
| `app/jobs.py` | SQLite-backed job store and the background `JobRunner` thread. Owns progress mapping, quality-preset → `PipelineConfig` translation, and engine selection (`v2` / `v1`). |
| `app/swapper.py` | The V1 engine, retained as a regression baseline. Still reachable via `engine: "v1"`. |
| `app/main.py` | FastAPI surface: chunked upload, opaque media ids, job creation, library, benchmark report download. |

### 1.2 Per-frame data flow

From the `pipeline.py` docstring and `render()`:

```
decode (raw BGR24 from ffmpeg)
  -> scene-cut check against the previous frame
       (on a cut: reset mask/colour/transform smoothers and tracker motion)
  -> detection.detect_robust()            -> list[Face]
  -> recognition.embed() per face         -> 512-d L2-normalised vectors
  -> tracker decides which face is OUR person
       TargetLock      (single-subject mode)  -> 0 or 1 face
       FaceTracker     (swap_all_faces mode)  -> {track_id: Face}
       ...or nothing at all, and the frame passes through untouched
  -> for each chosen face: FrameRenderer.render_face()
       swapping.swap()            align -> model (tiled if pixel-boosted)
       TransformSmoother          (if opts.temporal)
       masking.build()            box x model x parsing x (1-occluder) x XSeg
       MaskSmoother               (if opts.temporal)
       color.match()              LAB, inside the mask only
       ColorSmoother              then a second match() with smoothed params
       alignment.paste_back()     float composite through the mask
       restoration.restore()      (if an enhancer is configured)
  -> write frame to the encoder's stdin
  -> progress = frames_done / total_frames
```

After the loop: close the encoder stdin, wait (600 s timeout), kill the decoder,
then `video.finalize()` muxes audio back and applies `+faststart`.

Progress is a real count of work completed — `done / total * 100` — not a parsed
console bar. `jobs.JobRunner._overall()` maps a within-stage percentage onto the
whole-render bar using `STAGE_WEIGHTS`, where `processing` carries 0.80.

**Failure isolation.** A `RenderError` raised while rendering one face is caught
in `render()` and swallowed: "one bad face must not abort a whole render." A
failure in the encoder write is not — it is re-raised with the encoder's stderr
attached.

---

## 2. Why V2 replaced V1

V1 is `app/swapper.py`: a single file (515 lines as committed — the "400-line"
figure in the brief is approximate) running four ONNX models directly, with
ffmpeg pipes either side. It works, and it is still selectable as `engine: "v1"`
so the two can be compared on identical inputs.

What V1 does not have:

| Area | V1 | V2 |
|---|---|---|
| Model selection | Four hard-coded model names, each with its template and normalisation written inline as module constants (`ARCFACE_128`, `FFHQ_512`, `DETECT_SIZE`, `SWAP_SIZE`…). | A `ModelSpec` registry; the renderer reads the spec and never names a model. |
| Identity source | One photo, one embedding. | 1–5 photos, quality-weighted fusion with a consensus check. |
| Tracking | `_pick_tracked()` scores `identity + 0.35 * proximity` with a `0.15` floor, and falls back to the largest face when there is no reference yet. No persistent tracks, no prediction, no scale term, no global assignment. | Persistent `Track` objects with velocity prediction, a four-term score, global best-pair matching, and an identity floor that refuses a match outright. |
| Masking | The swapper's own mask only, eroded `5x5 x2` then blurred `11x11`. No semantics, no occlusion handling. | Box + BiSeNet parsing + XSeg occlusion, combined and temporally smoothed, feathered by face size. |
| Colour | None. The swapped face keeps the source photo's lighting. | LAB mean/std match inside the mask, temporally smoothed. |
| Temporal work | None. Every frame is computed independently. | Transform, mask and colour smoothers; scene-cut detection resets all three. |
| Restoration | GFPGAN at a fixed `blend=0.8`, composited through a fixed oval. | Four enhancers, blend as a benchmarked parameter, `adaptive_blend()` by face size, and an identity guard that can reject the enhancer entirely. |
| Measurement | None. | Eleven per-frame metrics, hard-frame sampling, a candidate matrix and a persisted JSON report. |
| Video | `-c:v h264_nvenc -preset p4 -cq 23`, no rotation handling, no VFR awareness, no `+faststart`. | Quality-tiered encoder args, NVENC p7/hq, rotation-aware geometry, audio stream-copy when safe, `+faststart`. |

The pattern across every row is the same: V1 makes one reasonable choice and
commits to it; V2 makes the choice explicit, measurable and replaceable.

> **Correction to the brief.** V1's tracking was not "largest face per frame" —
> `_pick_tracked()` is genuinely identity-driven, and its own docstring argues
> against the largest-face approach. Largest-face is only the bootstrap when
> `ref_embedding is None`. What V1 lacks is *persistence*: no track survives a
> gap, there is no velocity model, and a single reference embedding is compared
> against rather than a maintained per-track identity.

---

## 3. The model registry

`ModelSpec` (in `types.py`) is a frozen, slotted dataclass. It declares
everything that differs between models so the renderer can stay generic:

| Field | Purpose |
|---|---|
| `name`, `filename`, `role` | Identity and category — `swapper`, `detector`, `recognizer`, `parser`, `enhancer`. |
| `input_size` | Native model resolution. |
| `template` | Key into `alignment.TEMPLATES` — which canonical 5-point layout this model was trained against. |
| `normalization` | One of `ZERO_ONE` (`x/255`), `NEG_ONE_ONE` (`(x/255 - 0.5)/0.5`), `ARCFACE` (`(x - 127.5)/127.5`). |
| `license`, `source_url`, `sha256`, `expected_bytes` | Provenance and integrity. |
| `needs_embedding` | Whether this swapper is identity-conditioned at all. `swapping.swap()` refuses a model without it. |
| `embedding_normalized` | Whether the identity vector must be L2-normalised before it is fed in. Consumed by `prepare_embedding()`. |
| `outputs_mask` | Whether the model emits its own mask as a second output. |
| `pixel_boost` | The tuple of resolutions this model can be tiled to — `(256, 512, 768, 1024)` for the hyperswap family. |
| `providers` | Execution provider preference order. |
| `notes` | Human-readable rationale. |

### 3.1 How this keeps model names out of the renderer

Three concrete places:

* `swapping._to_blob()` / `_from_blob()` branch on `spec.normalization`, never on
  `spec.name`.
* `swapping._feed()` builds the ONNX input dict by inspecting
  `sess.get_inputs()` and matching **by shape**: a 4-D input gets the image
  blob, a 2-D input gets the embedding. Input names differ between families
  (`source`/`target` vs `embedding`/`img`), so names are never assumed. An input
  of any other rank raises `RenderError` rather than guessing.
* `detection._supports_dynamic_input()` reads the ONNX input shape to decide
  whether the detector can run at a non-native resolution at all — `yoloface_8n`
  is compiled to a fixed 640×640, so asking for 1280 would be an error, not an
  optimisation.

Adding a swapper is therefore a registry entry plus a download, not a code
change.

### 3.2 Registered models

| Role | Models |
|---|---|
| Detector | `yoloface_8n` (default), `scrfd_2.5g`, `retinaface_10g` |
| Recogniser | `arcface_w600k_r50` — 512-d, used for source identity, tracking *and* benchmark scoring |
| Swapper | `hyperswap_1a_256`, `hyperswap_1b_256`, `hyperswap_1c_256` |
| Enhancer | `gfpgan_1.4`, `codeformer`, `gpen_bfr_512`, `restoreformer_plus_plus` |
| Parser | `bisenet_resnet_34`, `xseg_1` |

### 3.3 Exclusions are data, not omissions

`EXCLUDED` records every model considered and why it was rejected, so an audit
can see the reasoning: `uniface_256` (no upstream licence file, and takes a
source *image* rather than an embedding — verified by inspecting the ONNX
inputs — so it cannot use the fused identity); `inswapper_128` (non-commercial
research weights, upstream objection to redistribution, and only 128 px);
`alphaface_256` (MIT repo but non-commercial weights — an unresolved
contradiction); `simswap_*`, `ghost_*`, `hififace_unofficial_256` (need an
`arcface_converter_*` remapping matrix and/or have unclear provenance); and
`nsfw_1/2/3`, excluded on scope grounds — see §13.

`commercial_safe(name)` is a helper that returns true only when the licence
string mentions MIT or Apache and does *not* mention non-commercial. On the
current registry that is `restoreformer_plus_plus` alone among generative
models.

### 3.4 Known gap

`masking.py` does **not** read `spec.normalization` for its two parsers — it
hard-codes ImageNet statistics for BiSeNet and NHWC `/255` for XSeg. The
registry currently declares `bisenet_resnet_34` as `NEG_ONE_ONE`, which the code
correctly ignores. `ModelSpec` also has no tensor-layout field, so XSeg's NHWC
convention lives only in `_prep_xseg()`. This is the one place the "never
special-case a model" rule is not honoured; see §6.

---

## 4. Identity: multi-photo fusion

`pipeline.load_source()` accepts up to five photo paths (`paths[:5]`). Each is
read, run through `detect_robust(img, 0.4, "quality")`, and the largest detected
face is kept. Unreadable files and photos with no face are collected into a
`problems` list; if *nothing* survives, the raised `RenderError` names each
failure individually.

`recognition.build_source_identity()` then does the fusion.

### 4.1 Quality weighting

`assess_source_face()` measures four signals per photo:

| Signal | Source |
|---|---|
| `size` | `Face.size` — geometric mean of box width and height, robust to aspect |
| `detector_score` | The detector's own confidence |
| `blur` | Laplacian variance of the 112 px aligned crop; higher is sharper |
| `yaw`, `roll` | `alignment.pose_from_kps()` — nose offset from the eye midpoint as a fraction of eye distance, scaled to degrees |

These become a single weight:

```python
w_size = min(1.0, q["size"] / 160.0)
w_conf = clip(q["detector_score"], 0.0, 1.0)
w_blur = min(1.0, q["blur"] / 120.0)
w_pose = exp(-abs(q["yaw"]) / 45.0)
weight = max(1e-3, w_size * w_conf * w_blur * (0.4 + 0.6 * w_pose))
```

The factors **multiply**, so any single disqualifier — a tiny face, a mush of
motion blur, an extreme profile — suppresses the whole weight. Pose is the
exception: it enters as `0.4 + 0.6 * w_pose`, so even a full profile retains 40%
of its pose factor rather than being zeroed. That is deliberate: an off-angle
photo is exactly what helps the swap hold up when the target turns their head.

A plain mean is not used because one bad photo would drag the identity.

### 4.2 The consensus check

After weighting, every surviving vector is compared against the **unweighted**
mean direction:

```python
mean = normalise(np.mean([e["vec"] for e in entries], axis=0))
for e in entries:
    e["consensus"] = similarity(e["vec"], mean)

if len(entries) >= 3:
    agree = [e for e in entries if e["consensus"] >= 0.35]
    if agree:
        entries = agree
```

A photo that disagrees sharply with the consensus is almost always a different
person — a group shot where the wrong face was the largest. Averaging it in
produces a face resembling neither.

Two guards on the guard: it only runs with **three or more** photos (with two,
there is no majority to be the consensus), and if filtering would leave nothing
it is abandoned rather than applied.

The fused vector is the weighted sum, renormalised to unit length, returned as
`SourceIdentity(embedding=(1,512), per_image=[...], n_images=N)`. `per_image`
carries the rounded quality figures and each photo's consensus score, and is
copied into the benchmark report so a later reviewer can see which photos
actually drove the identity.

---

## 5. Tracking

Two classes in `tracking.py`, sharing the same matcher.

### 5.1 The combined score

`FaceTracker._score(track, face, emb)`:

| Term | Weight | Computation |
|---|---:|---|
| identity | **0.60** | `dot(track.embedding, face_embedding)` — both L2-normalised, so this is cosine |
| IoU | **0.18** | Overlap of the detection box with a box centred on the track's *predicted* centre, carrying the track's current dimensions |
| motion | **0.14** | `exp(-dist * 8.0)`, where `dist` is the centre-to-prediction distance normalised by the frame diagonal |
| scale | **0.08** | `exp(-abs(log(face.size / track.size)) * 1.5)` — a face does not triple in size in one frame |

Identity dominates because it is the only signal that actually knows *who*
someone is. Geometry is there to disambiguate similar-looking candidates and to
carry a track through a brief identity dip (a blink, a turn, a frame of blur).

Prediction is constant-velocity: `predict()` returns `centre + velocity`, and
`velocity` is itself an EMA, `0.5 * old + 0.5 * (new_centre - old_centre)`.
The track's identity is a much slower EMA — `0.9 * old + 0.1 * new`, renormalised
— so it adapts to lighting and pose drift without letting one bad frame rewrite
who the track is.

### 5.2 Global best-pair matching

Greedy per-track matching lets two crossing faces both claim the same detection.
Instead, every `(track, detection)` pair is scored, the list is sorted
descending, and pairs are consumed in order with `used_t` / `used_f` sets
preventing reuse. A pair scoring below `MATCH_FLOOR = 0.30` is never consumed.

Unmatched tracks get `mark_missed()` (miss counter up, velocity decayed ×0.8 so a
long-lost track does not drift across frame). Unmatched detections with a valid
embedding become new tracks. Tracks exceeding `MAX_MISSES = 45` frames are
retired.

### 5.3 The `IDENTITY_FLOOR` rule

`IDENTITY_FLOOR = 0.28`. It is applied as a veto *inside* the matching loop,
after a pair has already won on combined score:

```python
if emb is not None:
    ident = float(np.dot(track.embedding.ravel(), emb.ravel()))
    if ident < self.identity_floor:
        self.stats["skipped_low_conf"] += 1
        continue
```

So a pair with excellent geometry and wrong identity is dropped, not accepted.
This is what stops the swap jumping to whoever is nearest the camera. The count
is reported as `low_confidence_rejections`.

`RenderOptions.identity_floor` defaults to the same `0.28` and is threaded
through to both `FaceTracker` and `TargetLock`.

### 5.4 `TargetLock` — swap nobody rather than the wrong person

This is the governing principle of the whole tracking layer, and `TargetLock`
enforces it in three separate places:

1. **No largest-face fallback.** The class docstring states it outright: if the
   locked identity is not confidently present, it returns `None` and the frame
   passes through untouched. Largest-face is used *only* for the very first
   lock (`face.area * max(face.score, 0.01)`), when there is nothing yet to be
   wrong about.
2. **Drift detection.** Even when the locked track *is* matched this frame, the
   track's current EMA identity is compared against the embedding originally
   locked. If that cosine falls below the floor, the track has quietly walked
   onto someone else: `identity_switches += 1`, the lock is released, the frame
   is skipped.
3. **Re-acquisition by identity alone.** When the locked track is missing, every
   assigned track is compared against `locked_embedding`, with `best_sim`
   *initialised to the identity floor* — so a candidate must beat the floor, not
   merely beat the others. If none does, the frame is skipped.

`report()` returns `frames_swapped`, `frames_skipped`, `reacquisitions`,
`identity_switches`, `tracks_created`, `tracks_lost`,
`low_confidence_rejections`. These land in `RenderResult.tracking` and are
surfaced through the library API.

`reset_motion()` is called on a scene cut: positions and miss counts no longer
carry over across a hard cut, but identity does.

---

## 6. Masking

A crude oval is what makes a swap look pasted-on. `masking.build()` combines
three sources, each answering a different question.

| Source | Question it answers | Why it alone is insufficient |
|---|---|---|
| **Box mask** — `box_mask(size, padding=0.06, blur=0.10)` | Where is the crop border? | Knows nothing about the face; a rectangle would cut through hair and neck. |
| **Parsing mask** — BiSeNet, 19 CelebAMask-HQ classes | Which pixels are *face*, as opposed to hair, glasses, hat, neck, clothing? | Has no concept of things that are not part of the person at all. |
| **Occlusion mask** — XSeg | Is something in front of the face — a hand, a microphone, an object crossing frame? | Does not distinguish face from hair; would happily replace a forehead through a fringe. |

Plus the swapper's own mask, when `spec.outputs_mask` and the model returned a
second output.

Class partition used from BiSeNet:

```python
FACE_CLASSES     = (1, 2, 3, 4, 5, 10, 11, 12, 13)   # skin, brows, eyes, nose, mouth, lips
OCCLUDER_CLASSES = (6, 9, 15, 16, 17, 18)            # glasses, earring, necklace, cloth, hair, hat
```

Combination, in order:

```python
mask = box_mask(size, padding)
mask = mask * fit(model_mask)                        # if the swapper emitted one
mask = mask * fit(face) * (1.0 - fit(occluder))      # BiSeNet
mask = mask * fit(occlusion_mask(...))               # XSeg
if not used["model"] and not used["parsing"]:
    mask = mask * oval_mask(size)                    # last-resort fallback
```

The union of what to exclude is intersected with what to include, so a hand over
the cheek leaves the original pixels untouched and the replacement face appears
genuinely *behind* the occluder.

Both parser calls are wrapped in `try/except RenderError` and degrade silently:
if a parser model is missing, the remaining sources still apply, and the `used`
dict records exactly which sources contributed. That dict is propagated to
`RenderResult.mask_sources`.

### 6.1 Adaptive feathering

```python
k = int(np.clip(face_size * 0.06, 3, 41))
mask = cv2.GaussianBlur(mask, (k | 1, k | 1), 0)
```

6% of the face's on-screen size, clamped to 3–41 px and forced odd. A 40 px face
and a 600 px close-up must not get the same absolute blur, or the small one
dissolves and the large one shows a hard edge.

Optional `erode` / `dilate` passes run before the feather.

### 6.2 Temporal smoothing

`MaskSmoother` is a straight EMA in mask space with `alpha = 0.55`. A mask that
changes shape every frame makes the seam crawl even when each individual frame
looks fine in isolation. Persistent shape changes (a hand arriving) come through
within a few frames; per-frame parser noise averages out.

`jitter()` returns the mean absolute change against the previous mask — a
flicker measure. It is read *before* smoothing is applied, stored as
`FrameRenderer.last_mask_jitter`, and consumed both by the benchmark
(`mask_jitter`) and by the full render (`tracking.mask_jitter_mean`).

`reset()` on a scene cut: carrying a mask across a cut is simply wrong.

### 6.3 Two silent-wrong-result preprocessing bugs

Both were verified from the ONNX graphs, and both would produce a plausible
result rather than an error — which is what makes them dangerous.

**BiSeNet needs ImageNet normalisation, not `[-1,1]`.**

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _prep_parser(crop):
    blob = crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return ((blob - IMAGENET_MEAN) / IMAGENET_STD)[None]      # NCHW
```

It was trained with ImageNet statistics, not the `[-1,1]` convention the
swappers use. Feed it the wrong normalisation and the segmentation is subtly
wrong while still having roughly the right shape — the mask looks fine, class
boundaries are off, and the hairline is quietly incorrect for the whole render.

**XSeg is NHWC, not NCHW.**

```python
def _prep_xseg(crop):
    return (crop[:, :, ::-1].astype(np.float32) / 255.0)[None]   # NHWC, no transpose
```

XSeg keeps channels last, unlike every other model in the registry. Its output
is `(N, 256, 256, 1)`, so `occlusion_mask()` uses `np.squeeze()` rather than
indexing a channel dimension that is not where NCHW code would expect it. Get
the layout wrong and the model still runs and still returns a float array in
`[0,1]` — it is simply garbage.

Neither convention is expressible in `ModelSpec` today (there is no layout
field, and `normalization` has no ImageNet member), which is why both are
hard-coded in `masking.py`. If a third parser is ever added, this is the place
that will need generalising.

---

## 7. Colour

A swapped face carries the source photo's lighting and white balance. Dropped
into a differently-lit shot it reads as a sticker.

`color.match()` corrects it in LAB space, matching the mean and standard
deviation of the swapped crop to the target crop it replaces.

### 7.1 Inside the mask only

`_masked_stats()` computes weighted mean and variance with the mask as the
weight, and bails to identity statistics if the mask sums to less than 16
pixels. Background pixels inside the crop therefore cannot drag the correction.

The correction is then applied through the same mask:

```python
m = mask[:, :, None] if mask.ndim == 2 else mask
return swapped * (1.0 - m) + corrected * m, params
```

so only the face is ever altered — the rest of the frame is untouched by
construction.

### 7.2 Asymmetric strength

```python
L_STRENGTH  = 1.0
AB_STRENGTH = 0.65
MAX_SHIFT   = 28.0

gain  = clip(t_std / max(s_std, 1e-3), 0.6, 1.6)
shift = clip(t_mean - s_mean * gain, -MAX_SHIFT, MAX_SHIFT)
strength = [L_STRENGTH, AB_STRENGTH, AB_STRENGTH]
gain  = 1.0 + (gain - 1.0) * strength
shift = shift * strength
```

L carries luminance — the dominant cue, and the one that must follow the scene's
lighting, so it is corrected at full strength. a/b carry chroma, corrected at
0.65 because pushing them hard recolours the person and starts eroding the
identity that was just swapped in. `MAX_SHIFT = 28.0` caps a correction that is
almost certainly fighting a bad mask rather than genuine lighting.

Note the strength is applied to the *deviation from neutral* (`1.0 + (gain - 1)
* strength`), not to the gain itself, so a strength of 0 is a true no-op.

### 7.3 Temporal smoothing

`ColorSmoother` is an EMA over the correction **parameters** (`alpha = 0.8`),
not over pixels. Recomputing per frame makes the face pulse as the target's
exposure wobbles.

`FrameRenderer.render_face()` calls `match()` twice when temporal smoothing is
on: once to derive this frame's parameters, then again with the smoothed
parameters fed back in. `match()` accepts a `params` dict precisely so the
second call skips statistics entirely and just applies the transform.

`reset()` on a scene cut.

---

## 8. Restoration

The swappers output 256 px. On a close-up that is an upscale by the time it is
pasted back, which reads as soft. Restorers re-synthesise the face at 512 px and
recover convincing detail.

### 8.1 Why blend is benchmarked rather than fixed

A restorer is a generative model with its own idea of what a face looks like. At
full strength it pulls the result toward that prior and *away* from the identity
just swapped in. Sharper and less like the person is not an improvement.

So blend is a first-class parameter — `restoration.BLEND_CANDIDATES = (0.0, 0.2,
0.4, 0.6, 0.8, 1.0)` — the benchmark measures identity at each blend level it
tests, and configurations that trade too much identity for sharpness are
rejected outright.

For CodeFormer's fidelity weight, `restore()` passes `0.85` (0 = prettier, 1 =
truer to input): identity is the priority, so it asks for fidelity.

### 8.2 `adaptive_blend()` by face size

```python
if face_size < 64:   return base_blend * 0.35
if face_size < 128:  return base_blend * 0.65
if face_size > 400:  return min(1.0, base_blend * 1.15)
return base_blend
```

Small faces get less because there is little real detail to recover, so a strong
restorer is mostly *inventing* texture — and invented texture differs frame to
frame, which is visible as shimmer even when each frame looks fine alone. Large
faces get a modest boost, clamped at 1.0.

After blending, the restored crop is composited through a soft blurred ellipse
(axes 0.44/0.54 of the crop, Gaussian 31×31) so the restored square never shows
an edge.

### 8.3 The identity guard

In `benchmark._identity_guard()`:

```python
def _identity_guard(bare, enhanced, max_loss=0.04) -> bool:
    b = bare["aggregate"].get("identity_mean")
    e = enhanced["aggregate"].get("identity_mean")
    if b is None or e is None:
        return True
    return (b - e) <= max_loss
```

Every enhanced candidate is compared against the **best bare** candidate. If the
enhancer costs more than **0.04 cosine** of mean identity, the candidate is
removed from the eligible set and annotated
`"rejected": "identity loss exceeded threshold vs bare swapper"` in the report —
it stays visible, with its score, so the trade-off is on the record rather than
hidden. A candidate whose identity could not be measured at all is given the
benefit of the doubt (`return True`).

---

## 9. Pixel boost

Pixel boost here is real tiling, not upscaling.

The swappers are 256 px natively. Naively upscaling their output just blurs.
Instead `swapping.swap()`:

1. Aligns the face at the **higher** resolution — `alignment.warp(frame, kps,
   spec.template, size)` where `size` is e.g. 512 or 1024, so the crop carries
   genuine frame detail.
2. Splits that crop into an `n × n` grid of exactly 256 px tiles, where
   `n = size // base`.
3. Runs the swapper **per tile**, each at its native 256 px.
4. Reassembles both the face and (if the model emits one) the mask into the
   full-resolution buffer.

```python
n = size // base
for r in range(n):
    for c in range(n):
        y, x = r * base, c * base
        t_face, t_mask = _run_tile(spec, crop[y:y+base, x:x+base], embedding)
        face[y:y+base, x:x+base] = t_face
```

Each tile therefore carries genuine model detail at full resolution. The cost is
`(boost / 256)²` model calls per face — 16× at 1024 — which is why it is a
benchmarked option rather than a default.

A requested boost is validated against `spec.pixel_boost`; an unsupported value
raises `RenderError` naming the supported set rather than silently degrading. A
boost less than or equal to the native size is ignored.

---

## 10. The benchmark system

`benchmark.run()` is AUTO mode. There is no universally best swapper: a model
that nails one face can be mediocre on another, and a restorer that sharpens
beautifully may quietly erase the identity. So AUTO does not pick by reputation
— it renders candidates on frames from *this* video and picks from the numbers.

### 10.1 Hard-frame sampling by trait

Scoring only easy frontal frames rewards the wrong pipeline.

`select_frames(target_path, info, want=14, scan=60)`:

1. Scan up to 60 indices spread evenly with `np.linspace` across the whole
   video, pulled in **one sequential decode** by `video.sample_frames()` —
   repeatedly seeking a long-GOP H.264 file is slower and can land on the wrong
   frame.
2. Detect in each; frames with no face are dropped. If nothing has a face
   anywhere, raise a `RenderError` that says so in plain terms.
3. Catalogue each surviving frame's traits via `_frame_traits()`: `n_faces`,
   `face_size`, `blur` (Laplacian variance inside the face box), `yaw`,
   `brightness` (frame grey mean), `score`.
4. Take the two most extreme frames for each of seven categories:

| Category | Key |
|---|---|
| Profile | max `yaw` |
| Motion blur | min `blur` |
| Close-up | max `face_size` |
| Small face | min `face_size` |
| Dark | min `brightness` |
| Bright | max `brightness` |
| Multiple people | max `n_faces` |

5. Fill the remainder up to `want = 14` with an even spread, so easy frames are
   represented too.

Because the picks go into a dict keyed by frame index, a frame that is extreme
in two categories is counted once.

### 10.2 The candidate matrix

`candidate_configs(available=None, thorough=True)`:

* Every registered swapper whose `.onnx` file is actually present on disk gets a
  **bare** config (`enhancer=None`, `mask="model"`). Bare configs isolate
  identity from restoration.
* Then, for the first swapper only, each present enhancer is paired with blends
  **0.4** and **0.7**.

This is deliberately *not* a full cross product — that would take far too long.
The bare pass finds the best swapper; the enhancer pass explores restoration
against a fixed swapper; the identity guard then decides whether restoration is
worth it at all.

If no swap model is installed, `RenderError` names the download command.

### 10.3 Every metric measured

Per frame, in `score_config()`. Note that identity is measured on a **re-detected
face in the output frame**, not at the input face's position — identity must be
measured on the face that actually ended up in the frame.

| Metric | Definition | Direction |
|---|---|---|
| `identity` | Cosine between the fused source embedding and the rendered face's embedding | higher better |
| `identity_stability` | Cosine between consecutive rendered faces — identity flicker | higher better |
| `expression_delta` | Landmark change with pose removed (centred and scale-normalised first, so a face that merely moved or got closer does not register) | lower better |
| `landmark_delta` | Raw geometry drift, normalised by face size | lower better |
| `seam` | Gradient magnitude in a band around the mask boundary divided by gradient magnitude in the mask interior | lower better |
| `color_discontinuity` | LAB distance between mean skin tone just inside and just outside the boundary | lower better |
| `sharpness` | Laplacian variance inside the face box | higher better |
| `texture_retention` | High-frequency energy of the render over that of the original face; ~1.0 means comparable detail, far below means plastic, far above can mean an enhancer inventing texture | ~1.0 best |
| `mask_jitter` | Mean absolute mask change vs the previous frame | lower better |
| `flow_flicker` | Farnebäck optical flow warps the previous output onto the current one; whatever difference remains is not motion, it is the render changing its mind | lower better |
| `ms` | Wall time for `render_face()` | lower better |

Plus per config: `frames_measured`, `frames_failed`, `vram_peak_mb` (sampled
from `nvidia-smi` after each frame), `wall_s`.

`seam_score()` deserves a note: the ratio is returned **raw, not clamped at
zero**. A clean blend sits below 1.0 (the boundary is smoother than the face
interior); a visible seam pushes it above 1.0. Clamping would collapse every
good blend to 0.0 and leave the metric unable to rank the candidates it exists
to separate. The `composite_score` normalisation band for `seam_mean` is
`0.55 → 1.45`, inverted, which reflects that.

### 10.4 The weights

`metrics.WEIGHTS`, in priority order — identity first, speed last:

| Component | Weight | Normalisation band in `composite_score()` |
|---|---:|---|
| `identity` | **0.34** | `identity_mean` 0.20 → 0.65 (0.2 is a poor swap, 0.65 excellent) |
| `identity_switches` | **0.18** | binary: 1.0 if none, else 0.0 |
| `temporal` | **0.16** | `identity_stability_mean` 0.80→0.99 ×0.5, `flow_flicker_mean` 2.0→14.0 inverted ×0.3, `mask_jitter_mean` 0.01→0.12 inverted ×0.2 |
| `blending` | **0.12** | `seam_mean` 0.55→1.45 inverted ×0.6, `color_discontinuity_mean` 2.0→22.0 inverted ×0.4 |
| `expression` | **0.10** | `expression_delta_mean` 0.008→0.075 inverted |
| `detail` | **0.07** | `sharpness_mean` 60→420 ×0.5, `texture_retention_mean` 0.45→1.25 ×0.5 |
| `speed` | **0.03** | `ms_per_frame` 60→1400 inverted |

Identity and identity stability together carry 0.52 — more than everything else
combined. Speed carries 0.03: it is a tie-breaker, not a criterion.

`identity_switches` is binary rather than graded because a single wrong-person
frame is a categorically different kind of failure from a slightly soft one.

> **Known gap.** `score_config()` writes `"identity_switches": 0` unconditionally
> into its aggregate — the benchmark renders a single largest face per sample
> frame and never runs a `TargetLock`, so no switch can be observed. That
> component therefore awards its full 0.18 to every candidate and currently
> discriminates between none of them. The metric *is* measured for real during a
> full render (`TargetLock.identity_switches`, surfaced via
> `RenderResult.tracking`); it is only the benchmark that cannot see it.

### 10.5 Why individual metrics are preserved

`composite_score()` returns `(score, contributions)` — the weighted contribution
of every component, not just the total. Collapsing to one number early would
hide *why* a pipeline won, and the trade-offs (sharper but less like the person)
are the whole point.

The report written to `data/benchmarks/{job_id}.json` carries:

* the video's probe data,
* `source_identity` — how many photos were used and each one's quality figures,
* `sample_frames` with the traits that got each one selected,
* `weights`,
* **every** candidate sorted by score, each with its full per-frame rows, its
  aggregate, its contributions, and any rejection reason,
* the winner,
* live GPU facts.

`app/main.py` exposes it at `GET /api/library/{jid}/benchmark`, so a decision
made months ago can still be explained rather than asserted.

### 10.6 Selection

```
ok        = candidates that scored >= 0
bare      = ok candidates with no enhancer
best_bare = max(bare, key=score)
eligible  = ok, minus enhanced candidates failing _identity_guard(best_bare, r)
winner    = max(eligible or ok, key=score)
```

If every candidate failed, a `RenderError` quotes the first error rather than
falling back to a guess.

---

## 11. Video I/O

Frames move over raw pipes as **BGR24** — no JPEG or PNG round-trip anywhere, so
the only generation loss in the whole pipeline is the single final encode.

### 11.1 Rotation metadata

Phone footage is often stored landscape with a 90° display matrix. Decoding raw
ignores that, so a portrait video would render sideways.

`probe()` reads the rotation from `side_data_list` first (the modern display
matrix), falling back to the legacy `tags.rotate`, normalised `% 360`. When the
rotation is 90 or 270, the reported `width`/`height` are **swapped**, because the
decoded frames come out transposed — and `video.frames()` reshapes the pipe
buffer using exactly those numbers, so getting this wrong would desynchronise
the entire raw stream, not merely rotate the picture.

> **Note.** `decode()` relies on ffmpeg's *default* autorotate behaviour to apply
> the rotation; its comment says the intent is stated explicitly, but no
> `-autorotate` flag is actually passed. The behaviour is correct on current
> ffmpeg; the safeguard the comment describes is not in the command.

### 11.2 VFR detection

```python
r_fps   = rate("r_frame_rate")      # container's nominal rate
avg_fps = rate("avg_frame_rate")    # actual average over the file
fps     = avg_fps or r_fps or 25.0
is_vfr  = bool(r_fps and avg_fps and abs(r_fps - avg_fps) / max(r_fps, 1e-6) > 0.01)
```

A gap of more than 1% between nominal and average rate means variable timing.
Re-encoding VFR at a fixed rate silently drifts audio out of sync over minutes.

Note that `fps` prefers `avg_frame_rate`, which is the honest number for a VFR
file and is what the encoder is given.

> **Known gap.** `is_vfr` is computed, carried on `VideoInfo` and serialised into
> the render record, but it is **not consumed anywhere**. `open_encoder()`
> always writes at a fixed `-r {info.fps}`. The module docstring's claim that we
> "hand ffmpeg the original timing instead of pretending it is CFR" describes an
> intent, not the current code. The `-shortest` flag on the mux limits how far
> drift can propagate, but VFR input is still resampled to CFR.

### 11.3 Encoding

`encoder_args(quality)` probes `ffmpeg -encoders` for `h264_nvenc` at call time
rather than assuming.

| Quality | NVENC available | Fallback |
|---|---|---|
| `quality` | `h264_nvenc -preset p7 -tune hq -rc vbr -cq 19 -b:v 0 -spatial-aq 1 -temporal-aq 1 -rc-lookahead 32 -bf 3 -profile:v high` | `libx264 -preset slow -crf 17 -profile:v high` |
| `fast` | `h264_nvenc -preset p4 -rc vbr -cq 25 -b:v 0` | `libx264 -preset veryfast -crf 23` |

p7/hq is NVENC's quality-oriented setting, not its fast default. NVENC keeps the
encode off the CPU while the GPU is already busy with inference.

Output is always `-pix_fmt yuv420p` — the only chroma format every browser
decodes reliably.

Every ffmpeg invocation in the module passes arguments as a **list**; `shell=True`
is used nowhere.

### 11.4 Audio and `+faststart`

`finalize()`:

* If the source has audio, it is **stream-copied** when the codec is already
  `aac` or `mp3` — lossless and faster — and re-encoded to `aac -b:a 192k`
  otherwise. `-shortest` guards against a length mismatch.
* `-movflags +faststart` moves the index to the front so a browser can seek
  before the whole file has arrived.
* Audio is treated as a bonus, never as a reason to lose a render: if the mux
  fails, the reason is recorded in the result dict and the silent video is
  re-muxed for faststart alone; if *that* fails too, the silent file is simply
  renamed into place.

The result dict (`{"audio": ..., "faststart": ...}`) is merged into
`RenderResult.video`, so the record says what actually happened.

---

## 12. GPU

### 12.1 `session.get_providers()`, not `get_available_providers()`

This is the single most dangerous failure mode in the stack, and it is
documented in `docs/FINDINGS.md` from a real measurement on this machine:

```
available providers : [..., 'CUDAExecutionProvider', 'CPUExecutionProvider']
session providers   : ['CPUExecutionProvider']     <-- silent fallback
```

ONNX Runtime happily **lists** CUDA in `get_available_providers()` and then
silently runs on CPU when its provider DLLs cannot load. Nothing raises.
Everything looks fine. A two-minute render takes hours.

`sessions._make()` therefore asserts on the session's own providers:

```python
active = sess.get_providers()
if not allow_cpu and (not active or active[0] != "CUDAExecutionProvider"):
    raise RenderError(
        f"CUDA provider did not bind for '{spec.name}' -- session is using "
        f"{active}. Refusing to run on CPU silently; a render would take "
        f"hours. Check the CUDA 13 runtime (see docs/FINDINGS.md).")
```

Note it checks `active[0]`, not membership: CUDA must be the *preferred*
provider, not merely present in the list.

`gpu_info()` still reports `ort.get_available_providers()` — but as a diagnostic
field, alongside live `nvidia-smi` figures (GPU name, total/used VRAM,
utilisation). It is never used as a decision input.

### 12.2 CPU fallback is a hard failure

Two layers enforce it:

* `sessions.get(spec, allow_cpu=False)` — the default. With `allow_cpu=False`,
  `CPUExecutionProvider` is **stripped from the provider list** before the
  session is built, so ORT cannot fall back even if it wanted to.
* `pipeline.render()` calls `sessions.verify_cuda(get_model(cfg.swapper))`
  *before* committing to a long render, and raises if `cuda_active` is false,
  naming the actual session providers.

The `allow_cpu` parameter exists but nothing in the render path passes `True`.

### 12.3 VRAM

`sessions.run()` translates an allocator failure into something actionable
rather than leaking a CUDA error string. It matches on `"out of memory"`,
`"cudnn_status_alloc_failed"` and `"cublas"`, and raises:

> insufficient VRAM running '{model}'. Close other GPU programs
> (games/browsers) and retry, or use a lower quality mode.

Sessions are cached in a lock-guarded dict because building one costs hundreds
of milliseconds and allocates VRAM; doing it per frame would dominate the render
and thrash the GPU. `release(name=None)` drops one or all of them to free VRAM.

---

## 13. Scope: no content classification, by design

**There is no content or NSFW classifier anywhere in the render path.** This is
a design decision, stated in three places in the code and enforced by the
registry.

`registry.py` module docstring:

> Deliberately absent: FaceFusion's nsfw_1/2/3 classifiers and its content
> analyser. This project does not classify the subject matter of user media.

`EXCLUDED["nsfw_1/2/3"]`:

> Content classifiers. Explicitly out of scope: this project does not classify
> the subject matter of user media.

`export_registry()` writes the same statement into `models/registry.json`, so
the generated manifest carries it too.

Consequently no such model is registered, none is downloadable through
`download.py`, and no code path invokes one.

### 13.1 The only reasons the pipeline fails

`RenderError` is documented in `types.py` as being raised **only** for genuine
technical faults, never for the subject matter of the media:

| Failure | Raised from |
|---|---|
| Corrupt or unreadable media | `video.probe()` — `"unreadable or corrupt media: {ffprobe stderr}"` |
| No video stream | `video.probe()` |
| Unsupported codec | `pipeline.render()` — `"no frames could be decoded from the target video (codec {codec})"` |
| Missing model file | `sessions._make()` — names the path and the download command |
| Missing ffmpeg/ffprobe | `video._tools_present()` — names the install command |
| No detectable face (source) | `pipeline.load_source()` — lists each photo's specific problem |
| No detectable face (target) | `benchmark.select_frames()` |
| CUDA / model init failure | `sessions._make()`, `pipeline.render()` |
| Insufficient VRAM | `sessions.run()` |
| Encoding failure | `pipeline.render()` — `"encoder failed"` / `"encoding produced no output"`, with the encoder's stderr |
| Disk error | Surfaces as the underlying `OSError` through the job layer |
| Degenerate landmarks | `alignment.estimate_matrix()` |
| Unknown model / unsupported pixel boost / non-identity-conditioned swapper | `registry.get_model()`, `swapping.swap()` |

`jobs.JobRunner._fail()` distinguishes the two cases explicitly: a `RenderError`
is an *explained technical fault* and is shown verbatim to the user (truncated
at 4000 chars); anything else is an unexpected exception and gets a type name
plus the tail of a traceback.

### 13.2 FaceFusion

FaceFusion is **not a runtime dependency**. It is not imported, not invoked, and
not installed by this project. V1's module docstring states it in its first
line: *"ONNX Runtime + OpenCV + ffmpeg. No FaceFusion."*

Its source was read as a **technical reference** for preprocessing conventions —
the 5-point templates, which normalisation each model family expects, which
models need an `arcface_converter_*` remapping matrix. Those facts are recorded
as data in `ModelSpec` fields and verified against the ONNX graphs, not copied
as code.

One factual qualification, so the record is accurate: `registry.ASSETS` points
at `github.com/facefusion/facefusion-assets/releases/download`, so the
**model weights are fetched from FaceFusion's asset releases**. That is a
download source for third-party ONNX files, not a code dependency — nothing from
FaceFusion runs in this process — and each model's licence and provenance is
recorded independently in its `ModelSpec`.

---

## 14. Configuration reference

### 14.1 `RenderOptions`

| Field | Default | Effect |
|---|---|---|
| `quality` | `"quality"` | `fast` \| `quality` \| `auto`. Selects the encoder tier, the detector resolution policy, and (in `auto`) whether to benchmark. |
| `swap_all_faces` | `False` | `False` → `TargetLock` (one person). `True` → `FaceTracker` (every tracked face). |
| `config` | `None` | An explicit `PipelineConfig`; AUTO fills it from the benchmark winner. |
| `detect_threshold` | `0.5` | Detector confidence floor. |
| `identity_floor` | `0.28` | Passed to the tracker; see §5.3. |
| `use_parsing` | `True` | Enable BiSeNet. |
| `use_occlusion` | `True` | Enable XSeg. |
| `color_match` | `True` | Enable LAB matching. |
| `temporal` | `True` | Enable all three smoothers. |
| `max_frames` | `None` | Benchmarking and tests only. |

Note that `masking.build()` is additionally gated on
`self.cfg.mask in ("model", "parsing")`, so a `PipelineConfig(mask="oval")`
disables both parsers regardless of the `RenderOptions` flags.

### 14.2 Quality presets, as `jobs.JobRunner._run_v2()` builds them

| Preset | Config |
|---|---|
| `auto` | Runs `benchmark.run()` and uses the winning `PipelineConfig` verbatim. |
| `quality` | `hyperswap_1a_256` + `gpen_bfr_512 @ 0.7`, `mask="model"`. |
| `fast` | `hyperswap_1a_256`, no enhancer, `color_match=False`, and `use_parsing`/`use_occlusion` forced off — a preview path that skips every optional model. |

### 14.3 Progress stage weights

`pipeline.STAGES` and `jobs.STAGE_WEIGHTS` are two separate tables. The job
layer's is the one that drives the UI bar:

| Stage | Weight |
|---|---:|
| preparing | 0.02 |
| analysing video | 0.02 |
| benchmarking swap models | 0.05 |
| testing restoration | 0.04 |
| selecting best pipeline | 0.01 |
| detecting | 0.01 |
| **processing** | **0.80** |
| encoding | 0.03 |
| restoring audio | 0.02 |

Progress writes are throttled to roughly two per second, because the pipeline
emits many times a second and unthrottled writes collide with the 1.5 s status
poller (`database is locked`). Stage changes are always persisted regardless of
the throttle.

---

## 15. Open items for a maintainer

Collected from the gaps noted above, all verified against the code:

1. **`is_vfr` is dead data.** Detected in `video.probe()`, never read.
   `open_encoder()` always writes CFR at `info.fps`. Either consume it (pass the
   original timestamps through, or use `-vsync passthrough` with a timestamp
   file) or downgrade the docstring claim.
2. **`decode()` does not pass `-autorotate`.** It relies on the ffmpeg default,
   which the comment claims it states explicitly. One flag would make the
   intent real.
3. **`identity_switches` is hard-coded to 0 in the benchmark aggregate**, so
   0.18 of the composite weight is constant across all candidates. Either run a
   `TargetLock` over the sampled frames in `score_config()`, or redistribute the
   weight and document that the component only applies to full renders.
4. **`masking.py` hard-codes two preprocessing conventions** that ought to be
   `ModelSpec` fields: an ImageNet normalisation mode and a tensor-layout
   (NCHW/NHWC) field. Today `bisenet_resnet_34` is declared `NEG_ONE_ONE` in the
   registry and that declaration is silently ignored by the code that uses it —
   a trap for whoever adds the third parser.
5. **`candidate_configs()` pairs enhancers with `swappers[0]` only**, not with
   the best bare swapper. Since `SWAPPERS` is a dict in insertion order, that is
   always `hyperswap_1a_256`. If a sibling wins the bare pass, restoration is
   never explored against it.
