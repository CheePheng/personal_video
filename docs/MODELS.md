# Model inventory and licensing record

**Status: research / personal use only. Nothing in this stack is cleared for commercial use.**

This is a local, single-user research and education project. It runs entirely on one machine,
against media the operator owns, with no hosted service, no third-party users and no
distribution of outputs. Every model listed here is used under research / personal-use terms,
and several are used under terms that are non-commercial, indeterminate, or simply absent.

The distinction this document cares about most is **code licence vs weights licence**. They are
different grants and they frequently disagree. InsightFace ships MIT *code* with explicitly
non-commercial *weights*. GFPGAN ships a file called `LICENSE` that says Apache-2.0 and then
carves out the very component that makes the model work. A repository with an MIT `LICENSE`
file tells you nothing about the `.onnx` you downloaded unless the project says so. Throughout
this document, "MIT code / non-commercial weights" means exactly that: you may reuse the source,
you may not commercially use the trained artefact.

Licence research below was conducted against primary upstream sources on 2026-09-21. Where
FaceFusion's own licence documentation disagrees with the upstream repository, the upstream
repository wins and the disagreement is recorded.

The registry in `app/render/registry.py` is the machine-readable form of this document. A model
that is not registered there is not fetched and cannot be run. Models considered and rejected
are kept in the `EXCLUDED` dict with their reason, so an audit can see what was looked at, not
just what was chosen.

Also deliberately absent: FaceFusion's `nsfw_1/2/3` classifiers and its content analyser. This
project does not classify the subject matter of the operator's own media, so those models are
not downloaded, not registered and not run.

---

## 1. Models in use

These are the thirteen models actually registered in `app/render/registry.py`. "Template" is the
alignment template the pipeline warps to before inference; "normalisation" is the input scaling
the ONNX graph expects.

### Detectors

| Model | Role | Size | Input | Template | Normalisation | Licence (code / weights) | Note |
|---|---|---|---|---|---|---|---|
| `yoloface_8n` | detector | 12.7 MB | fixed 640x640 | `arcface_112_v2` | `x/255` | Declares GPL-3.0, but derives from Ultralytics, which is **AGPL-3.0**. A downstream redistributor cannot downgrade AGPL. **Assume AGPL-3.0.** | Default detector: fast, good recall on frontal faces. AGPL section 13 (network use) is normally a dealbreaker for any hosted product. |
| `scrfd_2.5g` | detector | 3.3 MB | dynamic 160-640 | `arcface_112_v2` | `(x-127.5)/128` | InsightFace: **MIT code / non-commercial research weights** | Stronger on small and profile faces; different output decode to YOLO. |
| `retinaface_10g` | detector | 16.9 MB | dynamic 160-640 | `arcface_112_v2` | `(x-127.5)/128` | InsightFace: **MIT code / non-commercial research weights** | Heaviest, strongest recall. Reserved as the difficult-frame fallback. |

### Recogniser

| Model | Role | Size | Input | Template | Normalisation | Licence (code / weights) | Note |
|---|---|---|---|---|---|---|---|
| `arcface_w600k_r50` | recogniser | 174.4 MB | 112 | `arcface_112_v2` | `(x-127.5)/127.5` | InsightFace: **MIT code / NON-COMMERCIAL research weights** | 512-d identity vector. Used for source identity, tracking and benchmark scoring. **This is the binding commercial constraint - see section 4.** |

### Swappers

Nine swappers are registered and **all nine actually run inference** in Auto Max -- they are
benchmarked against the same sampled frames with the same metrics, not merely listed. Three
different identity-conditioning conventions are involved, and using the wrong one produces a
convincing face that is simply the wrong person:

| Convention | Families | What the model receives |
|---|---|---|
| L2-normalised ArcFace | HyperSwap 1a/1b/1c | the 512-d vector, unit length |
| Raw ArcFace | AlphaFace | the 512-d vector, **not** normalised |
| Converted | GHOST, SimSwap | ArcFace passed through a `crossface_*` 512->512 learned remap into that family's own identity space |

Measured identity on one reference face (cosine to source, higher is better), single frame,
same mask and colour pipeline for all:

| Swapper | Identity | ms | Notes |
|---|---|---|---|
| `hyperswap_1a_256` | **0.971** | ~1000 | best on this face |
| `hyperswap_1b_256` | 0.969 | ~770 | wins on dark footage (see below) |
| `hyperswap_1c_256` | 0.967 | ~775 | |
| `alphaface_256` | 0.902 | ~1080 | newest architecture (2026) |
| `ghost_1_256` | 0.811 | ~906 | best of the permissively-licensed family |
| `ghost_2_256` | 0.775 | ~1730 | |
| `ghost_3_256` | 0.768 | ~2590 | slowest |
| `simswap_unofficial_512` | 0.668 | ~1190 | only native-512 swapper |
| `simswap_256` | 0.604 | ~849 | |

HyperSwap leads on this face, but that ranking is **per-video, not universal** -- on the dark
clip Auto Max selected `hyperswap_1b_256` bare over `1a`, and over every enhanced pipeline.
That is exactly why the competition is run per render rather than decided once here.

The HyperSwap trio share an architecture at 256 px, take an **L2-normalised ArcFace 512-d
embedding**, and output their own mask.

| Model | Role | Size | Input | Template | Normalisation | Licence (code / weights) | Note |
|---|---|---|---|---|---|---|---|
| `hyperswap_1a_256` | swapper | 402.7 MB | 256 | `arcface_128` | mean/std 0.5 (`-1..1`) | **"ResearchRAIL-MS"** - terms indeterminate (see below) | V1's model. Strong identity, ships its own mask. Pixel boost 256/512/768/1024. |
| `hyperswap_1b_256` | swapper | 402.7 MB | 256 | `arcface_128` | mean/std 0.5 (`-1..1`) | **"ResearchRAIL-MS"** - terms indeterminate | Sibling of 1a; different identity/expression balance. |
| `hyperswap_1c_256` | swapper | 402.7 MB | 256 | `arcface_128` | mean/std 0.5 (`-1..1`) | **"ResearchRAIL-MS"** - terms indeterminate | Sibling of 1a; benchmarked, not assumed better. |

On "ResearchRAIL-MS": `facefusion-labs/hyperswap/LICENSE.md` is a **51-byte stub** containing only
the words "ResearchRAIL-MS license" plus a copyright line. There is **no grant of rights, no
enumerated restrictions, and no link to any canonical RAIL text**. licenses.ai does not define
this variant. The HuggingFace weight repositories carry zero licence metadata (`cardData: null`).
The obligations here are therefore **indeterminate, not merely strict** - we do not know what we
are permitted to do, which is a worse position than a clearly non-commercial licence. Treated as
research-only. Training data: VGGFace2.

#### Newly implemented in V2.1

| Model | Size | Input | Template | Normalisation | Licence (code / weights) | Note |
|---|---|---|---|---|---|---|
| `alphaface_256` | 555.6 MB | 256 | `arcface_128` | `x/255` (mean 0, std 1) | **MIT per upstream repo**; FaceFusion labels weights NC - unresolved | arXiv:2601.16429 (Yu et al., 2026). Takes the **raw** ArcFace vector. |
| `ghost_1_256` | 514.9 MB | 256 | `arcface_112_v1` | mean/std 0.5 | **Apache-2.0** (ai-forever); ONNX is a third-party conversion | The only genuinely permissive swapper family available. |
| `ghost_2_256` | 738.7 MB | 256 | `arcface_112_v1` | mean/std 0.5 | Apache-2.0 | Second training run. |
| `ghost_3_256` | 855.5 MB | 256 | `arcface_112_v1` | mean/std 0.5 | Apache-2.0 | Third training run. |
| `simswap_256` | 220.4 MB | 256 | `arcface_112_v1` | **ImageNet** mean/std | CC BY-NC 4.0 | The **only** model here on ImageNet statistics - a classic silent-bug source. |
| `simswap_unofficial_512` | 239.2 MB | 512 | `arcface_112_v1` | `x/255` | CC BY-NC 4.0 | Official neuralchen 512 beta despite the "unofficial" filename. |
| `crossface_ghost` | 22.1 MB | - | - | - | Apache-2.0 | 512->512 identity remap for GHOST (raw output). |
| `crossface_simswap` | 22.1 MB | - | - | - | CC BY-NC 4.0 | 512->512 identity remap for SimSwap (output re-normalised). |

GHOST and SimSwap were failing before V2.1 purely for want of the `arcface_112_v1` alignment
template, which predates `v2` and is what those families were trained against. With the wrong
template they still produced a face -- just a subtly misaligned, subtly wrong one, which is the
failure mode worth guarding against.

### Restorers / enhancers

All four operate at 512 px on the `ffhq_512` template with `(x/255 - 0.5)/0.5` normalisation.

| Model | Role | Size | Input | Template | Normalisation | Licence (code / weights) | Note |
|---|---|---|---|---|---|---|---|
| `gfpgan_1.4` | enhancer | 340.3 MB | 512 | `ffhq_512` | `(x/255-0.5)/0.5` | **NOASSERTION.** LICENSE says "Apache 2.0 **EXCEPT** for the third-party components listed below"; GitHub classifies the repo NOASSERTION. Embedded NVIDIA StyleGAN2 licence section 3.3: "may be used ... non-commercially ... research or evaluation purposes only". DFDNet also embedded under CC BY-NC-SA 4.0. **Effectively non-commercial.** | The StyleGAN2 carve-out binds because GFPGAN's generator *is* StyleGAN2. Also effectively unmaintained: last functional release 2022-09-16, 402 open issues. Strong detail, but can pull identity toward a generic face at 100%. |
| `codeformer` | enhancer | 377.0 MB | 512 | `ffhq_512` | `(x/255-0.5)/0.5` | **S-Lab License 1.0 - explicitly non-commercial** | Takes an optional fidelity "weight" input; often better identity retention than GFPGAN. |
| `gpen_bfr_512` | enhancer | 284.3 MB | 512 | `ffhq_512` | `(x/255-0.5)/0.5` | **NO LICENCE FILE AT ALL upstream** (verified) = all rights reserved. Worse than the "Non-Commercial" label FaceFusion applies. | Upstream README even states "I have to take it down due to commercial issues." Gentler than GFPGAN; good for medium faces. |
| `restoreformer_plus_plus` | enhancer | 294.3 MB | 512 | `ffhq_512` | `(x/255-0.5)/0.5` | **Verbatim standard Apache-2.0, no carve-outs. THE ONLY CLEAN RESTORER.** | Transformer restorer; benchmarked against the others rather than assumed best. |

### Parsers / masking

| Model | Role | Size | Input | Template | Normalisation | Licence (code / weights) | Note |
|---|---|---|---|---|---|---|---|
| `bisenet_resnet_34` | parser | 93.6 MB | 512 | `ffhq_512` | **ImageNet** mean `.485/.456/.406`, std `.229/.224/.225` | **MIT code**, but weights trained on CelebAMask-HQ, whose Dataset Agreement forbids commercially exploiting "any portion of derived data". **Code MIT, weights arguably non-commercial - an undeclared exposure.** | 19-class CelebAMask-HQ parsing. The clearest example in this stack of the code/weights split mattering. |
| `xseg_1` | parser | 70.3 MB | 256 | `arcface_128` | `x/255` | FaceFusion claims GPL-3.0; **provenance UNVERIFIED** (see section 6) | Occlusion-aware mask. **NHWC input layout, not NCHW - unique among these models** (verified from the ONNX graph). |

---

## 2. Models evaluated and excluded

Recorded in `EXCLUDED` in `app/render/registry.py` so the rejection survives in code, not just prose.

### Swappers

| Model | Reason for exclusion |
|---|---|
| `inswapper_128` / `inswapper_128_fp16` | InsightFace states that "models trained with these data are available for non-commercial research purposes only", and explicitly closed the auto-download loophole. Official download withdrawn around May 2023. Also only 128 px - lower resolution than every alternative here. |
| `uniface_256` | Upstream repo has **no LICENSE file** (verified via the GitHub contents API, `license: null`). Unlicensed means **all rights reserved - stricter than research-only, not more permissive.** Separately, it takes a source **image**, not an embedding (verified from its ONNX inputs: `target[1,3,256,256]` + `source[1,3,256,256]`), so it cannot consume our fused multi-photo identity. |
| `simswap_256` | **CC BY-NC 4.0.** Needs a crossface embedding converter (`arcface_converter_simswap`). Also the only swapper in the field using ImageNet mean/std. |
| `simswap_unofficial_512` | **CC BY-NC 4.0.** Note: despite the "unofficial" filename this is an **official neuralchen beta**. Still non-commercial, still needs a converter. |
| `ghost_1_256` / `ghost_2_256` / `ghost_3_256` | **Apache-2.0 - genuinely permissive** - but needs a crossface converter, and the ONNX files are third-party conversions (netrunner.exe) of GHOST v1. The Apache grant covers ai-forever's release; the conversion chain adds a link that grant does not obviously reach. VGGFace2 training-data terms may also be narrower than the code grant. Held back pending that review, not on licence grounds alone. |
| `hififace_unofficial_256` | **UNTRACEABLE.** `github.com/GuijiAI/HiFiFace-onnx` returns 404, the org has no public repos, FaceFusion's own metadata records Converter=Unknown / License=Unknown, and the original paper never released weights. Do not use. |
| `alphaface_256` | Repo LICENSE is standard MIT (Copyright 2026 Jongmin Andrew Yu, arXiv:2601.16429) but FaceFusion labels the **weights** Non-Commercial - an **unresolved contradiction** between code and weights grant. The newest credible entrant (2026), and it uses a **raw unnormalised** ArcFace embedding rather than an L2-normalised one. Excluded until upstream clarifies. |
| `blendswap_256` | **CC BY-NC-SA 4.0**, and takes a source image rather than an embedding. |

### Other

| Model | Reason for exclusion |
|---|---|
| `nsfw_1/2/3` | Content classifiers. Out of scope by design: this project does not classify the subject matter of the operator's media. Not a licensing exclusion. |

---

## 3. Commercial status of each registered model, taken alone

For orientation before the blocker section:

| Model | Commercially clean on its own? |
|---|---|
| `restoreformer_plus_plus` | **Yes** - plain Apache-2.0 |
| `yoloface_8n` | No - assume AGPL-3.0, section 13 |
| `scrfd_2.5g`, `retinaface_10g` | No - non-commercial weights |
| `arcface_w600k_r50` | **No - and unavoidable. See section 4.** |
| `hyperswap_1a/1b/1c_256` | No - indeterminate terms |
| `gfpgan_1.4`, `codeformer`, `gpen_bfr_512` | No - non-commercial, non-commercial, and all-rights-reserved respectively |
| `bisenet_resnet_34` | No - MIT code, non-commercial dataset terms on the weights |
| `xseg_1` | Unknown - provenance unverified |

`commercial_safe()` in `registry.py` returns `True` only for a licence string containing MIT or
Apache and not containing "non-commercial". It is a coarse screen over the strings recorded above,
not a legal opinion.

---

## 4. The commercial-use blocker

**Even if every other model in this stack were swapped for a permissive one, the stack would still
not be commercially clean, because of `arcface_w600k_r50`.**

The chain is short and has no branch in it:

1. Every swapper we can actually drive is **embedding-conditioned**: it takes a 512-d ArcFace
   identity vector as its conditioning input. That is true of all three HyperSwap variants, and
   it is equally true of GHOST, SimSwap and AlphaFace - the ones needing a crossface converter
   still ultimately consume an ArcFace-derived vector; the converter merely remaps it.
2. The only model in the field that produces that 512-d vector is `arcface_w600k_r50`, from
   InsightFace's buffalo_l pack.
3. InsightFace's position is explicit: **MIT code, non-commercial research weights.** The code
   licence is irrelevant here - we are not reusing their source, we are running their trained
   artefact, and the artefact is the thing restricted.
4. **No permissively-licensed 512-d face recogniser was found.** Not "none preferred" - none
   located at all in this research pass.

The alternatives that do not work:

- Swapping the recogniser for something MIT is not possible, because there is no MIT 512-d
  recogniser to swap to.
- Swapping to a swapper that takes a source **image** instead of an embedding (UniFace, BlendSwap)
  removes the ArcFace dependency but loses the multi-photo fused identity that is the point of
  this project's source handling - and both of those models are excluded on licence grounds anyway
  (all-rights-reserved and CC BY-NC-SA 4.0 respectively).
- Training a recogniser from scratch on permissively-licensed face data is not a swap; it is a
  separate research project.

So: the recogniser is load-bearing, it is non-commercial, and it has no permissive substitute.
**The stack cannot currently be made commercially clean.** Everything else in this document is
secondary to that fact.

---

## 5. If this ever needed to be commercially clean

Should the blocker in section 4 ever be resolved, these are the candidate permissive replacements
already identified for the other roles. This is a map for a future attempt, not a claim that the
attempt would succeed.

| Role | Current (not clean) | Permissive candidate | Status of the candidate |
|---|---|---|---|
| Detector | `yoloface_8n` (AGPL), `scrfd_2.5g` / `retinaface_10g` (NC weights) | **`yunet_2023_mar`** (0.2 MB) | **MIT, and opencv_zoo's README explicitly states the `.onnx` weights are covered** - the rare case of unambiguously licensed weights, where the code grant and the weights grant agree. Not currently used; this is the clean alternative. |
| Swapper | `hyperswap_1a/1b/1c_256` (indeterminate) | **GHOST** (`ghost_1/2/3_256`) | **Apache-2.0** on ai-forever's release. Caveats: needs a crossface converter, the ONNX files are third-party conversions (netrunner.exe) whose chain the Apache grant does not obviously cover, and VGGFace2 training-data terms may be narrower than the code grant. |
| Restorer | `gfpgan_1.4` / `codeformer` / `gpen_bfr_512` | **`restoreformer_plus_plus`** | **Already registered and already clean** - verbatim Apache-2.0 with no carve-outs. This role is solved today. |
| Parser | `bisenet_resnet_34` (NC dataset terms), `xseg_1` (unverified) | **SegFace** (AAAI 2025) | **MIT**, 88.96 mean F1 on CelebAMask-HQ. Important: use the **LaPa checkpoints**, not the CelebAMask-HQ ones - otherwise the replacement inherits exactly the dataset-terms problem that makes BiSeNet's weights unclean, and an MIT code licence would again not save it. |
| **Recogniser** | **`arcface_w600k_r50`** | **None found** | **UNSOLVED. This is the gap.** No permissively-licensed 512-d face recogniser was located. Until one exists, the four rows above cannot add up to a clean stack. |

Note the pattern in that table: three of the four solvable roles are solvable specifically because
someone published weights with an explicit grant (opencv_zoo, RestoreFormer++) or because the
checkpoint can be chosen to avoid an encumbered dataset (SegFace / LaPa). The recogniser row fails
for the opposite reason - the grant is explicitly withheld on the weights.

---

## 6. What could not be verified

Recorded because "unverified" is different from "fine", and the difference matters.

**1. The actual terms of "ResearchRAIL-MS" / "OpenRAIL-AS".** No published text exists anywhere for
these licence names. The HyperSwap `LICENSE.md` is a 51-byte stub naming the licence and asserting
copyright, with no grant of rights, no enumerated restrictions and no link to a canonical RAIL
document. licenses.ai does not define the variant. The HuggingFace weight repos carry
`cardData: null`. We cannot state what these licences permit or forbid, because the terms do not
appear to have been published at all. The three HyperSwap models - the stack's primary swappers -
are therefore used under terms nobody can read.

**2. The authorship and training data of the HifiFace and XSeg ONNX files.**

- `hififace_unofficial_256`: upstream repo 404s, the org has no public repos, FaceFusion's metadata
  says Converter=Unknown and License=Unknown, and the original paper never released weights. There
  is no traceable author and no traceable training set. Excluded.
- `xseg_1`: FaceFusion claims GPL-3.0, but **DeepFaceLab never shipped an ONNX XSeg** (it used
  `.npy` / SAEHD), the available variants match no DFL-official artefact, and the files are served
  from FaceFusion's own bucket with no upstream LICENSE and no model card. The GPL-3.0 claim is
  unsupported by anything we can check. It is registered and in use under a research-use posture,
  with this uncertainty recorded rather than papered over.

**3. Upstream licence documentation is not reliable.** `docs.facefusion.io/introduction/licenses` is
useful as a starting point but is **demonstrably wrong in places**: it labels GFPGAN Apache-2.0 (it
is NOASSERTION with a binding NVIDIA non-commercial carve-out), labels `gpen_bfr_512` merely
"Non-Commercial" (upstream has no licence file at all, which is stricter), and labels AlphaFace's
weights Non-Commercial against an upstream MIT LICENSE. Where this document and FaceFusion's
licence page disagree, this document followed the upstream repository.

### Known drift inside this repository

Two internal inconsistencies were found while writing this document. Neither is a licensing
question, but both affect whether the table in section 1 matches what the code does.

- **`models/registry.json` is a stale export.** `app/render/registry.py` is the source of truth.
  The checked-in JSON still lists `uniface_256` and `dfl_xseg` as registered models (neither is in
  `registry.py`), and carries older, less accurate licence strings - it calls `gfpgan_1.4`
  "Apache-2.0", `scrfd_2.5g` "Apache-2.0", `yoloface_8n` "MIT" and `restoreformer_plus_plus`
  "S-Lab License 1.0", all of which this document contradicts. Re-run `export_registry()` to
  regenerate it. Do not cite the JSON's licence fields; cite this document or `registry.py`.
- **Normalisation fields disagree with the verified values** for three models. `registry.py`
  declares `bisenet_resnet_34` as `NEG_ONE_ONE` while its own `notes` field says it "Needs
  IMAGENET normalization", and declares `scrfd_2.5g` / `retinaface_10g` as `ZERO_ONE` where the
  verified value is `(x-127.5)/128`. Section 1 above records the **verified** values. The registry
  entries should be reconciled to match.

---

## 7. How model files are obtained and verified

Model files are **not in git**. `.gitignore` excludes `models/` and `*.onnx` - the files total
several gigabytes, and their licences would make redistribution the wrong thing to do even if the
size did not.

They are fetched by `app/render/download.py`, which:

- reads the URL, filename and expected size for each model from the registry - the registry is the
  single source of truth for provenance;
- downloads resumably via HTTP Range, and does not append to a partial file if the server ignores
  the Range header;
- computes a **SHA256** for each file and records it in **`models/hashes.json`** alongside the byte
  count. The first download of a model has no recorded hash - that run is the one that records it;
- **re-verifies that SHA256 on every subsequent load.** A file whose hash does not match is treated
  as corrupt and raises, rather than being silently used;
- refuses a download under 1 KB as empty or truncated.

Usage:

```
python -m app.render.download                    # everything registered
python -m app.render.download hyperswap_1b_256 codeformer
```

This gives a reproducible, tamper-evident record of exactly which bytes were run, which is the
other half of a licensing record: knowing what a licence says is only useful if you also know
which artefact you actually executed.
