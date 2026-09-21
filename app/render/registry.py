"""Model registry: every model the pipeline can run, declared as data.

The renderer never special-cases a model by name -- it reads the spec. Adding
or swapping a model is a registry entry plus a download, not a code change.

Licensing is recorded per model and enforced by ``ALLOWED``: models whose terms
do not fit this project are listed in docs/MODELS.md with the reason and are
simply not registered here.

Deliberately absent: FaceFusion's nsfw_1/2/3 classifiers and its content
analyser. This project does not classify the subject matter of user media.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.render.types import ModelSpec, Normalization, RenderError

# PIXEL BOOST: 512 disabled, 768/1024 offered. This is a CORRECTION of an
# earlier, wrong conclusion, and the reasoning matters more than the setting.
#
# The first measurement said boost was harmful at every level and the option
# was withdrawn. That measurement scored identity with ArcFace -- the same
# model the swap is conditioned on. Boost changes the crop's frequency
# content, which is exactly where an in-loop recogniser is least trustworthy.
#
# Re-measured over 30 samples from 6 clips against SFace, an independently
# trained judge held out of the optimisation loop (app/render/judges.py):
#
#     boost   ArcFace (in-loop)   SFace (judge)   sharpness
#      256         0.9462            0.9034          863
#      512         0.9239 (-.022)    0.8892 (-.014)  896
#      768         0.9419 (-.004)    0.9151 (+.012)  891
#     1024         0.9435 (-.003)    0.9122 (+.009)  888
#
# The judges AGREE that 512 is worst -- two tiles split the face down the
# middle. They DISAGREE in sign at 768 and 1024: ArcFace says marginally
# worse (within noise), the independent judge says better.
#
# So 512 stays off, 768/1024 become benchmarked options, and the blanket
# claim "boost is harmful" is retracted. Note the honest limitation: Auto Max
# still ranks with ArcFace, so it will rarely select boost on its own. Which
# judge is right is not settled by these numbers -- that needs human
# evaluation. What IS settled is that the original conclusion was drawn from
# a closed loop and should not have been stated as flatly as it was.
ASSETS = "https://github.com/facefusion/facefusion-assets/releases/download"
REGISTRY_JSON = Path(__file__).resolve().parent.parent.parent / "models" / "registry.json"


def _u(tag: str, name: str) -> str:
    return f"{ASSETS}/{tag}/{name}"


# ---------------------------------------------------------------- detectors
DETECTORS: dict[str, ModelSpec] = {
    "yoloface_8n": ModelSpec(
        name="yoloface_8n", filename="yoloface_8n.onnx", role="detector",
        input_size=640, template="arcface_112_v2", normalization=Normalization.ZERO_ONE,
        license="Declares GPL-3.0 but derives from Ultralytics (AGPL-3.0); assume AGPL-3.0. Network deployment triggers section 13",
        source_url=_u("models-3.0.0", "yoloface_8n.onnx"),
        notes="Fast, good recall on frontal faces. Default detector."),
    "scrfd_2.5g": ModelSpec(
        name="scrfd_2.5g", filename="scrfd_2.5g.onnx", role="detector",
        input_size=640, template="arcface_112_v2", normalization=Normalization.CENTRED_128,
        license="InsightFace: MIT code / non-commercial research weights",
        source_url=_u("models-3.0.0", "scrfd_2.5g.onnx"),
        notes="Stronger on small/profile faces; different output decode to YOLO."),
    "retinaface_10g": ModelSpec(
        name="retinaface_10g", filename="retinaface_10g.onnx", role="detector",
        input_size=640, template="arcface_112_v2", normalization=Normalization.CENTRED_128,
        license="InsightFace: MIT code / non-commercial research weights",
        source_url=_u("models-3.0.0", "retinaface_10g.onnx"),
        notes="Heavier; strongest recall. Reserved as difficult-frame fallback."),
}

# ---------------------------------------------------------------- recognizer
RECOGNIZERS: dict[str, ModelSpec] = {
    "arcface_w600k_r50": ModelSpec(
        name="arcface_w600k_r50", filename="arcface_w600k_r50.onnx", role="recognizer",
        input_size=112, template="arcface_112_v2", normalization=Normalization.ARCFACE,
        license="MIT code / non-commercial research weights (InsightFace buffalo_l)",
        source_url=_u("models-3.0.0", "arcface_w600k_r50.onnx"),
        notes="512-d identity. Used for source ID, tracking and benchmark scoring."),
}

# ---------------------------------------------------------------- swappers
# Only models whose identity conditioning we can drive directly are registered.
# inswapper_128 is excluded on licensing (see docs/MODELS.md); simswap/ghost/
# hififace need an arcface_converter_* matrix, registered alongside them.
SWAPPERS: dict[str, ModelSpec] = {
    "hyperswap_1a_256": ModelSpec(
        name="hyperswap_1a_256", filename="hyperswap_1a_256.onnx", role="swapper",
        input_size=256, template="arcface_128", normalization=Normalization.NEG_ONE_ONE,
        license="ResearchRAIL-MS (FaceFusion) - no published terms text; treat as research-only",
        source_url=_u("models-3.3.0", "hyperswap_1a_256.onnx"),
        needs_embedding=True, embedding_normalized=True, outputs_mask=True,
        pixel_boost=(768, 1024),   # 512 excluded; see the note above
        notes="V1's model. Strong identity, ships its own mask."),
    "hyperswap_1b_256": ModelSpec(
        name="hyperswap_1b_256", filename="hyperswap_1b_256.onnx", role="swapper",
        input_size=256, template="arcface_128", normalization=Normalization.NEG_ONE_ONE,
        license="ResearchRAIL-MS (FaceFusion) - no published terms text; treat as research-only",
        source_url=_u("models-3.3.0", "hyperswap_1b_256.onnx"),
        needs_embedding=True, embedding_normalized=True, outputs_mask=True,
        pixel_boost=(768, 1024),   # 512 excluded; see the note above
        notes="Sibling of 1a; different identity/expression balance."),
    "hyperswap_1c_256": ModelSpec(
        name="hyperswap_1c_256", filename="hyperswap_1c_256.onnx", role="swapper",
        input_size=256, template="arcface_128", normalization=Normalization.NEG_ONE_ONE,
        license="ResearchRAIL-MS (FaceFusion) - no published terms text; treat as research-only",
        source_url=_u("models-3.3.0", "hyperswap_1c_256.onnx"),
        needs_embedding=True, embedding_normalized=True, outputs_mask=True,
        pixel_boost=(768, 1024),   # 512 excluded; see the note above
        notes="Sibling of 1a; benchmarked, not assumed better."),
    "alphaface_256": ModelSpec(
        name="alphaface_256", filename="alphaface_256.onnx", role="swapper",
        input_size=256, template="arcface_128", normalization=Normalization.ZERO_ONE,
        license="MIT per upstream repo (Yu et al. 2026, arXiv:2601.16429); "
                "FaceFusion labels the weights Non-Commercial - contradiction unresolved",
        source_url=_u("models-3.9.0", "alphaface_256.onnx"),
        needs_embedding=True, embedding_normalized=False, outputs_mask=False,
        notes="2026 entrant. Takes the RAW unnormalised ArcFace vector."),
    "ghost_1_256": ModelSpec(
        name="ghost_1_256", filename="ghost_1_256.onnx", role="swapper",
        input_size=256, template="arcface_112_v1", normalization=Normalization.NEG_ONE_ONE,
        license="Apache-2.0 (ai-forever GHOST); ONNX is a third-party conversion",
        source_url=_u("models-3.0.0", "ghost_1_256.onnx"),
        needs_embedding=True, embedding_normalized=False, outputs_mask=False,
        embedding_converter="crossface_ghost", converter_normalize=False,
        notes="The only genuinely permissive swapper family available."),
    "ghost_2_256": ModelSpec(
        name="ghost_2_256", filename="ghost_2_256.onnx", role="swapper",
        input_size=256, template="arcface_112_v1", normalization=Normalization.NEG_ONE_ONE,
        license="Apache-2.0 (ai-forever GHOST); ONNX is a third-party conversion",
        source_url=_u("models-3.0.0", "ghost_2_256.onnx"),
        needs_embedding=True, embedding_normalized=False, outputs_mask=False,
        embedding_converter="crossface_ghost", converter_normalize=False,
        notes="Second GHOST training run."),
    "ghost_3_256": ModelSpec(
        name="ghost_3_256", filename="ghost_3_256.onnx", role="swapper",
        input_size=256, template="arcface_112_v1", normalization=Normalization.NEG_ONE_ONE,
        license="Apache-2.0 (ai-forever GHOST); ONNX is a third-party conversion",
        source_url=_u("models-3.0.0", "ghost_3_256.onnx"),
        needs_embedding=True, embedding_normalized=False, outputs_mask=False,
        embedding_converter="crossface_ghost", converter_normalize=False,
        notes="Third GHOST training run."),
    "simswap_256": ModelSpec(
        name="simswap_256", filename="simswap_256.onnx", role="swapper",
        input_size=256, template="arcface_112_v1", normalization=Normalization.IMAGENET,
        license="CC BY-NC 4.0 (neuralchen SimSwap) - non-commercial research",
        source_url=_u("models-3.0.0", "simswap_256.onnx"),
        needs_embedding=True, embedding_normalized=False, outputs_mask=False,
        embedding_converter="crossface_simswap", converter_normalize=True,
        notes="The ONLY model here using ImageNet mean/std - classic silent-bug source."),
    "simswap_unofficial_512": ModelSpec(
        name="simswap_unofficial_512", filename="simswap_unofficial_512.onnx",
        role="swapper", input_size=512, template="arcface_112_v1",
        normalization=Normalization.ZERO_ONE,
        license="CC BY-NC 4.0 (official neuralchen 512 beta despite the filename)",
        source_url=_u("models-3.0.0", "simswap_unofficial_512.onnx"),
        needs_embedding=True, embedding_normalized=False, outputs_mask=False,
        embedding_converter="crossface_simswap", converter_normalize=True,
        notes="Native 512px input - the only swapper above 256."),
}

# Identity-space converters. Not swappers themselves: they remap an ArcFace
# vector into the space a given swapper family was trained against.
CONVERTERS: dict[str, ModelSpec] = {
    "crossface_ghost": ModelSpec(
        name="crossface_ghost", filename="crossface_ghost.onnx", role="converter",
        input_size=0, template="arcface_112_v2", normalization=Normalization.ZERO_ONE,
        license="Apache-2.0 (accompanies GHOST)",
        source_url=_u("models-3.4.0", "crossface_ghost.onnx"),
        notes="512-d -> 512-d remap for GHOST."),
    "crossface_simswap": ModelSpec(
        name="crossface_simswap", filename="crossface_simswap.onnx", role="converter",
        input_size=0, template="arcface_112_v2", normalization=Normalization.ZERO_ONE,
        license="CC BY-NC 4.0 (accompanies SimSwap)",
        source_url=_u("models-3.4.0", "crossface_simswap.onnx"),
        notes="512-d -> 512-d remap for SimSwap; output is L2-normalised."),
}

# ---------------------------------------------------------------- enhancers
ENHANCERS: dict[str, ModelSpec] = {
    "gfpgan_1.4": ModelSpec(
        name="gfpgan_1.4", filename="gfpgan_1.4.onnx", role="enhancer",
        input_size=512, template="ffhq_512", normalization=Normalization.NEG_ONE_ONE,
        license="NOASSERTION - Apache-2.0 EXCEPT embedded StyleGAN2 (NVIDIA non-commercial) and DFDNet (CC BY-NC-SA 4.0). Effectively non-commercial",
        source_url=_u("models-3.0.0", "gfpgan_1.4.onnx"),
        notes="Strong detail; can pull identity toward a generic face at 100%."),
    "codeformer": ModelSpec(
        name="codeformer", filename="codeformer.onnx", role="enhancer",
        input_size=512, template="ffhq_512", normalization=Normalization.NEG_ONE_ONE,
        license="S-Lab License 1.0 - NON-COMMERCIAL research use only",
        source_url=_u("models-3.0.0", "codeformer.onnx"),
        notes="Takes a fidelity weight; often better identity retention."),
    "gpen_bfr_512": ModelSpec(
        name="gpen_bfr_512", filename="gpen_bfr_512.onnx", role="enhancer",
        input_size=512, template="ffhq_512", normalization=Normalization.NEG_ONE_ONE,
        license="NO LICENSE FILE upstream - all rights reserved. Research use only",
        source_url=_u("models-3.0.0", "gpen_bfr_512.onnx"),
        notes="Gentler than GFPGAN; good for medium faces."),
    "restoreformer_plus_plus": ModelSpec(
        name="restoreformer_plus_plus", filename="restoreformer_plus_plus.onnx",
        role="enhancer", input_size=512, template="ffhq_512",
        normalization=Normalization.NEG_ONE_ONE,
        license="Apache-2.0 (verified clean - the only permissive restorer)",
        source_url=_u("models-3.0.0", "restoreformer_plus_plus.onnx"),
        notes="Transformer restorer; benchmarked against the others."),
}

# ---------------------------------------------------------------- masking
PARSERS: dict[str, ModelSpec] = {
    "bisenet_resnet_34": ModelSpec(
        name="bisenet_resnet_34", filename="bisenet_resnet_34.onnx", role="parser",
        input_size=512, template="ffhq_512", normalization=Normalization.IMAGENET,
        license="MIT code / weights trained on CelebAMask-HQ (non-commercial dataset terms apply to derived data)",
        source_url=_u("models-3.0.0", "bisenet_resnet_34.onnx"),
        notes="19-class CelebAMask-HQ parsing. Needs IMAGENET normalization."),
    "xseg_1": ModelSpec(
        name="xseg_1", filename="xseg_1.onnx", role="parser",
        input_size=256, template="arcface_128", normalization=Normalization.ZERO_ONE,
        license="GPL-3.0 claimed by FaceFusion; upstream provenance UNVERIFIED",
        source_url=_u("models-3.1.0", "xseg_1.onnx"),
        notes="Occlusion-aware mask. NHWC input (not NCHW) - verified from ONNX."),
}

ALL: dict[str, ModelSpec] = {
    **DETECTORS, **RECOGNIZERS, **SWAPPERS, **CONVERTERS, **ENHANCERS, **PARSERS,
}

# Models excluded on licensing/provenance grounds -- kept as data so
# docs/MODELS.md and any audit can show what was considered and why.
EXCLUDED: dict[str, str] = {
    "uniface_256": (
        "Upstream repo has NO license file (verified via GitHub contents API): "
        "unlicensed means all rights reserved, which is stricter than the "
        "research-only models, not more permissive. It also takes a source "
        "IMAGE rather than an identity embedding (verified by inspecting the "
        "ONNX inputs), so it cannot use our multi-photo fused identity."),
    "alphaface_256": (
        "Repo LICENSE is MIT but FaceFusion labels the weights "
        "Non-Commercial -- an unresolved contradiction. Excluded until the "
        "upstream author clarifies; see docs/MODELS.md."),
    "blendswap_256": "CC BY-NC-SA 4.0, and takes a source image not an embedding.",
    "inswapper_128": (
        "InsightFace's swapper weights are released for non-commercial research "
        "and the project has publicly objected to redistribution/deepfake use. "
        "Also only 128px -- lower resolution than every alternative here."),
    "inswapper_128_fp16": "Same licensing objection as inswapper_128.",
    "simswap_256": (
        "Needs arcface_converter_simswap to remap the identity vector; the "
        "upstream SimSwap release is research-only with unclear redistribution "
        "terms. Registered only if benchmarking justifies the added complexity."),
    "simswap_unofficial_512": (
        "'Unofficial' weights of unclear provenance -- excluded under the "
        "project's rule against models without a clear upstream source."),
    "ghost_1_256/2/3": (
        "GHOST weights are research-only and need arcface_converter_ghost. "
        "Held back pending a licensing review; see docs/MODELS.md."),
    "hififace_unofficial_256": (
        "'Unofficial' reimplementation weights; provenance unclear."),
    "nsfw_1/2/3": (
        "Content classifiers. Explicitly out of scope: this project does not "
        "classify the subject matter of user media."),
}


def get_model(name: str) -> ModelSpec:
    spec = ALL.get(name)
    if spec is None:
        raise RenderError(f"unknown model '{name}'. Registered: {sorted(ALL)}")
    return spec


def by_role(role: str) -> dict[str, ModelSpec]:
    return {k: v for k, v in ALL.items() if v.role == role}


def commercial_safe(name: str) -> bool:
    """True only for permissively-licensed models (MIT/Apache)."""
    lic = get_model(name).license.lower()
    return ("mit" in lic or "apache" in lic) and "non-commercial" not in lic


def export_registry(path: Path = REGISTRY_JSON,
                    names: Iterable[str] | None = None) -> Path:
    """Write models/registry.json -- the download manifest + licence record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = list(names) if names else sorted(ALL)
    payload = {
        "version": 2,
        "note": ("Model manifest for the V2 render pipeline. No content/NSFW "
                 "classifier is included by design."),
        "models": {
            n: {
                "filename": ALL[n].filename, "role": ALL[n].role,
                "url": ALL[n].source_url, "license": ALL[n].license,
                "sha256": ALL[n].sha256, "expected_bytes": ALL[n].expected_bytes,
                "input_size": ALL[n].input_size, "template": ALL[n].template,
                "normalization": ALL[n].normalization.value,
                "notes": ALL[n].notes,
            } for n in chosen
        },
        "excluded": EXCLUDED,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
