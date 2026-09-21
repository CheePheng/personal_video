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
        pixel_boost=(256, 512, 768, 1024),
        notes="V1's model. Strong identity, ships its own mask."),
    "hyperswap_1b_256": ModelSpec(
        name="hyperswap_1b_256", filename="hyperswap_1b_256.onnx", role="swapper",
        input_size=256, template="arcface_128", normalization=Normalization.NEG_ONE_ONE,
        license="ResearchRAIL-MS (FaceFusion) - no published terms text; treat as research-only",
        source_url=_u("models-3.3.0", "hyperswap_1b_256.onnx"),
        needs_embedding=True, embedding_normalized=True, outputs_mask=True,
        pixel_boost=(256, 512, 768, 1024),
        notes="Sibling of 1a; different identity/expression balance."),
    "hyperswap_1c_256": ModelSpec(
        name="hyperswap_1c_256", filename="hyperswap_1c_256.onnx", role="swapper",
        input_size=256, template="arcface_128", normalization=Normalization.NEG_ONE_ONE,
        license="ResearchRAIL-MS (FaceFusion) - no published terms text; treat as research-only",
        source_url=_u("models-3.3.0", "hyperswap_1c_256.onnx"),
        needs_embedding=True, embedding_normalized=True, outputs_mask=True,
        pixel_boost=(256, 512, 768, 1024),
        notes="Sibling of 1a; benchmarked, not assumed better."),
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
    **DETECTORS, **RECOGNIZERS, **SWAPPERS, **ENHANCERS, **PARSERS,
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
