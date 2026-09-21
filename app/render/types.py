"""Typed structures shared across the V2 render pipeline.

Kept free of heavy imports (no onnxruntime, no cv2 at module scope) so any
module can import these without dragging in the CUDA stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class RenderError(RuntimeError):
    """A genuine technical failure.

    Raised only for: corrupt media, unsupported codec, missing model, no
    detectable face, CUDA/model init failure, disk error, encoding failure,
    insufficient VRAM. Never for the subject matter of the media.
    """


@dataclass(slots=True)
class Face:
    """One detected face in one frame."""

    box: np.ndarray                      # (4,) x1,y1,x2,y2 in frame pixels
    kps: np.ndarray                      # (5,2) eyes, nose, mouth corners
    score: float
    embedding: Optional[np.ndarray] = None   # (512,) L2-normalised, lazily filled

    @property
    def area(self) -> float:
        return float((self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2,
                         (self.box[1] + self.box[3]) / 2], dtype=np.float32)

    @property
    def size(self) -> float:
        """Geometric mean of width/height -- a scale measure robust to aspect."""
        w = float(self.box[2] - self.box[0])
        h = float(self.box[3] - self.box[1])
        return float(np.sqrt(max(w, 1.0) * max(h, 1.0)))


@dataclass(slots=True)
class SourceIdentity:
    """The person being swapped IN, built from 1..N photos."""

    embedding: np.ndarray                # (1,512) fused, L2-normalised
    per_image: list[dict[str, Any]] = field(default_factory=list)
    n_images: int = 1


class Normalization(str, Enum):
    """How a model wants its input scaled."""

    ZERO_ONE = "0_1"            # x / 255
    NEG_ONE_ONE = "-1_1"        # (x/255 - 0.5) / 0.5
    ARCFACE = "arcface"         # (x - 127.5) / 127.5
    IMAGENET = "imagenet"       # (x/255 - imagenet_mean) / imagenet_std
    CENTRED_128 = "centred_128" # (x - 127.5) / 128   (InsightFace detectors)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything needed to run one ONNX model, declared not hard-coded.

    The pipeline reads these fields; it never special-cases a model by name.
    Adding a swapper is therefore a registry entry, not a code change.
    """

    name: str
    filename: str
    role: str                            # swapper | detector | recognizer | parser | enhancer
    input_size: int
    template: str                        # key into alignment.TEMPLATES
    normalization: Normalization
    license: str
    source_url: str
    sha256: Optional[str] = None
    expected_bytes: Optional[int] = None
    # Swapper-specific
    needs_embedding: bool = False
    embedding_normalized: bool = True    # hyperswap wants the L2-normed vector
    outputs_mask: bool = False
    pixel_boost: tuple[int, ...] = ()
    # Runtime
    providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    notes: str = ""


@dataclass(slots=True)
class FrameMetrics:
    """Per-frame quality measurements collected during benchmarking."""

    identity: float = 0.0                # cosine(source, rendered)
    identity_prev: float = 0.0           # cosine(rendered_t, rendered_t-1)
    landmark_delta: float = 0.0          # geometry drift vs target
    seam: float = 0.0                    # boundary discontinuity
    sharpness: float = 0.0               # Laplacian variance
    ms: float = 0.0


@dataclass(slots=True)
class PipelineConfig:
    """One candidate pipeline: which models and settings to run."""

    swapper: str
    enhancer: Optional[str] = None
    enhancer_blend: float = 0.0
    mask: str = "model"                  # model | parsing | oval
    pixel_boost: int = 0
    color_match: bool = True
    label: str = ""

    def describe(self) -> str:
        bits = [self.swapper]
        if self.enhancer:
            bits.append(f"{self.enhancer}@{int(self.enhancer_blend * 100)}%")
        else:
            bits.append("no-enhance")
        bits.append(f"mask:{self.mask}")
        if self.pixel_boost:
            bits.append(f"boost:{self.pixel_boost}")
        return " / ".join(bits)
