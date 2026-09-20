"""ONNX Runtime session management.

Two rules here, both learned the hard way (see docs/FINDINGS.md):

1. Sessions are cached. Building one costs hundreds of ms and allocates VRAM;
   doing it per frame would dominate the render and thrash the GPU.

2. We check ``session.get_providers()``, never just
   ``onnxruntime.get_available_providers()``. ORT happily *lists* CUDA and then
   silently runs on CPU when its DLLs cannot load -- turning a 2-minute render
   into an hour with no error anywhere. That silent fallback is treated as a
   hard failure.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

from app.render.types import ModelSpec, RenderError

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_lock = threading.Lock()
_cache: dict[str, ort.InferenceSession] = {}


def _make(spec: ModelSpec, allow_cpu: bool) -> ort.InferenceSession:
    path = MODELS_DIR / spec.filename
    if not path.exists():
        raise RenderError(
            f"missing model '{spec.name}' at {path}. "
            f"Run: python -m app.render.download {spec.name}")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = list(spec.providers)
    if not allow_cpu:
        providers = [p for p in providers if p != "CPUExecutionProvider"]
        if not providers:
            providers = ["CUDAExecutionProvider"]

    try:
        sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    except Exception as e:  # noqa: BLE001
        raise RenderError(
            f"could not initialise model '{spec.name}': {type(e).__name__}: {e}") from e

    active = sess.get_providers()
    if not allow_cpu and (not active or active[0] != "CUDAExecutionProvider"):
        raise RenderError(
            f"CUDA provider did not bind for '{spec.name}' -- session is using "
            f"{active}. Refusing to run on CPU silently; a render would take "
            f"hours. Check the CUDA 13 runtime (see docs/FINDINGS.md).")
    return sess


def get(spec: ModelSpec, allow_cpu: bool = False) -> ort.InferenceSession:
    """Fetch a cached session, building it on first use."""
    with _lock:
        s = _cache.get(spec.name)
        if s is None:
            s = _make(spec, allow_cpu)
            _cache[spec.name] = s
        return s


def run(spec: ModelSpec, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model, converting an OOM into an explicit, actionable error."""
    try:
        return get(spec).run(None, feeds)
    except Exception as e:  # noqa: BLE001
        text = str(e).lower()
        if "out of memory" in text or "cudnn_status_alloc_failed" in text or "cublas" in text:
            raise RenderError(
                f"insufficient VRAM running '{spec.name}'. Close other GPU "
                f"programs (games/browsers) and retry, or use a lower quality "
                f"mode. Underlying error: {str(e)[:200]}") from e
        raise RenderError(f"model '{spec.name}' failed: {type(e).__name__}: {str(e)[:250]}") from e


def release(name: Optional[str] = None) -> None:
    """Drop cached sessions to free VRAM (all, or one by name)."""
    with _lock:
        if name is None:
            _cache.clear()
        else:
            _cache.pop(name, None)


def loaded() -> list[str]:
    with _lock:
        return sorted(_cache)


# ---------------------------------------------------------------- GPU facts
def gpu_info() -> dict[str, Any]:
    """Live GPU facts, measured rather than assumed."""
    out: dict[str, Any] = {
        "available_providers": list(ort.get_available_providers()),
        "loaded_sessions": loaded(),
        "gpu_name": None, "vram_total_mb": None,
        "vram_used_mb": None, "gpu_util_pct": None,
    }
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            name, total, used, util = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            out.update(gpu_name=name, vram_total_mb=int(total),
                       vram_used_mb=int(used), gpu_util_pct=int(util))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return out


def verify_cuda(spec: ModelSpec) -> dict[str, Any]:
    """Prove CUDA actually bound for a given model."""
    sess = get(spec)
    active = sess.get_providers()
    info = gpu_info()
    info.update(model=spec.name, session_providers=active,
                cuda_active=bool(active and active[0] == "CUDAExecutionProvider"))
    return info


def free_vram_mb() -> Optional[int]:
    g = gpu_info()
    if g["vram_total_mb"] is None:
        return None
    return int(g["vram_total_mb"]) - int(g["vram_used_mb"])
