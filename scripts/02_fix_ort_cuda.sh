#!/usr/bin/env bash
# onnxruntime 1.30 requires CUDA 13 + cuDNN 9. torch cu128 ships CUDA 12 DLLs.
# Align BOTH on CUDA 13 so they share one runtime, then prove the CUDA EP
# actually binds (not the silent CPU fallback).
set -u
PY=/c/fsw/venv/Scripts/python.exe
UVP=(uv pip install --python "$PY")
exec > >(tee -a /c/fsw/fix_ort.log) 2>&1
echo "############ ORT/CUDA ALIGNMENT $(date) ############"

echo "=== [1/4] torch -> cu130 (CUDA 13 family, must keep sm_120) ==="
"${UVP[@]}" --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -6

echo "=== [2/4] CUDA 13 runtime wheels onnxruntime needs ==="
"${UVP[@]}" nvidia-cudnn-cu13 nvidia-cublas-cu13 nvidia-cufft-cu13 nvidia-curand-cu13 2>&1 | tail -6

echo "=== [3/4] which cublasLt is present now ==="
"$PY" - <<'PYEOF'
import glob, os, site
roots = site.getsitepackages()
hits = []
for r in roots:
    hits += glob.glob(os.path.join(r, "nvidia", "**", "cublasLt64_*.dll"), recursive=True)
    hits += glob.glob(os.path.join(r, "nvidia", "**", "cudnn64_*.dll"), recursive=True)
for h in sorted(set(hits)):
    print("  ", os.path.basename(h), "<-", h)
if not hits: print("   (no nvidia dlls found in site-packages)")
PYEOF

echo "=== [4/4] CUDA EP BIND TEST (must print CUDAExecutionProvider) ==="
"$PY" - <<'PYEOF'
import os, glob, site, sys

# Register every nvidia wheel bin/lib dir on the DLL search path BEFORE ORT loads.
def register_nvidia_dlls():
    added = []
    for r in site.getsitepackages():
        base = os.path.join(r, "nvidia")
        if not os.path.isdir(base): continue
        for sub in glob.glob(os.path.join(base, "*", "bin")) + glob.glob(os.path.join(base, "*", "lib")):
            if os.path.isdir(sub):
                try:
                    os.add_dll_directory(sub); added.append(sub)
                except Exception: pass
    return added

added = register_nvidia_dlls()
print(f"registered {len(added)} nvidia dll dirs")

# Importing torch first also primes CUDA DLLs on Windows.
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda,
          "sm_120:", "sm_120" in torch.cuda.get_arch_list())
except Exception as e:
    print("torch import failed:", e)

import numpy as np, onnxruntime as ort
print("onnxruntime:", ort.__version__)
from onnx import helper, TensorProto
import onnx
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [512, 512])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [512, 512])
g = helper.make_graph([helper.make_node("MatMul", ["X","X"], ["Y"])], "m", [X], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 10
onnx.save(m, "C:/fsw/_probe.onnx")
try:
    s = ort.InferenceSession("C:/fsw/_probe.onnx", providers=["CUDAExecutionProvider"])
    used = s.get_providers()
    print("session providers:", used)
    a = np.random.rand(512,512).astype(np.float32)
    ok = np.allclose(s.run(None, {"X": a})[0], a@a, atol=1e-2)
    print("RESULT_ORT:", "PASS" if used and used[0]=="CUDAExecutionProvider" and ok else "FAIL")
except Exception as e:
    import traceback; traceback.print_exc()
    print("RESULT_ORT: FAIL -", repr(e))
PYEOF
echo "############ DONE $(date) ############"
