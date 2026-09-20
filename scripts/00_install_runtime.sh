#!/usr/bin/env bash
# Installs + EMPIRICALLY VERIFIES the GPU runtime for Blackwell (sm_120).
# Order matters: onnxruntime-gpu first (it is the real bottleneck for ONNX swap
# models and is pickiest about CUDA), then torch on the SAME cu12 family so the
# shared nvidia-* wheels do not conflict.
set -u
PY=/c/fsw/venv/Scripts/python.exe
LOG=/c/fsw/install.log
exec > >(tee -a "$LOG") 2>&1
echo "################ RUNTIME INSTALL $(date) ################"

# uv venvs ship without pip; use `uv pip` against the venv interpreter (also far faster).
UVP=(uv pip install --python "$PY")

echo "=== [1/5] base tooling ==="
"${UVP[@]}" setuptools wheel 2>&1 | tail -3

echo "=== [2/5] onnxruntime-gpu (CUDA 12 family) ==="
"${UVP[@]}" "onnxruntime-gpu" "onnx" "numpy" 2>&1 | tail -8

echo "=== [3/5] torch cu128 (Blackwell sm_120 kernels) ==="
"${UVP[@]}" torch torchvision --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -8

echo "=== [4/5] ONNXRUNTIME GPU VERIFICATION ==="
"$PY" - <<'PYEOF'
import sys
try:
    import onnxruntime as ort
    print("onnxruntime version :", ort.__version__)
    print("available providers :", ort.get_available_providers())
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("RESULT_ORT: FAIL - CUDA EP not available")
        sys.exit(0)
    # Build a tiny model and force it onto CUDA to prove kernels load.
    import numpy as np
    from onnx import helper, TensorProto
    import onnx
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [256, 256])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [256, 256])
    node = helper.make_node("MatMul", ["X", "X"], ["Y"])
    g = helper.make_graph([node], "m", [X], [Y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.save(m, "C:/fsw/_probe.onnx")
    so = ort.SessionOptions()
    s = ort.InferenceSession("C:/fsw/_probe.onnx", so, providers=["CUDAExecutionProvider"])
    used = s.get_providers()
    print("session providers   :", used)
    a = np.random.rand(256, 256).astype(np.float32)
    r = s.run(None, {"X": a})[0]
    ok = np.allclose(r, a @ a, atol=1e-2)
    print("matmul correct      :", ok)
    print("RESULT_ORT:", "PASS" if (used and used[0] == "CUDAExecutionProvider" and ok) else "FAIL")
except Exception as e:
    import traceback; traceback.print_exc()
    print("RESULT_ORT: FAIL -", repr(e))
PYEOF

echo "=== [5/5] TORCH sm_120 VERIFICATION ==="
"$PY" - <<'PYEOF'
try:
    import torch
    print("torch version   :", torch.__version__)
    print("cuda available  :", torch.cuda.is_available())
    print("cuda build ver  :", torch.version.cuda)
    print("arch list       :", torch.cuda.get_arch_list())
    if not torch.cuda.is_available():
        print("RESULT_TORCH: FAIL - no cuda"); raise SystemExit
    cc = torch.cuda.get_device_capability(0)
    print("device          :", torch.cuda.get_device_name(0))
    print("compute cap     :", cc, "(Blackwell expects (12,0))")
    sm = f"sm_{cc[0]}{cc[1]}"
    print("needs kernel    :", sm, "present:", sm in torch.cuda.get_arch_list())
    # Real GPU math — proves kernels actually execute, not just that init succeeded.
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    torch.cuda.synchronize()
    print("gpu matmul ok   :", y == y)  # NaN check
    free, total = torch.cuda.mem_get_info()
    print(f"vram free/total : {free/1e9:.1f} / {total/1e9:.1f} GB")
    print("RESULT_TORCH: PASS")
except Exception as e:
    import traceback; traceback.print_exc()
    print("RESULT_TORCH: FAIL -", repr(e))
PYEOF
echo "################ DONE $(date) ################"
