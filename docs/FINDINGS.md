# Findings

Measured on this machine, 2026-09-21. These drove the architecture, and several
contradict what the documentation and the wider web say.

> **Status note (2026-09-21).** Sections marked *HISTORICAL* below describe the
> original FaceFusion-CLI architecture, which has since been replaced by a
> direct ONNX Runtime + OpenCV + ffmpeg pipeline (`app/swapper.py`). They are
> kept because the measurements are still true and explain why the current
> design looks the way it does. The GPU and Cloudflare findings still apply.

## GPU: onnxruntime needs CUDA 13, not 12

The first install used torch `cu128` on the assumption that onnxruntime-gpu was
built against the CUDA 12 wheel family. It is not — onnxruntime 1.29/1.30 want
CUDA 13:

    Error loading onnxruntime_providers_cuda.dll
    which depends on "cublasLt64_13.dll" which is missing

**It failed silently.** `get_available_providers()` still listed
`CUDAExecutionProvider`, but the actual session fell back to CPU:

    available providers : [..., 'CUDAExecutionProvider', 'CPUExecutionProvider']
    session providers   : ['CPUExecutionProvider']     <-- silent fallback

This is the single most dangerous failure mode: everything looks fine and a
render takes hours instead of minutes. **Always assert
`session.get_providers()[0] == 'CUDAExecutionProvider'`**, never just check
`get_available_providers()`.

Fix: align both on CUDA 13 (`torch` from the `cu130` index).

Verified working:

    torch 2.14.0+cu130, cuda 13.0, arch list includes sm_120
    onnxruntime-gpu 1.29.0
    session providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']

## The CUDA DLLs live in torch/lib, and need preloading

*(Still current: the venv's `sitecustomize.py` preload is what makes ORT find
CUDA 13, and the ONNX pipeline depends on it exactly as FaceFusion did.)*

All CUDA 13 runtime DLLs ship inside `site-packages/torch/lib`. FaceFusion is
pure-ONNX and never imports torch, so ORT could not find them.

`os.add_dll_directory()` alone does **not** fix this — onnxruntime resolves its
provider DLL's dependencies in a way that ignores the added directory. torch
works because it *preloads* the libraries with `ctypes`.

Fix: `app`-independent `sitecustomize.py` in site-packages preloads them via
`ctypes.WinDLL` at interpreter startup. This fixes every process in the venv
without touching FaceFusion's source — which matters, because FaceFusion hashes
`content_analyser.py` at startup and exits 2 if it changed.

## Cloudflare upload limits: the 100 MB figure is wrong

Widely stated (including by Cloudflare's own docs) as a hard 100 MB request body
cap on the free plan. Measured through a Quick Tunnel:

| Size | Result |
|---|---|
| 95 MB | OK |
| 99 MB | OK |
| 105 MB | OK |
| 150 MB | OK |
| 300 MB | OK (`received == declared`) |
| 600 MB | **413 Payload Too Large** |

So the real ceiling is somewhere between 300–600 MB, not 100 MB.

We chunk at 8 MB anyway, because the **binding constraint is time, not size**:
Cloudflare's proxy read timeout is 125 s (HTTP 524). A 300 MB upload on a normal
home uplink exceeds that regardless of any size cap. Chunking also gives
per-chunk retry over a flaky link.

## No Server-Sent Events on Quick Tunnels

*(Still current: this is why progress is JSON polling.)*

Cloudflare states Quick Tunnels do not support SSE. This has a consequence
beyond progress reporting: **FaceFusion's own Gradio UI cannot be reused**,
because Gradio delivers queue results over SSE at `/queue/data`. It would hang
over a trycloudflare URL.

That is why this project has its own FastAPI + vanilla JS front end rather than
just proxying FaceFusion's built-in UI. Progress is 1.5 s JSON polling.

## Progress parsing — *HISTORICAL*

*(No longer applies. Progress is now `frames_done / total_frames`, reported
directly by the in-process pipeline; nothing is parsed from a console.)*

FaceFusion's tqdm bar writes to **stderr**, redrawing with carriage returns and
no newline. `readline()` blocks until the bar completes, so progress jumps
0% → 100%. Read with `read1()` and split on both CR and LF.

Also: the bar is suppressed at `--log-level warn|error`, so `info` is pinned
explicitly. And `--workflow-strategy` defaults to `memory`, meaning frames never
touch disk — so the common "count files in the temp dir" progress trick reports
0% forever.

## Misc

- cloudflared prints its URL on **stderr**, and prints it *before* the hostname
  is routable. `start.ps1` polls `/healthz` through the tunnel before showing
  the link, otherwise the first click 404s.
- `install.py` uses `shutil.which('pip')`, so the venv's Scripts dir must be
  first on PATH or it installs into the wrong interpreter. uv venvs have no pip
  by default — it must be added.
- `install.py` takes onnxruntime as a **positional** arg (`cuda@13`), and
  requires `--skip-conda` on a machine without conda.
- Default swapper model is `hyperswap_1a_256`, not `inswapper_128`: higher
  resolution (256 vs 128) and avoids InsightFace's non-commercial model licence.
- `h264_nvenc` is available, keeping the final encode on the GPU.
