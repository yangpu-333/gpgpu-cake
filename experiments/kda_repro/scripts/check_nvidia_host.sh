#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "KDA host check failed: $*" >&2
  exit 1
}

command -v git >/dev/null || fail "git is not installed"
command -v python3 >/dev/null || fail "python3 is not installed"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is not installed or no NVIDIA GPU is exposed"

echo "== NVIDIA GPUs =="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

echo "== CUDA compiler =="
if command -v nvcc >/dev/null; then
  nvcc --version
else
  echo "WARNING: nvcc is unavailable; select a CUDA devel image before compiling candidates."
fi

echo "== Nsight Compute =="
if command -v ncu >/dev/null; then
  ncu --version | head -n 2
else
  echo "WARNING: ncu is unavailable; install or select an image that includes Nsight Compute before profiling."
fi

echo "== Python CUDA visibility =="
python3 - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"PyTorch unavailable: {type(exc).__name__}: {exc}")
else:
    print("torch:", torch.__version__)
    print("torch CUDA build:", torch.version.cuda)
    print("torch cuda available:", torch.cuda.is_available())
    print("torch device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("torch device 0:", torch.cuda.get_device_name(0))
PY

echo "status: host_check_complete"
