#!/usr/bin/env bash
# Create the Python runtime used by the generic KDA smoke workflow on V100.
# This intentionally does not install the B200-only FlashInfer contest stack.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: bash setup_v100_torch.sh [VENV_DIR]" >&2
  exit 64
fi

VENV_DIR="${1:-${HOME}/kda-repro/.venv}"
TORCH_VERSION="${KDA_TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${KDA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
CA_BUNDLE="${KDA_CA_BUNDLE:-}"

command -v python3 >/dev/null || {
  echo "python3 is required" >&2
  exit 1
}

if [[ -n "${CA_BUNDLE}" ]]; then
  [[ -r "${CA_BUNDLE}" ]] || {
    echo "KDA_CA_BUNDLE is not readable: ${CA_BUNDLE}" >&2
    exit 1
  }
  export PIP_CERT="${CA_BUNDLE}"
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"

python - <<'PY'
import sys
import torch

print(f"python: {sys.version.split()[0]}")
print(f"torch: {torch.__version__}")
print(f"torch_cuda: {torch.version.cuda}")
print(f"cuda_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA device")
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"capability: {torch.cuda.get_device_capability(0)}")
PY

cat <<EOF
status: v100_torch_ready
venv: ${VENV_DIR}
next: use ${VENV_DIR}/bin/python for the generic KDA smoke workflow
EOF
