#!/usr/bin/env bash
set -euo pipefail

# The BI-V150 image needs these vendor runtime directories before importing the
# preinstalled CoreX PyTorch/Triton packages.  This is process-local and does
# not install software or modify shell startup files.
export LD_LIBRARY_PATH="/usr/local/iluvatar/lib64:/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${1:-${PROJECT_ROOT}/experiments/megatron_ops/results/residual-rmsnorm-${STAMP}}"

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 65
fi
mkdir -p "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

python3 -m unittest discover -s experiments/megatron_ops -p 'test_residual_rmsnorm_autotune.py' -v \
  2>&1 | tee "${RESULT_DIR}/unit-tests.log"

python3 experiments/megatron_ops/residual_rmsnorm_autotune.py \
  --mode cpu-plan \
  --output "${RESULT_DIR}/plan.json"

python3 experiments/megatron_ops/residual_rmsnorm_autotune.py \
  --mode gpu \
  --device 0 \
  --shapes "64x768,64x1024,64x4096" \
  --dtype float16 \
  --warmup 10 \
  --samples 9 \
  --launches 30 \
  --output "${RESULT_DIR}/autotune.json"

printf 'Residual Add RMSNorm autotune completed. Results: %s\n' "${RESULT_DIR}"
