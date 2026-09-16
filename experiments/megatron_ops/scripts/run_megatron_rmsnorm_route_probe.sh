#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 || $# -lt 1 ]]; then
  echo "usage: bash run_megatron_rmsnorm_route_probe.sh MEGATRON_ROOT [RESULT_DIR]" >&2
  exit 64
fi

MEGATRON_ROOT="$(cd -- "$1" && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${2:-${PROJECT_ROOT}/experiments/megatron_ops/results/megatron-rmsnorm-route-${STAMP}}"

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 65
fi
mkdir -p "${RESULT_DIR}"

export LD_LIBRARY_PATH="/usr/local/iluvatar/lib64:/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
cd "${PROJECT_ROOT}"

python3 -m unittest discover -s experiments/megatron_ops -p 'test_megatron_rmsnorm_route_probe.py' -v \
  2>&1 | tee "${RESULT_DIR}/unit-tests.log"

python3 experiments/megatron_ops/megatron_rmsnorm_route_probe.py \
  --megatron-root "${MEGATRON_ROOT}" \
  --device 0 \
  --sequence-length 8 \
  --micro-batch-size 2 \
  --hidden-size 1024 \
  --num-attention-heads 8 \
  --dtype float16 \
  --output "${RESULT_DIR}/route-probe.json"

printf 'Megatron RMSNorm route probe completed. Results: %s\n' "${RESULT_DIR}"
