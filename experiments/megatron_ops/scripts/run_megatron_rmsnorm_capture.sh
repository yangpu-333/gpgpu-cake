#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: bash run_megatron_rmsnorm_capture.sh MEGATRON_ROOT [RESULT_DIR] -- COMMAND [ARG ...]" >&2
  exit 64
fi

MEGATRON_ROOT="$(cd -- "$1" && pwd)"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ "${1:-}" == "--" ]]; then
  RESULT_DIR="${PROJECT_ROOT}/experiments/megatron_ops/results/megatron-rmsnorm-capture-${STAMP}"
else
  RESULT_DIR="$1"
  shift
fi
if [[ "${1:-}" != "--" ]]; then
  echo "missing -- before the existing Megatron launch command" >&2
  exit 64
fi
shift
if [[ $# -eq 0 ]]; then
  echo "missing Megatron launch command" >&2
  exit 64
fi
if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 65
fi
mkdir -p "${RESULT_DIR}"

# This process-local ordering is the BI-V150 setting proven by the previous
# PyTorch/Triton validation.  It does not install software or alter the image.
export LD_LIBRARY_PATH="/usr/local/iluvatar/lib64:/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export MEGATRON_ROOT
export PYTHONPATH="${PROJECT_ROOT}/experiments/megatron_ops/instrumentation:${PYTHONPATH:-}"
export CAKE_RMSNORM_CAPTURE=1
export CAKE_RMSNORM_CAPTURE_OUTPUT_DIR="${RESULT_DIR}"
export CAKE_RMSNORM_CAPTURE_MAX_RECORDS="${CAKE_RMSNORM_CAPTURE_MAX_RECORDS:-64}"

cd "${PROJECT_ROOT}"
python3 experiments/megatron_ops/probe.py \
  --megatron-root "${MEGATRON_ROOT}" \
  --device 0 \
  --output "${RESULT_DIR}/environment.json"

set +e
"$@"
COMMAND_CODE=$?
set -e

set +e
python3 experiments/megatron_ops/summarize_rmsnorm_capture.py \
  --input-dir "${RESULT_DIR}" \
  --output "${RESULT_DIR}/rmsnorm-capture-summary.json"
SUMMARY_CODE=$?
set -e

if [[ ${COMMAND_CODE} -ne 0 ]]; then
  echo "Megatron command failed with exit code ${COMMAND_CODE}; capture reports, if any, were preserved in ${RESULT_DIR}" >&2
  exit "${COMMAND_CODE}"
fi
if [[ ${SUMMARY_CODE} -ne 0 ]]; then
  echo "Megatron command completed but no target RMSNorm calls were captured. See ${RESULT_DIR}" >&2
  exit "${SUMMARY_CODE}"
fi
printf 'Megatron RMSNorm capture completed. Results: %s\n' "${RESULT_DIR}"
