#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash run_first_validation.sh MEGATRON_ROOT [RESULT_DIR]" >&2
  exit 64
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MEGATRON_ROOT="$(cd -- "$1" && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${2:-${PROJECT_ROOT}/experiments/megatron_ops/results/first-${STAMP}}"

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 65
fi
mkdir -p "${RESULT_DIR}"

cd "${PROJECT_ROOT}"

python3 experiments/megatron_ops/probe.py \
  --megatron-root "${MEGATRON_ROOT}" \
  --device 0 \
  --output "${RESULT_DIR}/megatron-environment.json"

python3 -m unittest discover \
  -s experiments/basic_validation \
  -p "test_*.py" \
  -v 2>&1 | tee "${RESULT_DIR}/unit-tests.log"

python3 experiments/basic_validation/run.py \
  --mode probe \
  --device 0 \
  --output "${RESULT_DIR}/basic-probe.json"

python3 experiments/basic_validation/run.py \
  --mode gpu \
  --operator add \
  --device 0 \
  --warmup 3 \
  --samples 7 \
  --launches 20 \
  --output "${RESULT_DIR}/gpu-add.json"

python3 experiments/basic_validation/run.py \
  --mode gpu \
  --operator matmul \
  --device 0 \
  --warmup 3 \
  --samples 7 \
  --launches 20 \
  --output "${RESULT_DIR}/gpu-matmul.json"

printf 'First validation completed. Results: %s\n' "${RESULT_DIR}"
