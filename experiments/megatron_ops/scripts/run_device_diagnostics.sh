#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${PROJECT_ROOT}/experiments/megatron_ops/results/diagnostic-${STAMP}"
REPORT="${RESULT_DIR}/device-diagnostics.json"

mkdir -p "${RESULT_DIR}"

cd "${PROJECT_ROOT}"
python3 experiments/megatron_ops/device_diagnostics.py --device 0 --output "${REPORT}"
python3 -m json.tool "${REPORT}"
