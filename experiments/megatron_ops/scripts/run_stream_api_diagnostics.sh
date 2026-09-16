#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${PROJECT_ROOT}/experiments/megatron_ops/results/stream-api-${STAMP}-$$"
mkdir -p "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

python3 -u experiments/megatron_ops/stream_api_diagnostics.py \
  --device "${1:-0}" --output "${RESULT_DIR}/stream-api-diagnostics.json" \
  2>&1 | tee "${RESULT_DIR}/summary.log"
