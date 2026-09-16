#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${PROJECT_ROOT}/experiments/megatron_ops/results/runtime-${STAMP}-$$"
mkdir -p "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

# Visibility mappings and the parent shell environment stay as supplied by the platform.
python3 -u experiments/megatron_ops/runtime_diagnostics.py \
  --device "${1:-0}" --followup --output "${RESULT_DIR}/runtime-diagnostics.json" \
  2>&1 | tee "${RESULT_DIR}/summary.log"
