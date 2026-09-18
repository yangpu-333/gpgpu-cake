#!/usr/bin/env bash
# Run and record the first KDA baseline in an isolated V100 task workspace.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: bash run_v100_residual_rmsnorm_baseline.sh [TASK_WORKSPACE]" >&2
  exit 64
fi

TASK_WORKSPACE="${1:-${HOME}/kda-repro/workspaces/v100-residual-rmsnorm}"
VENV_DIR="${KDA_VENV_DIR:-${HOME}/kda-repro/.venv}"
PYTHON="${VENV_DIR}/bin/python"

[[ -f "${TASK_WORKSPACE}/validate.py" ]] || {
  echo "task workspace is missing: ${TASK_WORKSPACE}" >&2
  exit 1
}
[[ -x "${PYTHON}" ]] || {
  echo "V100 PyTorch environment is missing: ${PYTHON}" >&2
  exit 1
}

cd "${TASK_WORKSPACE}"
mkdir -p runs
RUN_ID="baseline-$(date -u +%Y%m%dT%H%M%SZ)"

"${PYTHON}" validate.py | tee "runs/${RUN_ID}-validation.log"
"${PYTHON}" benchmark.py | tee "runs/${RUN_ID}-benchmark.log"

"${PYTHON}" - "${RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

run_id = sys.argv[1]
record = {
    "id": "baseline-pytorch",
    "event": "measured",
    "status": "validated",
    "implementation": "src/reference.py",
    "evidence": [
        f"runs/{run_id}-validation.log",
        f"runs/{run_id}-benchmark.log",
        "benchmark.csv",
    ],
    "reason": "Forward and backward validation passed; median CUDA-event measurements recorded.",
}
with Path("candidates.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
PY

cat <<EOF
status: baseline_recorded
workspace: ${TASK_WORKSPACE}
run: ${RUN_ID}
EOF
