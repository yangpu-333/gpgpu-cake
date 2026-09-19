#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_DIR="$REPO_ROOT/experiments/kda_repro/official_v100"
KDA_WORK_ROOT="${KDA_WORK_ROOT:-$HOME/kda-repro}"
PYTHON_BIN="${KDA_PYTHON:-$KDA_WORK_ROOT/.venv/bin/python}"
DATA_ROOT="${KDA_DATA_ROOT:-$KDA_WORK_ROOT/data/official}"
RUN_ROOT="${KDA_RUN_ROOT:-$KDA_WORK_ROOT/runs}"
CA_BUNDLE="${KDA_CA_BUNDLE:-$HOME/.local/share/ca-certificates/scholar-git-ca-bundle.pem}"
STAGE="${1:-all}"

case "$STAGE" in
  all|decode|prefill) ;;
  *) echo "usage: $0 [all|decode|prefill]" >&2; exit 2 ;;
esac

test -x "$PYTHON_BIN" || { echo "missing Python runtime: $PYTHON_BIN" >&2; exit 2; }
"$PYTHON_BIN" -c 'import torch, triton, safetensors; assert torch.cuda.is_available(); print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))'
mkdir -p "$RUN_ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

SSL_CERT_FILE="$CA_BUNDLE" "$PYTHON_BIN" "$TASK_DIR/download_metadata.py" \
  --root "$DATA_ROOT" --decode-blobs --prefill-blobs

if [[ "$STAGE" == all || "$STAGE" == decode ]]; then
  "$PYTHON_BIN" "$TASK_DIR/evaluate_decode.py" \
    --data "$DATA_ROOT" --output "$RUN_ROOT/decode-$STAMP.json"
fi

if [[ "$STAGE" == all || "$STAGE" == prefill ]]; then
  "$PYTHON_BIN" "$TASK_DIR/evaluate_prefill.py" \
    --data "$DATA_ROOT" --output "$RUN_ROOT/prefill-$STAMP.json"
fi

echo "status: official_v100_run_complete"
echo "results: $RUN_ROOT"
