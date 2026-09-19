#!/usr/bin/env bash
set -euo pipefail
set +x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_DIR="$REPO_ROOT/experiments/kda_repro/official_v100"
KDA_WORK_ROOT="${KDA_WORK_ROOT:-$HOME/kda-repro}"
PYTHON_BIN="${KDA_PYTHON:-$KDA_WORK_ROOT/.venv/bin/python}"
DATA_ROOT="${KDA_DATA_ROOT:-$KDA_WORK_ROOT/data/official}"
AGENT_WORKSPACE="${KDA_AGENT_WORKSPACE:-$KDA_WORK_ROOT/agent-gdn-decode}"
AGENT_ENV="${KDA_AGENT_ENV:-$HOME/.config/gpgpu-cake/agent.env}"
CA_BUNDLE="${KDA_CA_BUNDLE:-$HOME/.local/share/ca-certificates/scholar-git-ca-bundle.pem}"
ITERATIONS="${1:-10}"

if [[ -f "$AGENT_ENV" ]]; then
  ENV_MODE="$(stat -c '%a' "$AGENT_ENV")"
  if [[ "$ENV_MODE" != "600" && "$ENV_MODE" != "400" ]]; then
    echo "refusing API key file with mode $ENV_MODE; run: chmod 600 $AGENT_ENV" >&2
    exit 2
  fi
  # The file lives outside Git and should have mode 600.
  # shellcheck disable=SC1090
  source "$AGENT_ENV"
fi

: "${KDA_LLM_API_KEY:?set KDA_LLM_API_KEY in $AGENT_ENV}"
: "${KDA_LLM_MODEL:?set KDA_LLM_MODEL in $AGENT_ENV}"
if [[ -z "${KDA_LLM_ENDPOINT:-}" && -z "${KDA_LLM_BASE_URL:-}" ]]; then
  echo "set KDA_LLM_BASE_URL or KDA_LLM_ENDPOINT in $AGENT_ENV" >&2
  exit 2
fi
test -x "$PYTHON_BIN" || { echo "missing Python runtime: $PYTHON_BIN" >&2; exit 2; }

SSL_CERT_FILE="$CA_BUNDLE" "$PYTHON_BIN" "$TASK_DIR/download_metadata.py" \
  --root "$DATA_ROOT" --decode-blobs

SSL_CERT_FILE="$CA_BUNDLE" "$PYTHON_BIN" "$TASK_DIR/agent_controller.py" \
  --data "$DATA_ROOT" --workspace "$AGENT_WORKSPACE" --iterations "$ITERATIONS"
