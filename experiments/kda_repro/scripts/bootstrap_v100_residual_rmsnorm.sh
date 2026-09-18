#!/usr/bin/env bash
# Create an isolated task workspace from the V100 KDA workflow template.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: bash bootstrap_v100_residual_rmsnorm.sh [TASK_WORKSPACE]" >&2
  exit 64
fi

TASK_WORKSPACE="${1:-${HOME}/kda-repro/workspaces/v100-residual-rmsnorm}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/../v100_residual_rmsnorm"

[[ -d "${TEMPLATE_DIR}" ]] || {
  echo "missing task template: ${TEMPLATE_DIR}" >&2
  exit 1
}

if [[ -e "${TASK_WORKSPACE}" ]] && [[ -n "$(find "${TASK_WORKSPACE}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "task workspace is not empty: ${TASK_WORKSPACE}" >&2
  exit 65
fi

mkdir -p "${TASK_WORKSPACE}"
cp -a "${TEMPLATE_DIR}/." "${TASK_WORKSPACE}/"
mkdir -p "${TASK_WORKSPACE}/runs" "${TASK_WORKSPACE}/outputs" "${TASK_WORKSPACE}/profile"

cat <<EOF
status: task_workspace_ready
workspace: ${TASK_WORKSPACE}
validate: cd ${TASK_WORKSPACE} && ${HOME}/kda-repro/.venv/bin/python validate.py
benchmark: cd ${TASK_WORKSPACE} && ${HOME}/kda-repro/.venv/bin/python benchmark.py
EOF
