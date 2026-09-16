#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: bash prepare_kda_sources.sh [WORKSPACE_ROOT]" >&2
  exit 64
fi

WORKSPACE_ROOT="${1:-${HOME}/kda-repro}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${SCRIPT_DIR}/../source-lock.json"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
[[ -f "${LOCK_FILE}" ]] || { echo "missing source lock: ${LOCK_FILE}" >&2; exit 1; }

clone_pinned() {
  local name="$1"
  local destination="$2"
  local url="$3"
  local revision="$4"
  if [[ -e "${destination}" && ! -d "${destination}/.git" ]]; then
    echo "destination exists but is not a Git checkout: ${destination}" >&2
    exit 65
  fi
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --recurse-submodules "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --tags origin
  git -C "${destination}" checkout --detach "${revision}"
  git -C "${destination}" submodule update --init --recursive
  echo "prepared ${name}: $(git -C "${destination}" rev-parse HEAD)"
}

mkdir -p "${WORKSPACE_ROOT}/reference" "${WORKSPACE_ROOT}/release" \
  "${WORKSPACE_ROOT}/third_party" "${WORKSPACE_ROOT}/workspaces"

mapfile -t SOURCES < <(python3 - "${LOCK_FILE}" <<'PY'
import json
import sys
for source in json.load(open(sys.argv[1], encoding="utf-8"))["sources"]:
    print("\t".join((source["name"], source["url"], source["revision"])))
PY
)

for record in "${SOURCES[@]}"; do
  IFS=$'\t' read -r name url revision <<<"${record}"
  case "${name}" in
    kda) destination="${WORKSPACE_ROOT}/reference/${name}" ;;
    mlsys2026-flashinfer-contest) destination="${WORKSPACE_ROOT}/release/${name}" ;;
    flashinfer-bench|DeepGEMM) destination="${WORKSPACE_ROOT}/third_party/${name}" ;;
    flashinfer-bench-starter-kit) destination="${WORKSPACE_ROOT}/workspaces/${name}" ;;
    *) echo "unrecognized locked source: ${name}" >&2; exit 65 ;;
  esac
  clone_pinned "${name}" "${destination}" "${url}" "${revision}"
done

cat <<EOF
status: sources_prepared
workspace: ${WORKSPACE_ROOT}
next: follow ${WORKSPACE_ROOT}/release/mlsys2026-flashinfer-contest/docs/reproduction.md
note: do not clone the final solution repository into a fresh agent workspace.
EOF
