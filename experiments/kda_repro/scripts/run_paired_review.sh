#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <decode|prefill> <candidate-id> <baseline-id> [additional-runs]" >&2
  exit 2
fi

TASK="$1"
CANDIDATE_ID="$2"
BASELINE_ID="$3"
ADDITIONAL_RUNS="${4:-2}"

case "$TASK" in
  decode)
    EVALUATOR="evaluate_decode.py"
    DEFAULT_WORKSPACE="agent-gdn-decode"
    ;;
  prefill)
    EVALUATOR="evaluate_prefill.py"
    DEFAULT_WORKSPACE="agent-gdn-prefill"
    ;;
  *)
    echo "task must be decode or prefill" >&2
    exit 2
    ;;
esac

[[ "$ADDITIONAL_RUNS" =~ ^[0-9]+$ ]] && (( ADDITIONAL_RUNS >= 2 )) || {
  echo "additional-runs must be an integer of at least 2" >&2
  exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_DIR="$REPO_ROOT/experiments/kda_repro/official_v100"
KDA_WORK_ROOT="${KDA_WORK_ROOT:-$HOME/kda-repro}"
PYTHON_BIN="${KDA_PYTHON:-$KDA_WORK_ROOT/.venv/bin/python}"
DATA_ROOT="${KDA_DATA_ROOT:-$KDA_WORK_ROOT/data/official}"
WORKSPACE="${KDA_REVIEW_WORKSPACE:-$KDA_WORK_ROOT/$DEFAULT_WORKSPACE}"
RUNS_DIR="$WORKSPACE/runs"
CANDIDATE_SOURCE="$WORKSPACE/candidates/$CANDIDATE_ID/candidate.py"
BASELINE_SOURCE="$WORKSPACE/candidates/$BASELINE_ID/candidate.py"
INITIAL_REPORT="$RUNS_DIR/$CANDIDATE_ID-full.json"

test -x "$PYTHON_BIN" || { echo "missing Python runtime: $PYTHON_BIN" >&2; exit 2; }
test -f "$CANDIDATE_SOURCE" || { echo "missing candidate: $CANDIDATE_SOURCE" >&2; exit 2; }
test -f "$BASELINE_SOURCE" || { echo "missing baseline: $BASELINE_SOURCE" >&2; exit 2; }
test -f "$INITIAL_REPORT" || { echo "missing initial paired report: $INITIAL_REPORT" >&2; exit 2; }

REPORTS=("$INITIAL_REPORT")
for run in $(seq 1 "$ADDITIONAL_RUNS"); do
  REPORT="$RUNS_DIR/$CANDIDATE_ID-paired-$run.json"
  "$PYTHON_BIN" "$TASK_DIR/$EVALUATOR" \
    --data "$DATA_ROOT" \
    --output "$REPORT" \
    --candidate-source "$CANDIDATE_SOURCE" \
    --baseline-source "$BASELINE_SOURCE" \
    --timing-repeats 21
  REPORTS+=("$REPORT")
done

REVIEW_REPORT="$RUNS_DIR/$CANDIDATE_ID-paired-review.json"
"$PYTHON_BIN" "$TASK_DIR/review_paired_results.py" \
  --workspace "$WORKSPACE" \
  --candidate-id "$CANDIDATE_ID" \
  --reports "${REPORTS[@]}" \
  > "$REVIEW_REPORT"

cat "$REVIEW_REPORT"
echo "review: $REVIEW_REPORT"
