#!/bin/bash
# Gate 0 orchestrator: run each (source, task) case in its OWN fresh process.
#
# Why: OmniGibson / Isaac Sim allows only one simulator instance per Python
# process. validate_operation_action_causality.py loops over cases in a single
# process, so the 2nd env build hits the locked ENABLE_TRANSITION_RULES macro
# and segfaults (observed 2026-07-16). Running one case per process avoids it.
# The single-case path itself (--source-id N --task T) is already validated.
#
# Usage: run_gate0_isolated.sh <output_root> [max_steps]
#   max_steps omitted -> full episode (real Gate 0)
#   max_steps set     -> short smoke to prove process isolation
set -u

REPO=/root/autodl-tmp/hrc_pipeline_traintest
PY=/root/autodl-tmp/behavior-venv/bin/python
SCRIPT="$REPO/scripts/evaluation/validate_operation_action_causality.py"
AGG="$REPO/scripts/evaluation/aggregate_gate0.py"

OUT_ROOT="${1:?usage: run_gate0_isolated.sh <output_root> [max_steps]}"
MAX_STEPS="${2:-}"

# Must match pi05 manifest source_splits.val (verified 2026-07-16).
SOURCES="0 2 15 18 63"
TASKS="pickup radio"

mkdir -p "$OUT_ROOT"
echo "GATE0_ORCH_START $(date -u +%Y-%m-%dT%H:%M:%SZ) out=$OUT_ROOT max_steps=${MAX_STEPS:-none}"

n=0
for sid in $SOURCES; do
  for task in $TASKS; do
    n=$((n + 1))
    case_log="$OUT_ROOT/case_$(printf '%02d' "$n")_src${sid}_${task}.log"
    echo "=== [orch] case $n/10: source=$sid task=$task -> $case_log ==="
    extra=""
    [ -n "$MAX_STEPS" ] && extra="--max-steps $MAX_STEPS"
    cd "$REPO"
    PYTHONUNBUFFERED=1 "$PY" "$SCRIPT" \
      --split val --source-id "$sid" --task "$task" \
      --output-root "$OUT_ROOT" $extra \
      > "$case_log" 2>&1
    rc=$?
    echo "=== [orch] case $n/10 source=$sid task=$task EXIT=$rc ==="
    # let the GPU/driver fully release before the next simulator boot
    sleep 5
  done
done

echo "GATE0_ORCH_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
PYTHONUNBUFFERED=1 "$PY" "$AGG" --output-root "$OUT_ROOT"
