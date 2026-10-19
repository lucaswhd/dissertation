#!/bin/bash
# One-key, resumable driver for the E5 π0.5 continuous CLEAN pilot.
#
# Stages (in order):
#   manual_qc -> tests -> export -> validate -> normstats -> preflight
#   -> e0 -> train5k -> offline -> rollout
#
# Usage:
#   run_pi05_continuous_clean_pipeline.sh --through <data|e0|train|offline|rollout> [--resume]
#
# --through picks how far to go (default: rollout). --resume skips stages whose
# output already exists AND passes its check. Default is NOT --overwrite: a stage
# with pre-existing output errors unless --resume lets it be skipped. Each stage
# writes its own log; any gate failure exits non-zero immediately. Training,
# policy server and OmniGibson never share the GPU concurrently.
set -u -o pipefail

OPENPI=/root/autodl-tmp/openpi_hrc
PIPE=/root/autodl-tmp/hrc_pipeline_traintest
PYO="$OPENPI/.venv/bin/python"
DATA_ROOT="$PIPE/outputs/hrc_batch_final"
QC="$DATA_ROOT/rescan_qc.json"
BASE_MANIFEST=/root/autodl-tmp/lerobot_datasets/hrc_pi05_v0/manifest.json
DATASET=/root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_clean_v0
MANUAL_QC="${DATASET}_manual_qc.json"
CKPT_DIR="$OPENPI/checkpoints/pi05_hrc_e5_continuous_main/hrc_e5_continuous_seed42"
CKPT_5K="$CKPT_DIR/5000"
E0_CKPT="$OPENPI/checkpoints/pi05_hrc_e5_continuous_e0/hrc_e5_continuous_e0_seed42/1000"
OFFLINE_DIR="$OPENPI/outputs/hrc_e5_continuous"
ROLLOUT_ROOT="$PIPE/outputs/pi05_continuous_clean/closed_loop"
LOGDIR="$PIPE/logs/pi05_continuous_clean"
mkdir -p "$LOGDIR"

# openpi jobs share these; keep training/serve/sim off the GPU at the same time.
export PYTHONPATH="$OPENPI/src"
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/root/autodl-tmp/.cache/openpi
export JAX_COMPILATION_CACHE_DIR=/root/autodl-tmp/.cache/jax
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_DIR=/root/autodl-tmp/wandb
export WANDB_MODE=offline
export TMPDIR=/root/autodl-tmp/tmp

THROUGH=rollout
RESUME=0
while [ $# -gt 0 ]; do
  case "$1" in
    --through) THROUGH="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) echo "unknown arg: $1"; exit 64 ;;
  esac
done

STAGES=(manual_qc tests export validate normstats preflight e0 train5k offline rollout)
case "$THROUGH" in
  data)    STOP=normstats ;;
  e0)      STOP=e0 ;;
  train)   STOP=train5k ;;
  offline) STOP=offline ;;
  rollout) STOP=rollout ;;
  *) echo "invalid --through: $THROUGH (data|e0|train|offline|rollout)"; exit 64 ;;
esac

log()  { echo "[$(date -u +%H:%M:%S)] $*"; }
die()  { echo "PIPELINE_FAIL at stage=$1: $2"; exit 1; }
json_ok() { "$PYO" -c "$1" 2>/dev/null; }  # exits 0/1 from the snippet's SystemExit

# ---- per-stage "already done and passing?" checks (for --resume) ------------ #
check_manual_qc() { [ -f "$MANUAL_QC" ] && json_ok "import json,sys; sys.exit(0 if json.load(open('$MANUAL_QC')).get('source_25')=='PASS' else 1)"; }
check_tests()     { [ -f "$LOGDIR/tests.log" ] && grep -qE "[0-9]+ passed" "$LOGDIR/tests.log" && ! grep -qiE "failed|error" "$LOGDIR/tests.log"; }
check_export()    { [ -f "$DATASET/export_report.json" ] && json_ok "import json,sys; r=json.load(open('$DATASET/export_report.json')); sys.exit(0 if r['splits']['train']['num_valid_anchors']==128336 else 1)"; }
check_validate()  { [ -f "$DATASET/validation_report.json" ] && json_ok "import json,sys; sys.exit(0 if json.load(open('$DATASET/validation_report.json'))['result']=='PASS' else 1)"; }
check_normstats() { [ -f "$OPENPI/assets/hrc/main_continuous_clean_natural/norm_stats.json" ]; }
check_preflight() { [ -f "$LOGDIR/preflight.ok" ]; }
check_e0()        { [ -d "$E0_CKPT" ]; }
check_train5k()   { [ -d "$CKPT_5K" ]; }
check_offline()   { [ -f "$OFFLINE_DIR/offline_metrics.json" ]; }
check_rollout()   { [ -f "$ROLLOUT_ROOT/closed_loop_report.json" ] && json_ok "import json,sys; r=json.load(open('$ROLLOUT_ROOT/closed_loop_report.json')); sys.exit(0 if not r['missing_runs'] and not r['invalid_runs'] else 1)"; }

# ---- stage runners --------------------------------------------------------- #
run_manual_qc() {
  if [ ! -f "$MANUAL_QC" ]; then
    echo "等待人工视频复核 (source 25/36): 缺 $MANUAL_QC" | tee "$LOGDIR/manual_qc.log"
    return 1
  fi
  check_manual_qc || { echo "source_25 未标记 PASS in $MANUAL_QC" | tee "$LOGDIR/manual_qc.log"; return 1; }
  echo "manual_qc OK: $MANUAL_QC" | tee "$LOGDIR/manual_qc.log"
}

run_tests() {
  ( cd "$OPENPI" && "$PYO" -m pytest -q \
      src/openpi/training/hrc_continuous_data_test.py \
      src/openpi/training/hrc_continuous_export_test.py \
      src/openpi/training/hrc_continuous_validation_test.py \
      src/openpi/training/hrc_dataset_test.py \
      src/openpi/training/hrc_config_test.py \
      scripts/hrc/evaluate_hrc_pi05_continuous_test.py ) >"$LOGDIR/tests.log" 2>&1
}

run_export() {
  ( cd "$OPENPI" && "$PYO" scripts/hrc/export_hrc_continuous.py \
      --data-root "$DATA_ROOT" --qc-report "$QC" --base-manifest "$BASE_MANIFEST" \
      --output-root "$DATASET" --manual-qc "$MANUAL_QC" ) >"$LOGDIR/export.log" 2>&1
}

run_validate() {
  ( cd "$OPENPI" && "$PYO" -m scripts.hrc.validate_hrc_continuous \
      --dataset-root "$DATASET" ) >"$LOGDIR/validate.log" 2>&1
}

run_normstats() {
  ( cd "$OPENPI" && "$PYO" scripts/hrc/compute_hrc_norm_stats.py \
      --config-name pi05_hrc_e5_continuous_main --overwrite ) >"$LOGDIR/normstats.log" 2>&1
}

run_preflight() {
  ( cd "$OPENPI" && "$PYO" -m scripts.hrc.run_training_preflight \
      --config-name pi05_hrc_e5_continuous_main --physical-batch-size 2 \
      --steps 10 --max-memory-mib 22528 ) >"$LOGDIR/preflight.log" 2>&1 \
    && touch "$LOGDIR/preflight.ok"
}

run_e0() {
  ( cd "$OPENPI" && "$PYO" scripts/b1k/train_b1k.py pi05_hrc_e5_continuous_e0 \
      --exp_name=hrc_e5_continuous_e0_seed42 --overwrite ) >"$LOGDIR/e0.log" 2>&1
}

run_train5k() {
  ( cd "$OPENPI" && "$PYO" scripts/b1k/train_b1k.py pi05_hrc_e5_continuous_main \
      --exp_name=hrc_e5_continuous_seed42 --overwrite ) >"$LOGDIR/train5k.log" 2>&1
}

run_offline() {
  ( cd "$OPENPI" \
    && "$PYO" -m scripts.hrc.evaluate_hrc_pi05_continuous --config pi05_hrc_e5_continuous_main \
         --checkpoint "$CKPT_5K" --dataset-root "$DATASET/val" --mode metrics \
         --max-anchors-per-context 300 --seed 42 --output-dir "$OFFLINE_DIR" \
    && "$PYO" -m scripts.hrc.evaluate_hrc_pi05_continuous --config pi05_hrc_e5_continuous_main \
         --checkpoint "$CKPT_5K" --dataset-root "$DATASET/val" --mode render \
         --source-id 15 --replan-every 8 --output-dir "$OFFLINE_DIR/source0015" \
  ) >"$LOGDIR/offline.log" 2>&1
}

# Closed loop: generate the plan, then per model group start the server (trap-
# cleaned), run its rollouts, stop the server; finally summarize.
run_rollout() {
  local plan="$ROLLOUT_ROOT/run_plan.json"
  ( "$PYO" "$PIPE/scripts/evaluation/prepare_pi05_continuous_runs.py" \
      --output-root "$ROLLOUT_ROOT" ) >"$LOGDIR/rollout.log" 2>&1 || return 1
  local groups; groups=$("$PYO" -c "import json;print(len(json.load(open('$plan'))['model_groups']))") || return 1
  for gi in $(seq 0 $((groups - 1))); do
    local server_cmd server_wd; server_cmd=$("$PYO" -c "import json;print(json.load(open('$plan'))['model_groups'][$gi]['server_command'])")
    server_wd=$("$PYO" -c "import json;print(json.load(open('$plan'))['model_groups'][$gi]['server_workdir'])")
    # Clear any stale server holding the port first (a prior run's server can
    # survive), else this group's sims would query the wrong checkpoint.
    pkill -9 -f "serve_hrc_pi05_rollout.py" 2>/dev/null; sleep 3
    # The policy server must NOT preallocate the GPU: it has to coexist with the
    # OmniGibson RTX renderer on one 4090. MEM_FRACTION=0.90 would starve the sim.
    ( cd "$server_wd"; export XLA_PYTHON_CLIENT_PREALLOCATE=false; unset XLA_PYTHON_CLIENT_MEM_FRACTION; eval "$server_cmd" ) >>"$LOGDIR/rollout_server_$gi.log" 2>&1 &
    trap 'pkill -9 -f "serve_hrc_pi05_rollout.py" 2>/dev/null' EXIT
    sleep 60  # let the JAX server load + bind the port
    local nruns; nruns=$("$PYO" -c "import json;print(len(json.load(open('$plan'))['model_groups'][$gi]['runs']))")
    for ri in $(seq 0 $((nruns - 1))); do
      local rcmd; rcmd=$("$PYO" -c "import json;print(json.load(open('$plan'))['model_groups'][$gi]['runs'][$ri]['command'])")
      ( cd "$PIPE" && eval "$rcmd" ) >>"$LOGDIR/rollout.log" 2>&1 || log "rollout run g$gi r$ri returned non-zero"
    done
    pkill -9 -f "serve_hrc_pi05_rollout.py" 2>/dev/null; sleep 3; trap - EXIT
  done
  ( "$PYO" "$PIPE/scripts/evaluation/summarize_pi05_closed_loop.py" \
      --plan "$plan" --output "$ROLLOUT_ROOT/closed_loop_report.json" ) >>"$LOGDIR/rollout.log" 2>&1
}

# ---- driver ---------------------------------------------------------------- #
log "pipeline start: through=$THROUGH resume=$RESUME"
for stage in "${STAGES[@]}"; do
  if [ "$RESUME" -eq 1 ] && "check_$stage"; then
    log "SKIP $stage (already done & passing)"
  else
    log "RUN  $stage"
    if ! "run_$stage"; then die "$stage" "run failed (see $LOGDIR/$stage.log)"; fi
    if ! "check_$stage"; then die "$stage" "post-check failed (see $LOGDIR/$stage.log)"; fi
    log "PASS $stage"
  fi
  [ "$stage" = "$STOP" ] && break
done
log "pipeline done through=$STOP"
