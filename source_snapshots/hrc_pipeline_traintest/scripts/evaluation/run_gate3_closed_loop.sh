#!/bin/bash
# Gate 3 orchestrator: for EACH model group (pi05_base_hrc_init, then e4_5k) bring
# up its server, run its 10 rollouts sequentially, tear the server down, wait for
# the GPU to free, switch to the next group. Then summarize all 20 runs.
#
# 4090 cannot host two servers at once -> strict group-serial. preallocate=false so
# the JAX server and OmniGibson share VRAM. Failed runs are kept (no deletion).
set -u

OPENPI=/root/autodl-tmp/openpi_hrc
PIPE=/root/autodl-tmp/hrc_pipeline_traintest
PLAN=$PIPE/outputs/pi05_closed_loop/gate3/run_plan.json
LOGDIR=$PIPE/logs/pi05_closed_loop/gate3
mkdir -p "$LOGDIR" "$OPENPI/logs/hrc/closed_loop"

jq_py() { python3 -c "import json,sys; print(json.load(open('$PLAN'))$1)"; }

echo "GATE3_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
NGROUPS=$(jq_py "['model_groups'].__len__()")
echo "model groups: $NGROUPS"

for gi in $(seq 0 $((NGROUPS-1))); do
  MODEL=$(jq_py "['model_groups'][$gi]['model']")
  SRVCMD=$(jq_py "['model_groups'][$gi]['server_command']")
  SRVLOG=$OPENPI/logs/hrc/closed_loop/gate3_${MODEL}_server.log
  echo "=== MODEL GROUP $gi/$((NGROUPS-1)): $MODEL ==="
  echo "SERVER: $SRVCMD"

  env XLA_PYTHON_CLIENT_PREALLOCATE=false \
      OPENPI_DATA_HOME=/root/autodl-tmp/.cache/openpi \
      HF_HOME=/root/autodl-tmp/.cache/huggingface \
      JAX_COMPILATION_CACHE_DIR=/root/autodl-tmp/.cache/jax \
      bash -lc "cd $OPENPI && $SRVCMD" > "$SRVLOG" 2>&1 &
  SRVPID=$!
  echo "$MODEL server pid=$SRVPID (log $SRVLOG)"

  up=0
  for i in $(seq 1 240); do
    if grep -q "listening on" "$SRVLOG"; then up=1; echo "$MODEL LISTENING after ~$((i*3))s"; break; fi
    if ! kill -0 "$SRVPID" 2>/dev/null; then echo "$MODEL SERVER DIED; tail:"; tail -n 40 "$SRVLOG"; break; fi
    sleep 3
  done
  if [ "$up" -ne 1 ]; then
    echo "$MODEL server not up -> skip group (runs will show as missing)"
    kill "$SRVPID" 2>/dev/null; pkill -9 -f serve_hrc_pi05_rollout 2>/dev/null; sleep 5; continue
  fi
  grep -m1 "loading .* @" "$SRVLOG" | sed 's/^/  server-ckpt: /'

  NRUNS=$(jq_py "['model_groups'][$gi]['runs'].__len__()")
  for ri in $(seq 0 $((NRUNS-1))); do
    RID=$(jq_py "['model_groups'][$gi]['runs'][$ri]['run_id']")
    CMD=$(jq_py "['model_groups'][$gi]['runs'][$ri]['command']")
    echo "--- [$MODEL] run $((ri+1))/$NRUNS $RID  $(date -u +%H:%M:%SZ) ---"
    bash -lc "cd $PIPE && $CMD" > "$LOGDIR/${RID}.log" 2>&1
    echo "$RID EXIT=$?"
  done

  # tear down server, wait for GPU to actually free before next group
  kill "$SRVPID" 2>/dev/null; sleep 3
  pkill -9 -f serve_hrc_pi05_rollout 2>/dev/null
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$u" -lt 800 ]; then echo "GPU freed (${u}MiB) after group $gi"; break; fi
    sleep 3
  done
done

echo "GATE3_ALL_ROLLOUTS_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
/root/autodl-tmp/behavior-venv/bin/python "$PIPE/scripts/evaluation/summarize_pi05_closed_loop.py" --plan "$PLAN"
echo "SUMMARY_EXIT=$?"
echo "GATE3_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
