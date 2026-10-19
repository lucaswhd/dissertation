#!/bin/bash
# Gate 2 orchestrator: bring up the E4@5k pi0.5 server, run ONE closed-loop
# rollout (val source 0) against it, tear the server down, then summarize.
#
# Two GPU processes share the one 4090: the JAX policy server and OmniGibson.
# The server MUST run with XLA_PYTHON_CLIENT_PREALLOCATE=false or JAX grabs ~90%
# of VRAM up front and the simulator OOMs (runbook §5 omits this).
set -u

OPENPI=/root/autodl-tmp/openpi_hrc
PIPE=/root/autodl-tmp/hrc_pipeline_traintest
PLAN=$PIPE/outputs/pi05_closed_loop/gate2/run_plan.json
CKPT=$OPENPI/checkpoints/pi05_hrc_e4_main/hrc_e4_seed42/5000
SRVLOG=$OPENPI/logs/hrc/closed_loop/gate2_e4_server.log
RUNLOG=$PIPE/logs/pi05_closed_loop/gate2_rollout.log

mkdir -p "$OPENPI/logs/hrc/closed_loop" "$PIPE/logs/pi05_closed_loop"
echo "GATE2_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. start the policy server (preallocate=false leaves VRAM for OmniGibson)
cd "$OPENPI"
env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OPENPI_DATA_HOME=/root/autodl-tmp/.cache/openpi \
    HF_HOME=/root/autodl-tmp/.cache/huggingface \
    JAX_COMPILATION_CACHE_DIR=/root/autodl-tmp/.cache/jax \
    .venv/bin/python scripts/hrc/serve_hrc_pi05_rollout.py \
    --config pi05_hrc_e4_main \
    --checkpoint "$CKPT" \
    --host 127.0.0.1 --port 18765 --execute-horizon 8 \
    > "$SRVLOG" 2>&1 &
SRVPID=$!
echo "server pid=$SRVPID  (log: $SRVLOG)"

# 2. wait up to ~10 min for the server to finish loading the checkpoint
up=0
for i in $(seq 1 200); do
  if grep -q "listening on 127.0.0.1:18765" "$SRVLOG"; then up=1; echo "server LISTENING after ~$((i*3))s"; break; fi
  if ! kill -0 "$SRVPID" 2>/dev/null; then echo "SERVER DIED before listening; tail:"; tail -n 40 "$SRVLOG"; exit 3; fi
  sleep 3
done
if [ "$up" -ne 1 ]; then echo "SERVER TIMEOUT waiting for listen"; kill "$SRVPID" 2>/dev/null; exit 4; fi

# 3. run the single rollout command straight from the plan (verbatim)
CMD=$(python3 -c "import json;print(json.load(open('$PLAN'))['model_groups'][0]['runs'][0]['command'])")
echo "ROLLOUT CMD: $CMD"
cd "$PIPE"
bash -lc "$CMD" > "$RUNLOG" 2>&1
rc=$?
echo "ROLLOUT_EXIT=$rc  (log: $RUNLOG)"

# 4. tear the server down cleanly
kill "$SRVPID" 2>/dev/null
sleep 3
pkill -9 -f serve_hrc_pi05_rollout 2>/dev/null

# 5. summarize (behavior-venv owns the summarizer)
cd "$PIPE"
/root/autodl-tmp/behavior-venv/bin/python scripts/evaluation/summarize_pi05_closed_loop.py --plan "$PLAN"
echo "SUMMARY_EXIT=$?"
echo "GATE2_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
