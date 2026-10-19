#!/bin/bash
set -u
PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
LOG_DIR=$PROJECT/logs
CKPT=/root/autodl-tmp/act_runs/hrc_v01_headonly_01/checkpoints/020000/pretrained_model
PORT=18766
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export TMPDIR=/root/autodl-tmp/tmp
export OMP_NUM_THREADS=8
cd $PROJECT
echo "[rollout-headonly] $(date '+%F %T') starting act_policy_server (head-only ckpt)"
/root/autodl-tmp/lerobot-venv/bin/python scripts/act_policy_server.py \
  --ckpt $CKPT --port $PORT > $LOG_DIR/act_server_step3.log 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 90); do
  grep -q "listening on" $LOG_DIR/act_server_step3.log 2>/dev/null && { READY=1; break; }
  sleep 2
done
if [ "$READY" = "1" ]; then
  echo "[rollout-headonly] $(date '+%F %T') server ready, rollout (10239 frames)"
  timeout -k 60 14400 /root/behavior-venv/bin/python scripts/act_rollout_demo.py \
    --config $PROJECT/configs/hrc_multitask_v01.yaml \
    --snapshot /root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10 \
    --output-dir /root/autodl-tmp/outputs/hrc_rollout_headonly_v01 \
    --server-port $PORT --ckpt-label act_runs/hrc_v01_headonly_01@020000 \
    > $LOG_DIR/act_rollout_step3.log 2>&1
  echo "[rollout-headonly] $(date '+%F %T') rollout exited rc=$?"
else
  echo "[rollout-headonly] $(date '+%F %T') ERROR: server never ready"
fi
kill $SERVER_PID 2>/dev/null
echo "[rollout-headonly] $(date '+%F %T') done"
