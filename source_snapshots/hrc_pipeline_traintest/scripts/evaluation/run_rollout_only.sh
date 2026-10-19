#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.
#
# 自包含：server(lerobot-venv) -> 完整 ACT 闭环 rollout(behavior-venv) -> 杀 server。
# 不接 step11 采集（step11 已完成）。由 cmd_rollout_only.sh 以 setsid nohup 脱终端调用。

set -u
PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
LOG_DIR=$PROJECT/logs
CKPT=/root/autodl-tmp/act_runs/hrc_v01_overfit_01/checkpoints/020000/pretrained_model
PORT=18765

export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export TMPDIR=/root/autodl-tmp/tmp
export OMP_NUM_THREADS=8
cd $PROJECT

echo "[rollout-only] $(date '+%F %T') starting act_policy_server"
/root/autodl-tmp/lerobot-venv/bin/python scripts/act_policy_server.py \
  --ckpt $CKPT --port $PORT > $LOG_DIR/act_server_step2.log 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 90); do
  if grep -q "listening on" $LOG_DIR/act_server_step2.log 2>/dev/null; then READY=1; break; fi
  sleep 2
done

if [ "$READY" = "1" ]; then
  echo "[rollout-only] $(date '+%F %T') server ready, starting full rollout (10239 frames)"
  timeout -k 60 14400 /root/behavior-venv/bin/python scripts/act_rollout_demo.py \
    --config $PROJECT/configs/hrc_multitask_v01.yaml \
    --snapshot /root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10 \
    --output-dir /root/autodl-tmp/outputs/hrc_rollout_act_v01 \
    --server-port $PORT --ckpt-label act_runs/hrc_v01_overfit_01@020000 \
    > $LOG_DIR/act_rollout_step2.log 2>&1
  echo "[rollout-only] $(date '+%F %T') rollout exited rc=$?"
else
  echo "[rollout-only] $(date '+%F %T') ERROR: server never became ready"
fi

kill $SERVER_PID 2>/dev/null
echo "[rollout-only] $(date '+%F %T') done (server killed)"
