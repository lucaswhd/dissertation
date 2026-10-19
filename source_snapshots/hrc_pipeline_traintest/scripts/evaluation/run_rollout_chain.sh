#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.
#
# §12.6 收尾链（由 cmd_rollout.sh 以 setsid nohup 整体脱终端调用）：
#   1. 起 act_policy_server（lerobot-venv，020000 checkpoint）
#   2. ACT 闭环 rollout（behavior-venv，整条 step10 时间轴）→ 视频供目检
#   3. 杀 server，跑一轮 step11 采集（bash cmd.sh）→ 实测 per-round success checker
# 任一环节失败不阻断后续环节（rollout 失败也照样验 success checker）。

set -u
PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
LOG_DIR=$PROJECT/logs
CKPT=/root/autodl-tmp/act_runs/hrc_v01_overfit_01/checkpoints/020000/pretrained_model
SERVER_LOG=$LOG_DIR/act_server_step1.log
ROLLOUT_LOG=$LOG_DIR/act_rollout_step1.log
PORT=18765

# lerobot-venv 缓存全部指向 autodl-tmp（系统盘只剩 ~2G，§12.3 踩过的坑）
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export TMPDIR=/root/autodl-tmp/tmp
export OMP_NUM_THREADS=8

cd $PROJECT

echo "[chain] $(date '+%F %T') starting act_policy_server"
/root/autodl-tmp/lerobot-venv/bin/python scripts/act_policy_server.py \
  --ckpt $CKPT --port $PORT > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# 等 server 就绪（加载 checkpoint ~30s）
READY=0
for i in $(seq 1 90); do
  if /root/behavior-venv/bin/python -c "
import socket;
s=socket.create_connection(('127.0.0.1', $PORT), timeout=2); s.close()" 2>/dev/null; then
    READY=1; break
  fi
  sleep 2
done

if [ "$READY" = "1" ]; then
  echo "[chain] $(date '+%F %T') server ready, starting rollout"
  # -k 60: timeout 后再宽限 60s 强杀（Isaac 收尾慢）
  timeout -k 60 14400 /root/behavior-venv/bin/python scripts/act_rollout_demo.py \
    --config $PROJECT/configs/hrc_multitask_v01.yaml \
    --snapshot /root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10 \
    --output-dir /root/autodl-tmp/outputs/hrc_rollout_act_v01 \
    --server-port $PORT \
    --ckpt-label act_runs/hrc_v01_overfit_01@020000 \
    > $ROLLOUT_LOG 2>&1
  echo "[chain] $(date '+%F %T') rollout exited rc=$?"
else
  echo "[chain] $(date '+%F %T') ERROR: server never became ready, skipping rollout"
fi

kill $SERVER_PID 2>/dev/null
sleep 5

echo "[chain] $(date '+%F %T') starting step11 collection (success-checker verification)"
bash $PROJECT/cmd.sh
echo "[chain] $(date '+%F %T') chain script done (collection detached, see cmd.sh log)"
