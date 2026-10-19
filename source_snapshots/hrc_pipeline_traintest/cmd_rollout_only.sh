#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.
#
# 只跑 ACT 闭环 rollout（不接 step11 采集——step11 已完成）。
# 双 venv 语法守卫 + setsid nohup 脱终端跑 scripts/run_rollout_only.sh。
# 输出: /root/autodl-tmp/outputs/hrc_rollout_act_v01/videos/  供用户目检。
# 日志: logs/rollout_only_step2.log（链）/ act_server_step2.log / act_rollout_step2.log

PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
LOG_DIR=$PROJECT/logs
mkdir -p $LOG_DIR

/root/behavior-venv/bin/python -m py_compile \
    $PROJECT/scripts/act_rollout_demo.py $PROJECT/scripts/hrc_multitask_demo.py \
    $PROJECT/scripts/hrc_minimal_demo.py $PROJECT/scripts/hrc_utils.py \
    $PROJECT/scripts/nav_online_driver.py \
  || { echo "[cmd_rollout_only] behavior-venv py_compile FAILED — 已中止"; exit 1; }
/root/autodl-tmp/lerobot-venv/bin/python -m py_compile $PROJECT/scripts/act_policy_server.py \
  || { echo "[cmd_rollout_only] lerobot-venv py_compile FAILED — 已中止"; exit 1; }

CHAIN_LOG=$LOG_DIR/rollout_only_step2.log
: > $CHAIN_LOG
setsid nohup bash $PROJECT/scripts/run_rollout_only.sh > $CHAIN_LOG 2>&1 < /dev/null &
echo "rollout-only PID: $!  -> $CHAIN_LOG"
