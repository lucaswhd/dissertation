#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.
#
# §12.6 收尾启动入口：ACT rollout 目检 + step11 采集（success checker 实测）。
# 语法守卫后用 setsid nohup 把整条链（scripts/run_rollout_chain.sh）脱终端：
#   1. act_policy_server (lerobot-venv, 020000)        -> logs/act_server_step1.log
#   2. ACT 闭环 rollout (behavior-venv, ~1-2h)          -> logs/act_rollout_step1.log
#      视频: /root/autodl-tmp/outputs/hrc_rollout_act_v01/videos/
#   3. step11 采集 (bash cmd.sh, ~55min)                -> logs/multitask_online_step11.log

PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
LOG_DIR=$PROJECT/logs
mkdir -p $LOG_DIR

# 语法守卫：任一脚本编译失败就不启动（避免白等数小时）。
/root/behavior-venv/bin/python -m py_compile \
    $PROJECT/scripts/act_rollout_demo.py \
    $PROJECT/scripts/hrc_multitask_demo.py \
    $PROJECT/scripts/hrc_minimal_demo.py \
    $PROJECT/scripts/hrc_utils.py \
    $PROJECT/scripts/nav_online_driver.py \
  || { echo "[cmd_rollout.sh] behavior-venv py_compile FAILED — 已中止"; exit 1; }
/root/autodl-tmp/lerobot-venv/bin/python -m py_compile \
    $PROJECT/scripts/act_policy_server.py \
  || { echo "[cmd_rollout.sh] lerobot-venv py_compile FAILED — 已中止"; exit 1; }

CHAIN_LOG=$LOG_DIR/rollout_chain_step1.log
: > $CHAIN_LOG
setsid nohup bash $PROJECT/scripts/run_rollout_chain.sh > $CHAIN_LOG 2>&1 < /dev/null &
echo "rollout chain PID: $!  -> $CHAIN_LOG"
