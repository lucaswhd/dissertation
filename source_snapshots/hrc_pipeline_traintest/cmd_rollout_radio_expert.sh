#!/bin/bash
# 对照实验启动器：专家动作开环 rollout（验证 radio success 指标在 rollout 物理环境里
# 是否可达）。behavior-venv 语法守卫 + setsid nohup 脱终端。无需 policy server。
# 输出: /root/autodl-tmp/outputs/hrc_rollout_radio_expert_control_v01/
# 日志: logs/rollout_radio_expert_step1.log（链）/ act_rollout_radio_expert_<RUNID>.log
# 验收: grep '[radio-dbg] SUMMARY' 看 ever_toggled；[rollout-success] 看 round 0 判定。

PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
LOG_DIR=$PROJECT/logs
mkdir -p $LOG_DIR

/root/behavior-venv/bin/python -m py_compile \
    $PROJECT/scripts/act_rollout_demo.py $PROJECT/scripts/hrc_multitask_demo.py \
    $PROJECT/scripts/hrc_minimal_demo.py $PROJECT/scripts/hrc_utils.py \
    $PROJECT/scripts/nav_online_driver.py \
  || { echo "[cmd_rollout_radio_expert] behavior-venv py_compile FAILED — 已中止"; exit 1; }

CHAIN_LOG=$LOG_DIR/rollout_radio_expert_step1.log
: > $CHAIN_LOG
setsid nohup bash $PROJECT/scripts/run_rollout_radio_expert_control.sh > $CHAIN_LOG 2>&1 < /dev/null &
echo "rollout-radio-expert PID: $!  -> $CHAIN_LOG"
