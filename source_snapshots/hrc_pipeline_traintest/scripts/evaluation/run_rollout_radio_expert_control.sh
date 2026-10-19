#!/bin/bash
set -u
# 对照实验：把 step10 快照里录下的 23 维「专家动作」开环喂进 rollout 物理环境
# （无 load_state、无 policy server），看 ToggledOn(radio_89) 能否被真实物理触发。
# 这是「radio 失败到底是策略问题还是 success 指标本身物理不可达」的判别实验。
# 输出独立目录，不覆盖 policy rollout 产物。
PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
LOG_DIR=$PROJECT/logs
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=/root/autodl-tmp/outputs/hrc_rollout_radio_expert_control_v01
ROLLOUT_LOG=$LOG_DIR/act_rollout_radio_expert_${RUN_ID}.log
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export TMPDIR=/root/autodl-tmp/tmp
export OMP_NUM_THREADS=8
cd $PROJECT
mkdir -p $LOG_DIR

echo "[rollout-radio-expert] $(date '+%F %T') start (expert open-loop control, no server)"
echo "[rollout-radio-expert] out_dir=$OUT_DIR"
echo "[rollout-radio-expert] rollout_log=$ROLLOUT_LOG"
timeout -k 60 7200 /root/behavior-venv/bin/python scripts/act_rollout_demo.py \
    --config $PROJECT/configs/hrc_multitask_v01.yaml \
    --snapshot /root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10 \
    --output-dir $OUT_DIR \
    --action-source expert \
    --ckpt-label EXPERT_OPENLOOP_CONTROL \
    --start-phase 3 --max-frames 1956 \
    --debug-radio \
    > $ROLLOUT_LOG 2>&1
RC=$?
echo "[rollout-radio-expert] $(date '+%F %T') rollout exited rc=$RC"
echo "[rollout-radio-expert] videos: $OUT_DIR/videos/head_rgb.mp4 $OUT_DIR/videos/third_person_rgb.mp4"
echo "[rollout-radio-expert] grep '[radio-dbg] SUMMARY' $ROLLOUT_LOG  # 看是否 ever_toggled"
echo "[rollout-radio-expert] $(date '+%F %T') done"
exit ${RC:-0}
