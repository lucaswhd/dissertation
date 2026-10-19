#!/bin/bash
set -u
PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
LOG_DIR=$PROJECT/logs
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
CKPT=/root/autodl-tmp/act_runs/hrc_v01_radio_exec_headonly_shift1_01/checkpoints/020000/pretrained_model
PORT=${PORT:-18768}
OUT_DIR=/root/autodl-tmp/outputs/hrc_rollout_radio_exec_shift1_v01
SERVER_LOG=$LOG_DIR/act_server_radio_shift1_${RUN_ID}.log
ROLLOUT_LOG=$LOG_DIR/act_rollout_radio_shift1_${RUN_ID}.log
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export TMPDIR=/root/autodl-tmp/tmp
export OMP_NUM_THREADS=8
cd $PROJECT
mkdir -p $LOG_DIR

echo "[rollout-radio-shift1] $(date '+%F %T') starting act_policy_server"
echo "[rollout-radio-shift1] ckpt=$CKPT"
echo "[rollout-radio-shift1] out_dir=$OUT_DIR"
echo "[rollout-radio-shift1] server_log=$SERVER_LOG"
echo "[rollout-radio-shift1] rollout_log=$ROLLOUT_LOG"
/root/autodl-tmp/lerobot-venv/bin/python scripts/act_policy_server.py   --ckpt $CKPT --port $PORT > $SERVER_LOG 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 90); do
  grep -q "listening on" $SERVER_LOG 2>/dev/null && { READY=1; break; }
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[rollout-radio-shift1] $(date '+%F %T') ERROR: server process exited early"
    tail -n 80 $SERVER_LOG 2>/dev/null || true
    exit 1
  fi
  sleep 2
done
if [ "$READY" = "1" ]; then
  echo "[rollout-radio-shift1] $(date '+%F %T') server ready, rollout shift1 checkpoint"
  timeout -k 60 7200 /root/behavior-venv/bin/python scripts/act_rollout_demo.py     --config $PROJECT/configs/hrc_multitask_v01.yaml     --snapshot /root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10     --output-dir $OUT_DIR     --server-port $PORT --ckpt-label act_runs/hrc_v01_radio_exec_headonly_shift1_01@020000     --start-phase 3 --max-frames 1956     > $ROLLOUT_LOG 2>&1
  RC=$?
  echo "[rollout-radio-shift1] $(date '+%F %T') rollout exited rc=$RC"
  echo "[rollout-radio-shift1] videos: $OUT_DIR/videos/head_rgb.mp4 $OUT_DIR/videos/third_person_rgb.mp4"
else
  echo "[rollout-radio-shift1] $(date '+%F %T') ERROR: server never ready"
  tail -n 120 $SERVER_LOG 2>/dev/null || true
  RC=1
fi
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "[rollout-radio-shift1] $(date '+%F %T') done"
exit ${RC:-0}
