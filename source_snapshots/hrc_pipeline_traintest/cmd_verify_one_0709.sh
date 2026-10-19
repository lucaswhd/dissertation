#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.
#
# 单条 episode 验证启动 (0709 实验分支).
# 只跑 scripts/collection/hrc_multitask_demo_0709.py + configs/hrc_multitask_0001_0709.yaml，
# 不经过默认 wrapper / 默认 yaml，避免污染主线。

set -u
PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
VENV=/root/behavior-venv
source "$VENV/bin/activate"
cd "$PROJECT"

# ===== 参数(前面 VAR=值 覆盖,例: DEST="7,3" TAG=mytest ./cmd_verify_one_0709.sh) =====
TRASH=${TRASH:-/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0001/episode_00012610.hdf5}
RADIO=${RADIO:-/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0000/episode_00000620.hdf5}
DEST=${DEST:-}                 # forced relocate 终点 "x,y"; 留空=用 0709 config 的预定终点
TAG=${TAG:-stair_kitchen_0709_sidecar}
OUT=${OUT:-$PROJECT/outputs/hrc_$TAG}
LOG=$PROJECT/logs/$TAG.log

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# 关键:清掉预检变量,否则脚本会触发 boot 预检然后直接退出、根本不跑 episode
unset HRC_ENDPOINT_PRECHECK
unset HRC_PRETASK_PRECHECK
# DEST 设了就强制走该终点; 否则用 config 的预定终点
if [ -n "$DEST" ]; then export HRC_FORCE_RELOC_DEST="$DEST"; else unset HRC_FORCE_RELOC_DEST; fi
# 第三人称视频默认关(只出 head_rgb); 要同时出第三人称: export HRC_RECORD_THIRD=1

# ===== 语法守卫:任一脚本语法错就不启动 =====
python -m py_compile \
  "$PROJECT/scripts/collection/hrc_multitask_demo_0709.py" \
  "$PROJECT/scripts/collection/hrc_humanoid.py" \
  "$PROJECT/scripts/collection/hrc_minimal_demo.py" \
  "$PROJECT/scripts/collection/nav_online_driver.py" \
  || { echo "[cmd0709] py_compile FAILED — 已中止,未启动"; exit 1; }

mkdir -p "$PROJECT/logs"
: > "$LOG"
{
  echo "[launcher0709] date=$(date -Iseconds)"
  "$VENV/bin/python" -c "import sys,omnigibson; print('[launcher0709] py='+sys.executable); print('[launcher0709] omnigibson='+omnigibson.__version__)"
  echo "[launcher0709] script=$PROJECT/scripts/collection/hrc_multitask_demo_0709.py"
  echo "[launcher0709] config=$PROJECT/configs/hrc_multitask_0001_0709.yaml"
  echo "[launcher0709] trash=$TRASH"
  echo "[launcher0709] radio=$RADIO"
  echo "[launcher0709] out=$OUT  dest=${DEST:-<config predefined>}"
} >> "$LOG" 2>&1

# setsid+nohup:脱离终端,SSH 断开也不会被 SIGHUP 杀掉
setsid nohup "$VENV/bin/python" "$PROJECT/scripts/collection/hrc_multitask_demo_0709.py" \
  --online-nav --config "$PROJECT/configs/hrc_multitask_0001_0709.yaml" \
  --raw-hdf5 "$TRASH" --external-hdf5 "1:$RADIO" \
  --output-dir "$OUT" \
  >> "$LOG" 2>&1 < /dev/null &

PID=$!
echo "[cmd0709] 单条验证已启动: PID=$PID"
echo "  终点 : ${DEST:+forced $DEST}${DEST:-config 预定终点}"
echo "  输出 : $OUT   视频: $OUT/videos/head_rgb.mp4"
echo "  日志 : $LOG"
echo ""
echo "  看关键行 : grep -E 'pre-follow sidecar|PREDEFINED|FORCED|human walking|DEADLOCK|relocation\] done|SUCCESS|ERROR' $LOG"
echo "  跟日志   : tail -f $LOG | grep -vE '\[diag\]|Warning'"
echo "  停止     : kill $PID   (或 pkill -f hrc_$TAG)"
