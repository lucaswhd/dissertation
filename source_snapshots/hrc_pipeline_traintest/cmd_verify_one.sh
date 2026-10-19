#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
#
# ============================================================================
# 单条 episode 验证启动 (预定路径方案) — 改完代码想跑一条看效果时用这个.
# 与批量 cmd_collect_endpt.sh 的区别:只跑一条 raw pair,方便盯日志/看视频.
# ============================================================================

set -u
PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
VENV=/root/behavior-venv
source "$VENV/bin/activate"
cd "$PROJECT"

# ===== 参数(前面 VAR=值 覆盖,例: DEST="7,3" TAG=mytest ./cmd_verify_one.sh) =====
TRASH=${TRASH:-/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0001/episode_00012610.hdf5}
RADIO=${RADIO:-/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0000/episode_00000620.hdf5}
DEST=${DEST:-}                 # forced relocate 终点 "x,y" (调试指定路径); 留空=用 config 的预定终点
TAG=${TAG:-verify_one}
OUT=${OUT:-$PROJECT/outputs/hrc_$TAG}
LOG=$PROJECT/logs/$TAG.log

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# 关键:清掉预检变量,否则脚本会触发 boot 预检然后直接退出、根本不跑 episode
unset HRC_ENDPOINT_PRECHECK
# DEST 设了就强制走该终点(关 taper+wander 的确定直线); 否则用 config 的预定终点
if [ -n "$DEST" ]; then export HRC_FORCE_RELOC_DEST="$DEST"; else unset HRC_FORCE_RELOC_DEST; fi
# 第三人称视频默认关(只出 head_rgb); 要同时出第三人称: export HRC_RECORD_THIRD=1

# ===== 语法守卫:任一脚本语法错就不启动 =====
python -m py_compile \
  "$PROJECT/scripts/collection/hrc_multitask_demo.py" \
  "$PROJECT/scripts/collection/hrc_humanoid.py" \
  "$PROJECT/scripts/collection/hrc_minimal_demo.py" \
  "$PROJECT/scripts/collection/nav_online_driver.py" \
  || { echo "[cmd] py_compile FAILED — 已中止,未启动"; exit 1; }

mkdir -p "$PROJECT/logs"
: > "$LOG"
{
  echo "[launcher] date=$(date -Iseconds)"
  "$VENV/bin/python" -c "import sys,omnigibson; print('[launcher] py='+sys.executable); print('[launcher] omnigibson='+omnigibson.__version__)"
  echo "[launcher] trash=$TRASH"
  echo "[launcher] radio=$RADIO"
  echo "[launcher] out=$OUT  dest=${DEST:-<config predefined>}"
} >> "$LOG" 2>&1

# setsid+nohup:脱离终端,SSH 断开也不会被 SIGHUP 杀掉
setsid nohup "$VENV/bin/python" "$PROJECT/scripts/hrc_multitask_demo.py" \
  --online-nav --config "$PROJECT/configs/hrc_multitask_0001.yaml" \
  --raw-hdf5 "$TRASH" --external-hdf5 "1:$RADIO" \
  --output-dir "$OUT" \
  >> "$LOG" 2>&1 < /dev/null &

PID=$!
echo "[cmd] 单条验证已启动: PID=$PID"
echo "  终点 : ${DEST:+forced $DEST}${DEST:-config 预定终点}"
echo "  输出 : $OUT   视频: $OUT/videos/head_rgb.mp4"
echo "  日志 : $LOG"
echo ""
echo "  看关键行 : grep -E 'PREDEFINED|FORCED|human walking|DEADLOCK|relocation\] done|SUCCESS' $LOG"
echo "  跟日志   : tail -f $LOG | grep -vE '\[diag\]|Warning'"
echo "  停止     : kill $PID   (或 pkill -f hrc_$TAG)"
