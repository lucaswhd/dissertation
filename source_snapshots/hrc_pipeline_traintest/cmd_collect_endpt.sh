#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.
#
# ============================================================================
# HRC 预定路径 Relocate 批量采集 (2026-07-07 预定终点方案)
# ----------------------------------------------------------------------------
# Relocate 阶段不再随机采样目的地,而是走 config 里指定的开阔终点:
#   configs/hrc_multitask_0001.yaml  ->  relocation_dest: auto
# auto = 每对 pair 用 endpoint_search.py 在机器人体宽腐蚀图上现搜最直的终点
# (3-6m 带内, 机器人当前位姿被 mask 掉 -> 人的路径不可能穿过机器人).
# 人去 taper 匀速走完直接 A* 路径,机器人 leash 跟随;只渲染第一人称视频.
# 每个 episode 一个子进程(Isaac Sim 是进程级单例),--online-nav 默认开.
# ============================================================================

set -u
PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
VENV=/root/behavior-venv
source "$VENV/bin/activate"
cd "$PROJECT"

export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=8
# 清掉调试用环境变量,否则会覆盖 config 预定终点 / 触发预检 hook:
unset HRC_FORCE_RELOC_DEST
unset HRC_ENDPOINT_PRECHECK
# 第三人称视频默认关(只出 head_rgb);要同时出第三人称就: export HRC_RECORD_THIRD=1

# ===== 可调参数(命令行前面 VAR=... 覆盖,例: COUNT=10 PARALLEL=2 ./cmd_collect_endpt.sh) =====
MANIFEST=${MANIFEST:-$PROJECT/outputs/hrc_batch_0001/pairing_manifest.json}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/hrc_endpt_v2}   # 输出目录(每条一个 ep_XXXX/)
COUNT=${COUNT:-65}                                    # 采集条数(manifest 共 65 对)
START=${START:-0}                                     # 从 manifest 第几个开始
PARALLEL=${PARALLEL:-2}                               # 并发子进程数(24G VRAM 约 2)
TIMEOUT_MIN=${TIMEOUT_MIN:-90}                        # 单条硬超时

# ===== 语法守卫:任一脚本语法错就不启动,避免白等 40min =====
python -m py_compile \
  "$PROJECT/scripts/collection/hrc_multitask_demo.py" \
  "$PROJECT/scripts/collection/hrc_humanoid.py" \
  "$PROJECT/scripts/collection/hrc_minimal_demo.py" \
  "$PROJECT/scripts/collection/nav_online_driver.py" \
  "$PROJECT/scripts/collection/batch_collect_multitask.py" \
  || { echo "[cmd] py_compile FAILED — 已中止,未启动"; exit 1; }

mkdir -p "$PROJECT/logs"
LOG="$PROJECT/logs/collect_$(basename "$OUT_ROOT").log"
: > "$LOG"

# setsid+nohup:脱离终端,SSH 断开也不会被 SIGHUP 杀掉(--resume 断点续跑)
setsid nohup python "$PROJECT/scripts/collection/batch_collect_multitask.py" \
  --manifest "$MANIFEST" \
  --out-root "$OUT_ROOT" \
  --start "$START" --count "$COUNT" \
  --parallel "$PARALLEL" --timeout-min "$TIMEOUT_MIN" \
  --resume \
  > "$LOG" 2>&1 < /dev/null &

echo "[cmd] 批量采集(预定终点)已启动: PID=$!"
echo "  终点 : $(grep 'relocation_dest:' "$PROJECT/configs/hrc_multitask_0001.yaml" | tail -1 | sed 's/^ *//')"
echo "  输出 : $OUT_ROOT"
echo "  日志 : $LOG"
echo "  规模 : count=$COUNT start=$START parallel=$PARALLEL timeout=${TIMEOUT_MIN}min"
echo ""
echo "  看进度   : grep -E '\[batch\] ep' $LOG"
echo "  看通过率 : grep 'cumulative yield' $LOG | tail -1"
echo "  停止     : pkill -f batch_collect_multitask   (再 pkill -f $(basename "$OUT_ROOT") 停子进程)"
