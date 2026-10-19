#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

PROJECT=/root/autodl-tmp/hrc_pipeline_traintest
VENV=/root/behavior-venv
LOG_DIR=$PROJECT/logs

source $VENV/bin/activate
mkdir -p $LOG_DIR

export OMP_NUM_THREADS=8

# ============ 1. 快速验证: 人类+机器人截图 (从 scripts/ 运行) ============
# nohup python $PROJECT/scripts/behavior_task_with_human.py --activity assembling_gift_baskets > $LOG_DIR/behavior_task_human.log 2>&1 &
# echo "behavior_task_with_human PID: $!"

# ============ 2. HRC完整视频 (人行走模式, 完整回放~12分钟) ============
# nohup python $PROJECT/scripts/hrc_minimal_demo.py \
#   --config $PROJECT/configs/hrc_v01.yaml \
#   --return-steps 30 \
#   --face-human-steps 10 \
#   --go-to-task-steps 30 \
#   --wait-steps 3 \
#   > $LOG_DIR/hrc_demo_walk.log 2>&1 &
# echo "hrc_minimal_demo (walk) PID: $!"

# ============ 5. 多任务HRC: task-0026场景 + task-0000的radio (--online-nav) ============
# Round 0 = 打开收音机 (回放 task-0000 外部段)
# Round 1 = 把篮子放到桌上 (回放 task-0026 segment 0)
# --online-nav: 导航段=真实底盘物理驱动+记真实action;GoToTask/FaceHuman原地转身=物理;
#   进操作段前手臂+腰渐进"落位"到录制起始构型 -> 与 BEHAVIOR-1K 任务段连贯无瞬变。
# 语法守卫:任一脚本语法错就不启动仿真,避免白等 ~40min。
python -m py_compile $PROJECT/scripts/nav_online_driver.py \
                     $PROJECT/scripts/hrc_utils.py \
                     $PROJECT/scripts/hrc_minimal_demo.py \
                     $PROJECT/scripts/hrc_multitask_demo.py \
  || { echo "[cmd.sh] py_compile FAILED — 已中止,未启动仿真"; exit 1; }
LOG=$LOG_DIR/multitask_online_step11.log
# Truncate the log at startup so each run is self-contained.
: > $LOG
# setsid+nohup: 脱离终端,SSH 断开也不会被 SIGHUP 杀掉(踩过坑)。
setsid nohup python $PROJECT/scripts/hrc_multitask_demo.py \
  --online-nav \
  --config $PROJECT/configs/hrc_multitask_v01.yaml \
  > $LOG 2>&1 < /dev/null &
echo "hrc_multitask_demo (--online-nav) PID: $!  -> $LOG"

# ============ 3. 人类行走测试 (30秒视频验证动画) ============
# nohup python $PROJECT/scripts/test_human_walk.py \
#   --config $PROJECT/configs/hrc_v01.yaml \
#   --walk-steps 900 \
#   > $LOG_DIR/walk_test.log 2>&1 &
# echo "test_human_walk PID: $!"

# ============ 4. 批量采集 (从 scripts/ 运行) ============
# nohup python $PROJECT/scripts/batch_collect_hrc.py --config $PROJECT/configs/hrc_task.yaml > $LOG_DIR/batch_hrc.log 2>&1 &
# echo "batch_collect_hrc PID: $!"
