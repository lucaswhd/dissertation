#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.
#
# 组合 D — 厨房·热狗+培根 (house_single_floor / kitchen)
#   Round 0 = Cook hot dogs (回放 task-0045, external)
#   Round 1 = Cook bacon    (回放 task-0046 segment 0, base)
# 单 episode 验证用; 批量采集用 scripts/run_collection.py。

PROJECT=/root/hrc_pipeline
VENV=/root/behavior-venv
LOG_DIR=$PROJECT/logs

source $VENV/bin/activate
mkdir -p $LOG_DIR

export OMP_NUM_THREADS=8
export OMNIGIBSON_DATA_PATH=/root/autodl-tmp/datasets
export OMNI_KIT_ACCEPT_EULA=YES

# Truncate the log so each run is self-contained.
: > $LOG_DIR/hrc_combo_D.log
nohup python $PROJECT/scripts/hrc_multitask_demo.py \
  --config $PROJECT/configs/hrc_combo_D.yaml \
  --episode-id 0 \
  > $LOG_DIR/hrc_combo_D.log 2>&1 &
echo "hrc_combo_D (hot dogs + bacon) PID: $!"
echo "log: $LOG_DIR/hrc_combo_D.log"
echo "tail -f $LOG_DIR/hrc_combo_D.log   # 跟踪进度"
