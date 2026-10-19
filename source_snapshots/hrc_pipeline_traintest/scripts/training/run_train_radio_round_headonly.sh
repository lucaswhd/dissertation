#!/bin/bash
set -u
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export TMPDIR=/root/autodl-tmp/tmp
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/hf_datasets
export WANDB_DIR=/root/autodl-tmp/.cache/wandb
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/hrc_pipeline_traintest
echo "[train-radio-round] $(date '+%F %T') start"
/root/autodl-tmp/lerobot-venv/bin/lerobot-train \
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
  --wandb.enable=true --wandb.project=hrc-vla \
  --dataset.repo_id=hrc/hrc_v01_radio_round_headonly \
  --dataset.root=/root/autodl-tmp/lerobot_datasets/hrc_v01_radio_round_headonly \
  --dataset.video_backend=pyav \
  --batch_size=8 --steps=20000 --seed=1000 --save_freq=5000 --log_freq=200 \
  --output_dir=/root/autodl-tmp/act_runs/hrc_v01_radio_round_headonly_01
echo "[train-radio-round] $(date '+%F %T') exit rc=$?"
