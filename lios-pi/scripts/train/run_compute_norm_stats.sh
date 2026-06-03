#!/bin/bash

# Compute normalization statistics for robotwin dataset using distributed training
# Usage: bash scripts/train/run_compute_norm_stats.sh

torchrun --standalone --nproc_per_node=8 scripts/train/compute_norm_stats.py \
    pi05_robotwin \
    --data.repo_id /path/to/lerobot_dataset \
    --data.apply_delta_transform \
    --model.action_horizon 10 \
    --batch_size 64