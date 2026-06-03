#!/bin/bash

# Batch compute normalization statistics for all datasets in robotwin pi_data
# Usage: bash scripts/train/batch_compute_norm_stats.sh

set -e  # Exit on error

# Base directory containing all datasets
BASE_DIR="/path/to/datasets"
SUBDATASET="aloha-agilex_demo-clean"

# Log file
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/batch_norm_stats_$(date +%Y%m%d_%H%M%S).log"

echo "Starting batch normalization stats computation at $(date)" | tee -a "$LOG_FILE"
echo "Base directory: $BASE_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Count total datasets
total_datasets=$(ls -d "$BASE_DIR"/*/ | wc -l)
current=0
success=0
failed=0

# Iterate through each dataset directory
for dataset_dir in "$BASE_DIR"/*/; do
    current=$((current + 1))
    dataset_name=$(basename "$dataset_dir")
    full_path="${dataset_dir}${SUBDATASET}"

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$current/$total_datasets] Processing: $dataset_name" | tee -a "$LOG_FILE"
    echo "Path: $full_path" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # Check if the subdataset directory exists
    if [ ! -d "$full_path" ]; then
        echo "WARNING: Directory not found, skipping: $full_path" | tee -a "$LOG_FILE"
        failed=$((failed + 1))
        continue
    fi

    # Run the computation
    echo "Running computation..." | tee -a "$LOG_FILE"
    start_time=$(date +%s)

    if torchrun --standalone --nproc_per_node=8 scripts/train/compute_norm_stats.py \
        pi05_robotwin \
        --data.repo_id "$full_path" \
        --data.apply_delta_transform \
        --model.action_horizon 10 \
        --batch_size 64 2>&1 | tee -a "$LOG_FILE"; then

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "SUCCESS: Completed in ${duration}s" | tee -a "$LOG_FILE"
        success=$((success + 1))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "ERROR: Failed after ${duration}s" | tee -a "$LOG_FILE"
        failed=$((failed + 1))
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Batch processing completed at $(date)" | tee -a "$LOG_FILE"
echo "Total datasets: $total_datasets" | tee -a "$LOG_FILE"
echo "Successful: $success" | tee -a "$LOG_FILE"
echo "Failed: $failed" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

if [ $failed -gt 0 ]; then
    echo "WARNING: Some datasets failed. Check log file: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
else
    echo "All datasets processed successfully!" | tee -a "$LOG_FILE"
    exit 0
fi
