#!/bin/zsh
# Example multi-dataset training launcher (LeRobot backend).
# Setup (run once):
#   pixi install -e dev
#   pixi run -e dev lerobot   # separate task; pixi has no --no-deps (see pixi.toml)
#
# Usage (single node):
#   zsh scripts/train/start_example.sh <exp_name> <num_gpus>
# For multi-node, provide NNODES / NODE_RANK / MASTER_ADDR via your scheduler's env vars.

set -uo pipefail

# ╔════════════════════════════════════════════════════════════════════╗
# ║  >>> MODIFY THIS SECTION FOR EACH EXPERIMENT <<<                   ║
# ╚════════════════════════════════════════════════════════════════════╝

EXP_NAME="${1:-airbot_test}"
NUM_GPUS="${2:-8}"

# Optional multi-node env (falls back to single-node localhost)
NNODES="${SLURM_NNODES:-1}"
NODE_RANK=$(( ${SLURM_PROCID:-${JOB_COMPLETION_INDEX:-0}} ))
MASTER_ADDR="${SLURM_JOB_FIRST_NODE_IP:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# One or more LeRobot dataset directory names under DATA_ROOT.
DATASETS=(
    your-dataset-1
    your-dataset-2
)

DATA_ROOT="/path/to/your/dataset"
EXPERIMENT_DIR="/path/to/experiments"
ASSET_ID="airbot"
PROJECT_NAME="pi05_airbot"
CHECKPOINT_BASE_DIR="/path/to/checkpoints"
POLICY_CONFIG="pi05_airbot"

# ============== Setup Symlink Directory ==============
EXP_DATA_DIR="${EXPERIMENT_DIR}/${EXP_NAME}"
mkdir -p "$EXP_DATA_DIR"
rm -f "$EXP_DATA_DIR"/*

echo "Creating experiment directory: $EXP_DATA_DIR"
for ds in "${DATASETS[@]}"; do
    src="${DATA_ROOT}/${ds}"
    if [[ -d "$src" ]]; then
        ln -sf "$src" "$EXP_DATA_DIR/$ds"
        echo "  Linked: $ds"
    else
        echo "  WARNING: Dataset not found: $src"
    fi
done

echo ""
echo "Datasets for experiment '$EXP_NAME' (${#DATASETS[@]} datasets):"
ls -la "$EXP_DATA_DIR"

# ============== Log Configuration ==============
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/airbot_${EXP_NAME}_node${NODE_RANK}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "Airbot Training (Multi-Node): $EXP_NAME"
echo "Job started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo "Data root: $DATA_ROOT"
echo "Asset ID: $ASSET_ID"
echo "Nodes: $NNODES | GPUs per node: $NUM_GPUS | Total GPUs: $((NNODES * NUM_GPUS))"
echo "Node rank: $NODE_RANK | Master: $MASTER_ADDR:$MASTER_PORT"
echo "=============================================="

# ============== Environment Variables ==============
export CUDA_LAUNCH_BLOCKING=0
export TORCH_MULTIPROCESSING_START_METHOD=spawn
# Cluster-specific NCCL tuning — adjust the interface / InfiniBand settings for
# your network, or remove this block for single-node runs.
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIME_OUT="22"
export CONDA_OVERRIDE_CUDA=12.9
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_HOME="/path/to/hf_cache"
# Avoid HF datasets cache race condition across multi-GPU processes
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache_node${NODE_RANK}"
rm -rf "$HF_DATASETS_CACHE" 2>/dev/null || true
mkdir -p "$HF_DATASETS_CACHE"

# ============== Training ==============
TOTAL_GPUS=$((NNODES * NUM_GPUS))
BATCH_SIZE=$((TOTAL_GPUS * 24))

EXIT_CODE=0
pixi run -e dev torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NUM_GPUS" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/train/train_pytorch_fsdp.py \
    "$POLICY_CONFIG" \
    --shuffle \
    --lr-schedule.peak-lr 5e-5 \
    --lr-schedule.decay-lr 2e-5 \
    --lr-schedule.warmup-steps 5000 \
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
    --project-name "$PROJECT_NAME" \
    --exp-name "airbot_${EXP_NAME}_$(date +%Y%m%d_%H)" \
    --model.action_horizon 30 \
    --model.state_history_frames 1 \
    --model.state_delay_frames 0 \
    --parent-data-dir "$EXP_DATA_DIR" \
    --data.asset_id "$ASSET_ID" \
    --data.test_ep_num 10 \
    --num-epochs 30 \
    --test_step_interval 5000 \
    --save_step_interval 5000 \
    --save_epoch_interval None \
    --num-workers 8 \
    --batch-size "$BATCH_SIZE" \
    --overwrite \
    || EXIT_CODE=$?

echo "=============================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE

# if resume training, add:
# --no-overwrite \
# --resume \