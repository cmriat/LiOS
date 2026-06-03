# Training

[English](./training.md) | [中文](./training.zh-CN.md)

Single- and multi-node FSDP training over LeRobot datasets, with optional LMDB-queue
preprocessing pipeline, EMA, gradient checkpointing, and Torch profiler hooks.

> **No-GPU note.** Both the FSDP trainer and `compute_norm_stats.py` require CUDA. There
> is no CPU fallback. Use a workstation with at least one CUDA 12.x GPU before continuing.

## 1. Prerequisites

```bash
# One-time setup
pixi install -e dev
pixi run -e dev lerobot   # installs LeRobot from a pinned git ref via a separate
                          # task because pixi has no equivalent of pip --no-deps
                          # (see the [tasks.lerobot] block in pixi.toml)
```

Notes on dependencies in `pixi.toml`:

- One of the conda channels (`https://srgconda.bj.bcebos.com/`) is maintained by
  cmriat on Baidu BOS and serves CUDA builds (pytorch, flash-attn, transformer-engine,
  deepep/deepgemm/grouped-gemm/liger-kernel) that aren't on conda-forge. It is
  publicly readable — no credentials required.
- `transformers==4.55.0` is **pinned**. The Gemma modeling code under `src/pi/models/`
  is forked from this exact version; upgrading transformers without re-forking will
  break AdaRMS.
- `lerobot` is installed via pip from a pinned commit
  (`huggingface/lerobot@0cf86487`).

## 2. Resource requirements

Estimates from `scripts/train/start_example.sh` defaults
(`gemma_2b + gemma_300m`, bfloat16, FSDP):

| Setup            | GPUs   | per-GPU batch | Global batch | VRAM/GPU | Throughput |
|------------------|--------|---------------|--------------|----------|-----------|
| Single node A100 | 8×80GB | 24            | 192          | ~35 GB   | ~1.8 step/s |
| Single node H100 | 8×80GB | 24            | 192          | ~30 GB   | ~3.0 step/s |
| 2-node A100      | 16×80GB| 24            | 384          | ~35 GB   | ~3.5 step/s |

Time-to-train rough estimates (`num_epochs=50`, 200k-step regime):

- LIBERO scale (~50K frames, 250 samples): **6–10 h** on 8×A100
- AirBot small (~250 episodes): **12–24 h** on 8×A100
- RoboTwin (full): **3–7 days** on 16×A100

Tune `batch_size`, `num_workers`, and `action_horizon` to fit your VRAM budget. If OOM,
in order of preference: (1) reduce `batch_size`, (2) reduce `action_horizon`,
(3) enable gradient checkpointing, (4) reduce `state_history_frames`.

## 3. Quickstart: single-node 8-GPU training

```bash
# 1. Compute normalization statistics for the dataset (one-time per dataset)
pixi run -e dev torchrun --standalone --nproc_per_node=8 \
    scripts/train/compute_norm_stats.py \
    pi05_airbot \
    --data.repo_id /abs/path/to/lerobot_dataset

# 2. Edit the `>>> MODIFY THIS SECTION <<<` block at the top of
#    scripts/train/start_example.sh. See the placeholder table below.

# 3. Launch
zsh scripts/train/start_example.sh my_experiment 8
```

### 3.1 Placeholders in `start_example.sh`

The script ships with placeholder values that **must** be replaced before the first
run. Five are hard placeholders (the script will fail at runtime if left as-is); the
remaining three are tied to the bundled `pi05_airbot` config and only need to change
when targeting a different dataset / policy config.

| Variable              | Default in script             | What to set                                                  | Required?              |
|-----------------------|-------------------------------|--------------------------------------------------------------|------------------------|
| `DATA_ROOT`           | `/path/to/your/dataset`       | Parent directory under which `DATASETS=()` are looked up      | Yes                    |
| `DATASETS=( ... )`    | `your-dataset-1`, `your-dataset-2` | LeRobot dataset directory names under `DATA_ROOT`             | Yes                    |
| `EXPERIMENT_DIR`      | `/path/to/experiments`        | Where the per-experiment symlink directory will be created   | Yes                    |
| `CHECKPOINT_BASE_DIR` | `/path/to/checkpoints`        | Root for `torch.distributed.checkpoint` shards                | Yes                    |
| `HF_HOME`             | `/path/to/hf_cache`           | HuggingFace cache; the script sets `HF_HUB_OFFLINE=1` so the directory must already contain the required cached artifacts | Yes |
| `ASSET_ID`            | `airbot`                      | `assets/<asset_id>/norm_stats.json` lookup key                | Only if not airbot     |
| `PROJECT_NAME`        | `pi05_airbot`                 | wandb project name                                            | Optional               |
| `POLICY_CONFIG`       | `pi05_airbot`                 | Config name from `src/pi/training/instance_config.py`         | Only if using a different config |

The launcher creates a per-experiment symlink directory under `EXPERIMENT_DIR`, then
calls `torchrun … scripts/train/train_pytorch_fsdp.py <POLICY_CONFIG>` with the
config name (e.g. `pi05_airbot`) plus CLI overrides for learning rate, batch size,
checkpoint dir, etc.

Configs are defined in `src/pi/training/instance_config.py`. The CLI uses
[tyro](https://github.com/brentyi/tyro), so any field on `TrainConfig` / `DatasetConfig`
/ `PiConfig` can be overridden:

```bash
torchrun ... scripts/train/train_pytorch_fsdp.py pi05_airbot \
    --batch-size 16 \
    --num-epochs 30 \
    --model.action_horizon 30 \
    --data.repo_id /abs/path/to/dataset \
    --data.test_ep_num 10
```

## 4. Multi-node training

`start_example.sh` already supports SLURM and Kubernetes-style multi-node setups:

```bash
# SLURM
sbatch --nodes=2 --gres=gpu:8 scripts/train/start_example.sh my_exp 8

# Generic launcher reads:
#   NNODES        (from SLURM_NNODES)
#   NODE_RANK     (from SLURM_PROCID or JOB_COMPLETION_INDEX)
#   MASTER_ADDR   (from SLURM_JOB_FIRST_NODE_IP)
#   MASTER_PORT   (default 29500)
```

The cluster-specific NCCL block in `start_example.sh` is **tuned for one specific
cluster** and should be adapted or removed:

```bash
export NCCL_SOCKET_IFNAME="eth0"     # ← change to your interface
export NCCL_IB_GID_INDEX="3"         # ← cluster-specific
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIME_OUT="22"
```

For single-node runs, remove this block entirely.

## 5. Data loading

Standard PyTorch `DataLoader` with multi-worker preprocessing, distributed sampling,
and a CUDA-stream prefetcher. Tune via `TrainConfig`:

```python
# pi05_airbot config snippet
num_workers=8,
batch_size=24,
shuffle=True,
```

The `CUDAPrefetcher` in `scripts/train/data_loader.py` overlaps host→device transfers
with compute; pinned memory + persistent workers are on by default. Image
augmentations (random crop, rotation, brightness/contrast) run on the GPU inside the
collate function.

## 6. Distributed training stack

| Layer                | Implementation                                                  |
|----------------------|-----------------------------------------------------------------|
| Sharding             | FSDP (`fsdp_wrap` in `scripts/train/utils.py`)                  |
| Compilation          | `torch.compile(raw_model, fullgraph=True)` *before* FSDP wrap   |
| Mixed precision      | bfloat16 forward / fp32 master weights (FSDP `MixedPrecision`)  |
| Gradient clipping    | Cross-mesh gradient clipping (`clip_grad_norm_`)                |
| EMA                  | `torch._foreach_lerp_` on shadow params, default `decay=0.999`  |
| Checkpoint format    | `torch.distributed.checkpoint` (sharded, multi-shard files)     |
| Activation ckpt      | Opt-in via `model.gradient_checkpointing_enable()`              |

### Key params kept in fp32

A subset of PaliGemma params are forced to fp32 even with bfloat16 training:
patch embedding, position embedding, and all `*_layernorm` weights. See
`PaliGemmaWithExpertModel.to_bfloat16_for_selected_params`.

## 7. Resume

```bash
zsh scripts/train/start_example.sh my_exp 8 \
    --no-overwrite --resume
```

Resume is **step-granular**: the trainer resumes from the exact `(epoch, batch_idx)`
within the epoch by computing `start_epoch = global_step // steps_per_epoch` and
skipping the first `global_step % steps_per_epoch` batches of the resumed epoch.
`sampler.set_epoch(epoch)` is called so shuffling stays deterministic.

EMA and optimizer state are also restored. See
`resume_from_fsdp_model_checkpoint` in `scripts/train/utils.py`.

## 8. Profiling

The Torch profiler and CUDA memory snapshotter are wired up but **off by default**.
Toggle in `train_pytorch_fsdp.py`:

```python
profiling_config = ProfilingConfig(
    enable_profiling=True,                                        # ← turn on
    save_traces_folder=f"./traces_{base_config.batch_size}",
    profile_freq=10,
    enable_memory_snapshot=True,                                  # ← turn on
    save_memory_snapshot_folder=f"./memory_snapshot_{base_config.batch_size}",
)
```

Traces dump per-rank Chrome trace JSON files; memory snapshots are pickle dumps that
load into `https://pytorch.org/memory_viz`.

### Per-step timing (always-on for wandb)

`train_pytorch_fsdp.py` records four metrics per step using CUDA events:

| Metric                | Meaning                              |
|-----------------------|--------------------------------------|
| `perf/step_total_s`   | End-to-end step wall time            |
| `perf/data_iter_s`    | Time waiting on dataloader next()    |
| `perf/model_fwd_bwd_s`| Forward + backward + grad clip       |
| `perf/optimizer_s`    | `optimizer.step()`                   |

These appear in wandb when `wandb_enabled=True`.

## 9. Checkpoints

```
checkpoints/<project>/<exp_name>/
├── step_5000/
│   ├── __0_0.distcp           # FSDP sharded model
│   ├── __1_0.distcp           # one .distcp per rank
│   ├── ...
│   ├── .metadata              # torch.distributed.checkpoint metadata
│   ├── optim.pt               # optimizer state
│   ├── ema.pt                 # EMA shadow params (if enabled)
│   └── train_state.pt         # global_step, epoch
├── step_10000/
└── step_15000/
```

Save cadence is controlled by `save_step_interval` (steps) and `save_epoch_interval`
(epochs); either or both can be enabled.

For loading these checkpoints from non-FSDP code (e.g. offline inference), use
`scripts/deployment/normalizer.py::MetadataNormalizingPlanner` to strip the
`_orig_mod.` / `_fsdp_wrapped_module.` / `_checkpoint_wrapped_module.` key prefixes
automatically. The offline inference CLI (`scripts/inference.py`) does this for you.

## 10. Common pitfalls

1. **Forgetting to recompute norm_stats after changing the dataset.** Symptoms:
   loss explodes or plateaus immediately. Always re-run `compute_norm_stats.py` when
   the dataset content changes.
2. **`action_horizon` mismatch between training and inference.** The checkpoint
   stores no horizon metadata; mismatched horizons silently produce truncated /
   wrong-length action chunks at inference.
3. **`asset_id=None` semantics.** If `asset_id` is `None`, `norm_stats.json` is
   expected to live next to `repo_id`; otherwise it lives under `assets/<asset_id>/`.
4. **`HF_DATASETS_CACHE` race conditions on multi-GPU.** The launcher script clears
   `/tmp/hf_datasets_cache_node${NODE_RANK}` per node to avoid contention; custom
   launchers must reproduce the same cleanup.
5. **`overwrite=True` without backup.** This wipes the entire `checkpoint_base_dir/
   <exp_name>` directory at startup. Pair with `--resume` instead when continuing.

## See also

- [Architecture](./architecture.md) — model internals
- [Datasets](./datasets.md) — `repack_transform`, norm_stats
- [Deployment](./deployment.md) — serving the trained checkpoint
