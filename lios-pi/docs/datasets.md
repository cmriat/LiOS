# Datasets

[English](./datasets.md) | [中文](./datasets.zh-CN.md)

How to bring a new robot dataset into the training pipeline. The library uses
[LeRobot](https://github.com/huggingface/lerobot) as the dataset backend; everything
else (repacking, normalization, action delta transforms) lives in `src/pi/`.

## 1. LeRobot dataset layout

A LeRobot dataset directory looks like this:

```
my_dataset/
├── meta/
│   ├── info.json              # fps, episode count, feature schema
│   ├── stats.json             # raw min/max/mean/std (LeRobot-native)
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
└── videos/
    └── chunk-000/
        └── observation.images.<camera>/
            └── episode_000000.mp4
```

The library reads `meta/info.json` for the FPS and feature schema, then loads frames
via LeRobot's `LeRobotDataset`. See `src/pi/training/compute_norm_stats.py::BaseLeRobotDataloader`.

## 2. The four extension points

To add a new dataset family, four pieces need to align:

1. **A `repack_transform` branch in `src/pi/data.py`** — maps LeRobot column names to
   the canonical `observation/*` and `actions` keys this codebase expects.
2. **A preset config in `src/pi/training/instance_config.py`** — TrainConfig with
   `policy_name`, `repo_id`, `action/state_sequence_keys`, `apply_delta_transform`, etc.
3. **Normalization stats** — computed once per dataset via `compute_norm_stats.py` and
   saved as `norm_stats.json` (location depends on `asset_id`).
4. **Optional policy module** — a `<dataset>_policy.py` under `src/pi/policies/` if your
   dataset needs custom inference-time transforms (e.g. quaternion handling).

## 3. `repack_transform`: column-name remapping

`src/pi/data.py::_repack_transform` dispatches on `policy_name` to produce a uniform
dict layout. The canonical output keys are:

```python
{
    "observation/cam_env":         <H,W,3> image,   # required
    "observation/cam_left_wrist":  <H,W,3> image,   # required (use a dummy black image if N/A)
    "observation/cam_right_wrist": <H,W,3> image,   # required
    "observation/state":           <D,> or <T,D> float vector,
    "actions":                     <H, action_dim> float (training only),
    "prompt":                      str  (optional task description),
}
```

Existing branches (as of writing):

```python
# airbot / pi05_airbot
"observation.images.cam_env"          → observation/cam_env
"observation.images.cam_left_wrist"   → observation/cam_left_wrist
"observation.images.cam_right_wrist"  → observation/cam_right_wrist
"observation.state"                   → observation/state
"action"                              → actions
"task"                                → prompt

# robotwin / pi05_robotwin
"head_image"                          → observation/cam_env
"left_wrist_image"                    → observation/cam_left_wrist
"right_wrist_image"                   → observation/cam_right_wrist
"state"                               → observation/state
"action"                              → actions
"task"                                → prompt
```

To add a new dataset, append an `elif` branch keyed off a substring of your
`policy_name`. If your robot has fewer than three cameras, pass a black placeholder
and set the matching `image_mask` to `False` (see `_data_inputs` in the same file).

## 4. Action delta transform

When `apply_delta_transform=True` (default for most configs), actions are converted
from absolute joint targets to deltas relative to the current state:

```python
actions[..., :dims] -= current_state[..., :dims] * mask
```

The mask is built by `_make_bool_mask(6, -1, 6, -1)` for dual-arm 6-DoF + gripper
setups: 6 joint dims get the delta transform, 1 gripper dim stays absolute, repeated
for the second arm. **The gripper is intentionally NOT delta-transformed.** If your
robot has a different DOF layout (e.g. 7-DoF single arm), adjust the mask in
`_apply_delta_actions`.

Delta transform happens **before** normalization. If you change the mask, recompute
norm_stats.

## 5. Computing normalization statistics

```bash
pixi run -e dev torchrun --standalone --nproc_per_node=8 \
    scripts/train/compute_norm_stats.py \
    pi05_airbot \
    --data.repo_id /abs/path/to/dataset \
    --batch_size 64
```

This computes per-dimension `mean`, `std`, `q01`, `q99` for both `state` and `actions`
over the entire dataset, and writes `norm_stats.json` to either:

- `assets/<asset_id>/norm_stats.json` if `asset_id` is set on `DatasetConfig`, or
- `<repo_id>/norm_stats.json` if `asset_id=None`.

`compute_norm_stats.py` runs on GPU and aggregates statistics across all `torchrun`
processes via `RunningStats` (`src/pi/shared/normalize.py`).

### Which stat is used?

| Model variant | Normalization | Stats used      |
|---------------|---------------|-----------------|
| Pi0           | z-score       | `mean`, `std`   |
| Pi0.5         | quantile      | `q01`, `q99`    |

Selection is automatic based on `model.model_type` in
`data_loader.py::_build_dataset_configs`.

### Batch script for multiple datasets

`scripts/train/batch_compute_norm_stats.sh` iterates over subdirectories of a parent
directory. Useful when you have many task-specific datasets under one root. Edit
`BASE_DIR` and `SUBDATASET` at the top before running.

## 6. State history & delay

| Field                  | Effect                                                              |
|------------------------|---------------------------------------------------------------------|
| `state_history_frames` | Number of historical state frames to feed the model. Default 1.    |
| `state_delay_frames`   | Random delay in [0, N] frames applied at training only (sim noise). |

When `state_history_frames > 1`:

- The `observation/state` field becomes a `[T, state_dim]` tensor with index `0` being
  the most recent frame.
- For Pi0.5, if `T > num_latents (=32)`, the Perceiver Resampler compresses it to 32
  tokens before the action expert sees it.
- `state_history_frames` should be smaller than the shortest episode length, or the
  data loader will drop short episodes.

## 7. Multi-dataset training

`TrainConfig.parent_data_dir` lets you train on the union of several datasets sharing
the same schema. The launcher script symlinks each dataset directory into a per-run
folder, then `build_configs_from_parent_dir` globs first-level subdirs and replicates
the template config with each `repo_id`.

```bash
torchrun ... scripts/train/train_pytorch_fsdp.py pi05_airbot \
    --parent-data-dir /path/to/experiment_dir \
    --data.asset_id shared_norm_stats
```

Each dataset must use the **same** `asset_id` (so it loads one shared `norm_stats.json`)
unless you precompute per-dataset stats and write loader code for them.

## 8. Adding a new policy module

If your robot needs inference-time transforms (e.g. quaternion → euler conversion,
custom action denormalization), add a module under `src/pi/policies/`. Existing
modules like `assets/pi05_airbot` are referenced via `asset_id`.

The minimum interface a policy module needs (informal — there is no abstract base
class):

1. A `repack_transform` branch in `src/pi/data.py` (as above).
2. A `norm_stats.json` at the path implied by `asset_id`.

For inference, the deployment notebook (`scripts/deployment/inference.ipynb`) uses
the same `_data_inputs` / `_normalize_array` helpers as training, so as long as the
dataset matches the training-time repack rules, inference works automatically.

## 9. Public datasets you can test with

| Dataset  | Notes                                                                |
|----------|----------------------------------------------------------------------|
| RoboTwin | Dual-arm sim. Used by `pi05_robotwin` config; data from cmriat.      |
| DROID    | Norm stats provided in `openpi-assets`; for use with `pi0_base`.     |
| LIBERO   | Single-arm pick & place. No preset config currently; bring your own  |
|          | TrainConfig + `repack_transform` branch (see §3).                    |

Public `pi0` / `pi05` base checkpoints are hosted at:

- `gs://openpi-assets/checkpoints/pi0_base`
- `gs://openpi-assets/checkpoints/pi05_base`

These are JAX checkpoints; see [porting-from-openpi](./porting-from-openpi.md) for the
JAX → PyTorch conversion procedure.

## See also

- [Architecture](./architecture.md) — model inputs / outputs spec
- [Training](./training.md) — what to do after norm_stats are computed
- [Porting from openpi](./porting-from-openpi.md) — JAX weight conversion
