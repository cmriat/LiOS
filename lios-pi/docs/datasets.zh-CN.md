# 数据集

[English](./datasets.md) | [中文](./datasets.zh-CN.md)

如何把一个新的机器人数据集接入训练流水线。本库以 [LeRobot](https://github.com/huggingface/lerobot)
作为数据后端；其余（key 重映射、归一化、动作 delta 变换）都在 `src/pi/` 下。

## 1. LeRobot 数据集目录结构

一个 LeRobot 数据集目录长这样：

```
my_dataset/
├── meta/
│   ├── info.json              # fps、episode 数、feature schema
│   ├── stats.json             # 原始 min/max/mean/std（LeRobot 自带）
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

本库从 `meta/info.json` 读取 FPS 和 feature schema，然后用 LeRobot 的 `LeRobotDataset`
读帧。详见 `src/pi/training/compute_norm_stats.py::BaseLeRobotDataloader`。

## 2. 四个扩展点

接入一个新数据集需要对齐四个地方：

1. **`src/pi/data.py` 里的 `repack_transform` 分支** —— 把 LeRobot 的列名映射到本库
   约定的 `observation/*` 和 `actions` 键。
2. **`src/pi/training/instance_config.py` 里的 TrainConfig 预设** —— 包含
   `policy_name`、`repo_id`、`action/state_sequence_keys`、`apply_delta_transform` 等。
3. **归一化统计** —— 每个数据集跑一次 `compute_norm_stats.py`，输出 `norm_stats.json`
   （路径依赖 `asset_id`）。
4. **可选的 policy 模块** —— 如果数据集需要推理时特殊变换（如四元数处理），在
   `src/pi/policies/` 下加 `<dataset>_policy.py`。

## 3. `repack_transform`：列名重映射

`src/pi/data.py::_repack_transform` 按 `policy_name` 分发，产出统一的 dict 布局。
约定的输出键如下：

```python
{
    "observation/cam_env":         <H,W,3> 图像，  # 必填
    "observation/cam_left_wrist":  <H,W,3> 图像，  # 必填（没有就用纯黑占位）
    "observation/cam_right_wrist": <H,W,3> 图像，  # 必填
    "observation/state":           <D,> 或 <T,D> float 向量,
    "actions":                     <H, action_dim> float（仅训练时）,
    "prompt":                      str (可选, 任务描述),
}
```

现有分支（截至本文）：

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

接入新数据集时，按 `policy_name` 子串匹配新增 `elif` 分支。如果机器人少于三个相机，
用纯黑图占位，并在同文件 `_data_inputs` 中把相应的 `image_mask` 设为 `False`。

## 4. 动作 delta 变换

`apply_delta_transform=True`（大多数 config 的默认）时，actions 会从绝对关节目标变成
相对于当前 state 的增量：

```python
actions[..., :dims] -= current_state[..., :dims] * mask
```

mask 由 `_make_bool_mask(6, -1, 6, -1)` 生成，对应双臂 6 关节 + 夹爪：每臂 6 个关节维做
delta，1 个夹爪维保持绝对值，左右臂各一份。**夹爪维有意保留绝对值**。若机器人的 DOF
布局不同（例如单臂 7 自由度），需相应调整 `_apply_delta_actions` 中的 mask。

Delta 变换发生在归一化**之前**。如果改了 mask，必须重新计算 norm_stats。

## 5. 计算归一化统计

```bash
pixi run -e dev torchrun --standalone --nproc_per_node=8 \
    scripts/train/compute_norm_stats.py \
    pi05_airbot \
    --data.repo_id /abs/path/to/dataset \
    --batch_size 64
```

对整个数据集的 `state` 和 `actions` 计算每个维度的 `mean`、`std`、`q01`、`q99`，
然后把 `norm_stats.json` 写入下面两个位置之一：

- `DatasetConfig` 上设了 `asset_id`：写到 `assets/<asset_id>/norm_stats.json`
- `asset_id=None`：写到 `<repo_id>/norm_stats.json`

`compute_norm_stats.py` 跑在 GPU 上，通过 `RunningStats`（`src/pi/shared/normalize.py`）
跨 `torchrun` 进程聚合。

### 选哪种统计量？

| 模型变体 | 归一化方式 | 使用的统计量    |
|----------|------------|-----------------|
| Pi0      | z-score    | `mean`, `std`   |
| Pi0.5    | 分位数     | `q01`, `q99`    |

按 `model.model_type` 在 `data_loader.py::_build_dataset_configs` 中自动选择。

### 批量计算多个数据集

`scripts/train/batch_compute_norm_stats.sh` 会遍历一个父目录下的所有子数据集。有多个
任务相关的数据集挂在同一根目录下时很方便。运行前先改顶部的 `BASE_DIR` 和 `SUBDATASET`。

## 6. State 历史与延迟

| 字段                   | 作用                                                                  |
|------------------------|-----------------------------------------------------------------------|
| `state_history_frames` | 输入模型的历史 state 帧数。默认 1。                                   |
| `state_delay_frames`   | 训练时随机在 [0, N] 帧间施加延迟（模拟噪声）。                        |

当 `state_history_frames > 1`：

- `observation/state` 变成 `[T, state_dim]` 张量，index `0` 是最新帧。
- 对 Pi0.5，如果 `T > num_latents (=32)`，进入 action expert 前 Perceiver Resampler 会
  把它压缩到 32 个 token。
- `state_history_frames` 必须小于最短 episode 长度，否则 dataloader 会丢掉过短的 episode。

## 7. 多数据集训练

`TrainConfig.parent_data_dir` 允许在多个**同 schema** 数据集的并集上训练。启动脚本
会把每个数据集 symlink 到本次实验目录下，然后 `build_configs_from_parent_dir` 会
glob 一级子目录，把模板 config 用每个 `repo_id` 各复制一份。

```bash
torchrun ... scripts/train/train_pytorch_fsdp.py pi05_airbot \
    --parent-data-dir /path/to/experiment_dir \
    --data.asset_id shared_norm_stats
```

各数据集必须使用**相同**的 `asset_id`（共享同一份 `norm_stats.json`）；如需加载多份
统计量，需自行扩展加载逻辑。

## 8. 添加一个新的 policy 模块

如果机器人需要推理时变换（如四元数转欧拉、自定义反归一化），在 `src/pi/policies/` 下
加一个模块。已有模块比如 `assets/pi05_airbot` 通过 `asset_id` 被引用。

policy 模块的最小接口（约定，没有抽象基类）：

1. 在 `src/pi/data.py` 中加一个 `repack_transform` 分支（同 §3）。
2. 在 `asset_id` 对应的路径下放一个 `norm_stats.json`。

推理时，部署 notebook（`scripts/deployment/inference.ipynb`）和训练用同样的
`_data_inputs` / `_normalize_array` helpers，所以只要数据集和训练时的 repack 规则一致，
推理就能自动跑通。

## 9. 可用于测试的公开数据集

| 数据集   | 说明                                                                  |
|----------|-----------------------------------------------------------------------|
| RoboTwin | 双臂仿真。`pi05_robotwin` 配置使用，数据来自 cmriat。                 |
| DROID    | `openpi-assets` 中带 norm_stats，配合 `pi0_base` 使用。               |
| LIBERO   | 单臂抓取与放置。当前**无预设配置**；需要自己加 TrainConfig +          |
|          | `repack_transform` 分支（见 §3）。                                    |

公开的 `pi0` / `pi05` 基础 checkpoint 存放在：

- `gs://openpi-assets/checkpoints/pi0_base`
- `gs://openpi-assets/checkpoints/pi05_base`

这些是 JAX checkpoint；JAX → PyTorch 的转换流程见
[porting-from-openpi](./porting-from-openpi.zh-CN.md)。

## 相关文档

- [架构](./architecture.zh-CN.md) — 模型输入 / 输出规格
- [训练](./training.zh-CN.md) — norm_stats 算完之后怎么训
- [从 openpi 移植](./porting-from-openpi.zh-CN.md) — JAX 权重转换
