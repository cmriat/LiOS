# 训练

[English](./training.md) | [中文](./training.zh-CN.md)

支持单机 / 多机 FSDP 训练，数据走 LeRobot 后端；可选 LMDB 队列预处理流水线；附带 EMA、
梯度检查点、Torch profiler 等。

> **无 GPU 提示**。FSDP 训练器和 `compute_norm_stats.py` 都要求 CUDA，没有 CPU 回退路径。
> 继续之前请准备好至少一张 CUDA 12.x 的 GPU。

## 1. 环境准备

```bash
# 一次性环境安装
pixi install -e dev
pixi run -e dev lerobot   # 从固定 git commit 安装 LeRobot；走单独 task 是因为 pixi
                          # 没有 pip --no-deps 等价物（详见 pixi.toml 的
                          # [tasks.lerobot]）
```

`pixi.toml` 依赖说明：

- 其中一个 conda channel（`https://srgconda.bj.bcebos.com/`）由 cmriat 维护在百度 BOS 上，
  提供 conda-forge 没有的 CUDA 构建（pytorch、flash-attn、transformer-engine、
  deepep/deepgemm/grouped-gemm/liger-kernel）。**公开可读，无需凭据**。
- `transformers==4.55.0` **固定版本**。`src/pi/models/` 下的 Gemma 模型代码是从这个具体版本
  fork 出来的；不重新 fork 就升级 transformers 会破坏 AdaRMS。
- `lerobot` 通过 pip 从固定 commit 安装（`huggingface/lerobot@0cf86487`）。

## 2. 资源需求

以下估算基于 `scripts/train/start_example.sh` 的默认参数
（`gemma_2b + gemma_300m`、bfloat16、FSDP）：

| 配置             | GPU     | 单卡 batch | 全局 batch | 单卡显存 | 吞吐 |
|------------------|---------|-----------|------------|----------|------|
| 单机 A100        | 8×80GB  | 24        | 192        | ~35 GB   | ~1.8 step/s |
| 单机 H100        | 8×80GB  | 24        | 192        | ~30 GB   | ~3.0 step/s |
| 双机 A100        | 16×80GB | 24        | 384        | ~35 GB   | ~3.5 step/s |

训练耗时粗估（`num_epochs=50`，约 20 万步）：

- LIBERO 规模（~5 万帧，250 样本）：8×A100 约 **6–10 小时**
- AirBot 小规模（~250 episode）：8×A100 约 **12–24 小时**
- RoboTwin 全量：16×A100 约 **3–7 天**

按显存预算调整 `batch_size`、`num_workers`、`action_horizon`。OOM 时按优先级处理：
(1) 降低 `batch_size`；(2) 降低 `action_horizon`；(3) 打开梯度检查点；
(4) 降低 `state_history_frames`。

## 3. 快速开始：单机 8 卡训练

```bash
# 1. 为数据集计算归一化统计（每个数据集做一次）
pixi run -e dev torchrun --standalone --nproc_per_node=8 \
    scripts/train/compute_norm_stats.py \
    pi05_airbot \
    --data.repo_id /abs/path/to/lerobot_dataset

# 2. 修改 scripts/train/start_example.sh 顶部 `>>> MODIFY THIS SECTION <<<`
#    标记块。完整占位符列表见下文 §3.1。

# 3. 启动
zsh scripts/train/start_example.sh my_experiment 8
```

### 3.1 `start_example.sh` 中的占位符

脚本默认值包含一组占位符，首次运行前**必须**替换。其中 5 项为硬占位（保留默认值
会在运行时报错）；另外 3 项与内置的 `pi05_airbot` 配置绑定，仅在切换数据集 / 策略
配置时才需要调整。

| 变量                  | 脚本默认值                    | 替换为                                                        | 是否必需                 |
|-----------------------|-------------------------------|--------------------------------------------------------------|-------------------------|
| `DATA_ROOT`           | `/path/to/your/dataset`       | `DATASETS=()` 列出的数据集所在的父目录                       | 是                      |
| `DATASETS=( ... )`    | `your-dataset-1`、`your-dataset-2` | `DATA_ROOT` 下的 LeRobot 数据集目录名                        | 是                      |
| `EXPERIMENT_DIR`      | `/path/to/experiments`        | 实验 symlink 目录的创建位置                                  | 是                      |
| `CHECKPOINT_BASE_DIR` | `/path/to/checkpoints`        | `torch.distributed.checkpoint` shard 输出根目录              | 是                      |
| `HF_HOME`             | `/path/to/hf_cache`           | HuggingFace 缓存目录；脚本同时设置 `HF_HUB_OFFLINE=1`，因此该目录中必须已包含所需的缓存内容 | 是 |
| `ASSET_ID`            | `airbot`                      | `assets/<asset_id>/norm_stats.json` 查找键                   | 非 airbot 时必改         |
| `PROJECT_NAME`        | `pi05_airbot`                 | wandb 项目名                                                 | 可选                    |
| `POLICY_CONFIG`       | `pi05_airbot`                 | `src/pi/training/instance_config.py` 中的 config 名          | 切换 config 时必改       |

启动脚本会在 `EXPERIMENT_DIR` 下为本次实验建一个 symlink 目录，然后调用
`torchrun … scripts/train/train_pytorch_fsdp.py <POLICY_CONFIG>`，把 config 名
（如 `pi05_airbot`）以及 lr / batch / checkpoint 路径等覆盖项传进去。

Config 定义在 `src/pi/training/instance_config.py`，CLI 用 [tyro](https://github.com/brentyi/tyro)，
所以 `TrainConfig` / `DatasetConfig` / `PiConfig` 上的任意字段都可以在命令行覆盖：

```bash
torchrun ... scripts/train/train_pytorch_fsdp.py pi05_airbot \
    --batch-size 16 \
    --num-epochs 30 \
    --model.action_horizon 30 \
    --data.repo_id /abs/path/to/dataset \
    --data.test_ep_num 10
```

## 4. 多机训练

`start_example.sh` 已经支持 SLURM 和 Kubernetes 风格的多机部署：

```bash
# SLURM
sbatch --nodes=2 --gres=gpu:8 scripts/train/start_example.sh my_exp 8

# 通用启动器读取的环境变量:
#   NNODES        (来自 SLURM_NNODES)
#   NODE_RANK     (来自 SLURM_PROCID 或 JOB_COMPLETION_INDEX)
#   MASTER_ADDR   (来自 SLURM_JOB_FIRST_NODE_IP)
#   MASTER_PORT   (默认 29500)
```

`start_example.sh` 中的 NCCL 配置块**针对特定集群调优**，须根据目标集群的网卡与 InfiniBand 配置调整：

```bash
export NCCL_SOCKET_IFNAME="eth0"     # ← 替换为目标集群的网卡名
export NCCL_IB_GID_INDEX="3"         # ← 集群相关
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIME_OUT="22"
```

单机训练时整块删掉即可。

## 5. 数据加载

标准 PyTorch `DataLoader` + 多 worker 预处理 + 分布式采样 + CUDA stream prefetcher。
通过 `TrainConfig` 调参：

```python
# pi05_airbot 配置片段
num_workers=8,
batch_size=24,
shuffle=True,
```

`scripts/train/data_loader.py` 里的 `CUDAPrefetcher` 把 host→device 拷贝和计算重叠；
pinned memory + persistent workers 默认开。图像增强（随机裁剪、旋转、亮度 / 对比度）
在 collate 函数里跑 GPU。

## 6. 分布式训练栈

| 层                  | 实现                                                            |
|---------------------|-----------------------------------------------------------------|
| 参数分片            | FSDP（`scripts/train/utils.py::fsdp_wrap`）                     |
| 编译                | `torch.compile(raw_model, fullgraph=True)`，在 FSDP wrap **之前** |
| 混合精度            | bfloat16 forward / fp32 主权重（FSDP `MixedPrecision`）          |
| 梯度裁剪            | 跨 device mesh 的梯度裁剪（`clip_grad_norm_`）                   |
| EMA                 | shadow 参数走 `torch._foreach_lerp_`，默认 `decay=0.999`         |
| Checkpoint 格式     | `torch.distributed.checkpoint`（分片，多文件）                   |
| 激活检查点          | 通过 `model.gradient_checkpointing_enable()` 选择性开启          |

### 保留为 fp32 的关键参数

即使开 bfloat16 训练，PaliGemma 中一部分参数仍强制保留 fp32：patch embedding、
position embedding、所有 `*_layernorm` 权重。见
`PaliGemmaWithExpertModel.to_bfloat16_for_selected_params`。

## 7. 断点续训

```bash
zsh scripts/train/start_example.sh my_exp 8 \
    --no-overwrite --resume
```

恢复粒度到 **step 级**：训练器通过
`start_epoch = global_step // steps_per_epoch` 算出 epoch，然后跳过被恢复 epoch 内前
`global_step % steps_per_epoch` 个 batch。`sampler.set_epoch(epoch)` 会被调用，shuffle
仍然是确定的。

EMA 和 optimizer state 也会被恢复。见 `scripts/train/utils.py::resume_from_fsdp_model_checkpoint`。

## 8. 性能分析

Torch profiler 和 CUDA memory snapshot 已接好，但**默认关闭**。在
`train_pytorch_fsdp.py` 中切换：

```python
profiling_config = ProfilingConfig(
    enable_profiling=True,                                        # ← 打开
    save_traces_folder=f"./traces_{base_config.batch_size}",
    profile_freq=10,
    enable_memory_snapshot=True,                                  # ← 打开
    save_memory_snapshot_folder=f"./memory_snapshot_{base_config.batch_size}",
)
```

traces 是按 rank 导出的 Chrome trace JSON；memory snapshot 是 pickle dump，可加载到
`https://pytorch.org/memory_viz` 查看。

### 单步耗时统计（始终上报到 wandb）

`train_pytorch_fsdp.py` 用 CUDA event 记录每步的四项指标：

| 指标                  | 含义                                |
|-----------------------|-------------------------------------|
| `perf/step_total_s`   | 单步端到端 wall time                |
| `perf/data_iter_s`    | 等 dataloader next() 的耗时         |
| `perf/model_fwd_bwd_s`| forward + backward + 梯度裁剪       |
| `perf/optimizer_s`    | `optimizer.step()`                  |

`wandb_enabled=True` 时这些会出现在 wandb 面板。

## 9. Checkpoint 结构

```
checkpoints/<project>/<exp_name>/
├── step_5000/
│   ├── __0_0.distcp           # FSDP 分片模型
│   ├── __1_0.distcp           # 每个 rank 一个 .distcp
│   ├── ...
│   ├── .metadata              # torch.distributed.checkpoint 元数据
│   ├── optim.pt               # optimizer state
│   ├── ema.pt                 # EMA shadow 参数（启用时）
│   └── train_state.pt         # global_step、epoch
├── step_10000/
└── step_15000/
```

保存频率由 `save_step_interval`（按步）和 `save_epoch_interval`（按 epoch）控制，可同时开启。

非 FSDP 代码（如离线推理）加载这些 checkpoint 时，用
`scripts/deployment/normalizer.py::MetadataNormalizingPlanner` 自动剥掉
`_orig_mod.` / `_fsdp_wrapped_module.` / `_checkpoint_wrapped_module.` 这些前缀。
离线推理 CLI（`scripts/inference.py`）已经替你处理好。

## 10. 常见坑

1. **改了数据集忘了重新计算 norm_stats**。症状：loss 立即爆掉或一直 plateau。数据集内容
   变化后**必须**重新跑 `compute_norm_stats.py`。
2. **训练和推理的 `action_horizon` 不一致**。checkpoint 里不存 horizon 元数据，不一致时
   推理输出的 action 段会被静默截断或长度错乱。
3. **`asset_id=None` 的含义**。`asset_id` 为 `None` 时，`norm_stats.json` 必须放在
   `repo_id` 同级目录；否则需要放在 `assets/<asset_id>/` 下。
4. **多卡下的 `HF_DATASETS_CACHE` 竞争**。启动脚本会按 node 清理
   `/tmp/hf_datasets_cache_node${NODE_RANK}` 以规避竞争；自定义启动脚本须保留同等处理。
5. **`overwrite=True` 没备份就跑**。启动时会清空整个
   `checkpoint_base_dir/<exp_name>` 目录。续训时请改用 `--resume`。

## 相关文档

- [架构](./architecture.zh-CN.md) — 模型内部细节
- [数据集](./datasets.zh-CN.md) — `repack_transform`、norm_stats
- [部署](./deployment.zh-CN.md) — 训练完的 checkpoint 如何上线
