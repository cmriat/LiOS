# pi

[English](./README.md) | [中文](./README.zh-CN.md)

**Pi0** 和 **Pi0.5** 视觉-语言-动作 (VLA) 模型的纯 PyTorch 实现，移植自
[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)。
原生支持 FSDP 训练、LeRobot 数据集、以及一套 WebRTC + WebSocket 的参考推理栈。

## 包管理器：pixi

本项目使用 [**pixi**](https://pixi.sh) 管理依赖和环境。pixi 是
[prefix.dev](https://prefix.dev) 开发的、用 Rust 写的 conda 兼容包管理器：它复用了
conda 生态（所以 CUDA、PyTorch 这些原生包**开箱即用**），同时引入了 cargo 风格的项目
工作流 —— 确定性 `pixi.lock`、按环境隔离、并发解析速度快。仓库根目录的 `pixi.toml`
固定了本项目需要的所有依赖。

安装 pixi（一行命令，无需 sudo，不依赖 Python）：

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Windows / Homebrew / Nix 等其他安装方式见 <https://pixi.sh/latest/installation>。
装完之后，下面所有命令直接 `pixi run -e dev …` 运行，不需要再做任何环境配置。

## 快速开始

```bash
# 1. 安装
pixi install -e dev
pixi run -e dev lerobot   # 单独 task：pixi 没有 --no-deps，详见 pixi.toml

# 2. 给数据集算一次归一化统计
pixi run -e dev torchrun --standalone --nproc_per_node=8 \
    scripts/train/compute_norm_stats.py pi05_airbot \
    --data.repo_id /abs/path/to/lerobot_dataset

# 3. 训练 —— 先按 scripts/train/start_example.sh 顶部 `>>> MODIFY THIS SECTION <<<`
#    标记块的提示，配置 DATA_ROOT、DATASETS、EXPERIMENT_DIR、CHECKPOINT_BASE_DIR、
#    HF_HOME（若不使用内置 pi05_airbot 配置，还需调整 ASSET_ID、PROJECT_NAME、
#    POLICY_CONFIG）。完整占位符表见 docs/training.zh-CN.md §3。
zsh scripts/train/start_example.sh my_experiment 8
```

对已训练的 checkpoint 执行离线单次推理：

```bash
pixi run -e dev python scripts/inference.py \
    --config-name pi05_airbot \
    --checkpoint-dir /path/to/checkpoints/<exp>/step_10000
```

## 文档

| 主题                                                | 文档                                                 |
|-----------------------------------------------------|------------------------------------------------------|
| 模型架构、Pi0 vs Pi0.5、模型尺寸                    | [docs/architecture.zh-CN.md](./docs/architecture.zh-CN.md) |
| FSDP 训练、多机、断点续训、性能分析                 | [docs/training.zh-CN.md](./docs/training.zh-CN.md)   |
| 新增数据集、norm_stats、delta action                | [docs/datasets.zh-CN.md](./docs/datasets.zh-CN.md)   |
| WebRTC + WebSocket 推理栈                           | [docs/deployment.zh-CN.md](./docs/deployment.zh-CN.md) |
| JAX → PyTorch 权重转换、数值对齐                    | [docs/porting-from-openpi.zh-CN.md](./docs/porting-from-openpi.zh-CN.md) |

## 预训练 checkpoint

公开的 JAX checkpoint（需要转换到 PyTorch，见
[porting-from-openpi.zh-CN.md](./docs/porting-from-openpi.zh-CN.md)）：

- `gs://openpi-assets/checkpoints/pi0_base`
- `gs://openpi-assets/checkpoints/pi05_base`

## 仓库结构

```
src/pi/
├── models/                # PaliGemma, Gemma, SigLIP（transformers fork）
├── models_pytorch/        # Pi0 / Pi0.5 模型、Perceiver resampler、AdaRMS
├── training/              # config 数据类、instance_config 预设
├── shared/                # normalize、tokenizer、image_tools、download
└── data.py                # LeRobot loader、repack_transform、delta action

scripts/
├── train/                 # FSDP 训练器、queue dataloader、norm stats、profiling
├── deployment/            # WebRTC + WebSocket 推理（notebook + 协议 README）
└── inference.py           # 离线单次 CLI

assets/                    # 各数据集（按 asset_id 区分）的 norm_stats.json
tests/                     # 单测 / 算子测试
```

## 许可与致谢

Apache-2.0。本仓库改编自以下项目的代码：
[openpi](https://github.com/Physical-Intelligence/openpi)（Physical Intelligence）、
[transformers](https://github.com/huggingface/transformers)（HuggingFace）、
[torchtitan](https://github.com/pytorch/torchtitan)（PyTorch），
并以 [LeRobot](https://github.com/huggingface/lerobot) 为数据后端。完整致谢见
[NOTICE](./NOTICE)。

## 贡献指南

仓库规范（代码风格、commit 格式、PR 要求）见 [AGENTS.md](./AGENTS.md)。
