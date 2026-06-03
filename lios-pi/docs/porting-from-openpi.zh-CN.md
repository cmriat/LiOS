# 从 openpi 移植 (JAX → PyTorch)

[English](./porting-from-openpi.md) | [中文](./porting-from-openpi.zh-CN.md)

本文档说明本 PyTorch 库与上游 [openpi (JAX)](https://github.com/Physical-Intelligence/openpi)
的差异，以及如何把 JAX checkpoint 转换为本库可加载的格式。

## 1. 哪些是移植的，哪些是重写的

| 组件                | 状态                                   | 源文件                                |
|---------------------|----------------------------------------|---------------------------------------|
| 模型架构            | 移植，数学上一致                       | `src/pi/models_pytorch/`              |
| PaliGemma / Gemma   | 从 `transformers==4.55.0` fork         | `src/pi/models/gemma_/`、`paligemma/` |
| SigLIP              | 直接用 `transformers`                  | `src/pi/models/siglip/`               |
| Tokenizer           | SentencePiece，词表与 openpi 一致      | `src/pi/models/tokenizer.py`          |
| AdaRMS              | 在 fork 的 Gemma RMSNorm 上新增        | `src/pi/models/gemma_/modeling_gemma.py` |
| Perceiver resampler | 新增（openpi 中无对应物）              | `src/pi/models_pytorch/attention_pooling.py` |
| 流匹配              | 移植，Beta(1.5, 1.0) 时间分布相同      | `src/pi/models_pytorch/pi0_pytorch.py` |
| 数据流水线          | 移植，原生 LeRobot                     | `src/pi/data.py`                      |
| 归一化              | 移植（z-score + 分位数）               | `src/pi/shared/normalize.py`          |
| 训练循环            | 为 FSDP 重写                           | `scripts/train/train_pytorch_fsdp.py` |
| 推理                | 重写（WebRTC + WebSocket）             | `scripts/deployment/inference.ipynb`  |

模型数学部分（attention、流匹配、prefix-LM mask）目标是和 openpi 严格对齐。差异集中在
(a) 训练基础设施（JAX/TPU → PyTorch/FSDP），(b) 历史 state 的 Perceiver resampler
（本库新增）。

## 2. JAX → PyTorch 权重转换

上游 `pi0_base` 和 `pi05_base` checkpoint 是 JAX/orbax 格式。按照 openpi 的官方流程转换：

> [openpi: Converting JAX Models to PyTorch](https://github.com/Physical-Intelligence/openpi/blob/main/README.md#converting-jax-models-to-pytorch)

大致步骤：

1. 从 GCS 下载 JAX checkpoint：
   ```bash
   gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ./pi05_base_jax
   ```
2. 在 openpi 仓库下运行 `scripts/convert_jax_model_to_pytorch.py`，得到一个键名符合
   PyTorch 习惯的 `model.safetensors`。
3. 把本库 `TrainConfig.pytorch_weight_path` 指向那个 `model.safetensors` 所在目录：
   ```python
   _config.TrainConfig(
       ...,
       pytorch_weight_path="/path/to/pi05_base_pytorch",
   )
   ```

`train_pytorch_fsdp.py` 会调用 `safetensors.torch.load_model(raw_model, ..., strict=False)`
并记录任何缺失或多余的键。

### 键名差异

PyTorch 模型多了几层 wrapper，JAX 那边没有：

| Wrapper                       | 前缀                              |
|-------------------------------|-----------------------------------|
| `torch.compile`               | `_orig_mod.`                      |
| FSDP                          | `_fsdp_wrapped_module.`           |
| 激活检查点                    | `_checkpoint_wrapped_module.`     |
| 旧 TorchTitan `.module`       | `.module`                         |

加载 JAX 转换后的 checkpoint 到有 wrapper 的模型（或反过来）时，用
`scripts/deployment/normalizer.py::MetadataNormalizingPlanner`（一个
`torch.distributed.checkpoint` planner）在加载时自动剥前缀。离线推理 CLI
（`scripts/inference.py`）和部署 notebook 已经替你做好了。

## 3. 与上游 openpi 的行为差异

以下是**有意为之**的差异：

| 主题                       | openpi (JAX)                            | 本库 (PyTorch)                                      |
|----------------------------|-----------------------------------------|-----------------------------------------------------|
| 分布式框架                 | JAX sharding（`jit` + `pmap`）          | PyTorch FSDP                                        |
| 混合精度                   | 全 bfloat16                             | bfloat16 + 部分 fp32 保留列表（见下）               |
| LoRA                       | 支持                                    | **暂未移植**（config key 存在但被注释）             |
| 历史 state                 | 仅当前帧                                | `state_history_frames` 可配 + Perceiver             |
| 训练数据后端               | TFRecords + RLDS                        | LeRobot（`huggingface/lerobot`）                    |
| Norm stats                 | TF 风格                                 | LeRobot 原生（`norm_stats.json`）                   |
| 推理部署                   | 自带 BYO                                | WebRTC + WebSocket 参考栈                           |
| Action chunk 格式          | `[H, action_dim=32]`（一致）            | `[H, 32]`（一致；padding 规则相同）                 |

### fp32 保留列表

`PaliGemmaWithExpertModel.to_bfloat16_for_selected_params` 在整体 bf16 训练下仍把
以下参数强制保留 fp32：

```python
"vision_tower.vision_model.embeddings.patch_embedding.weight"
"vision_tower.vision_model.embeddings.patch_embedding.bias"
"vision_tower.vision_model.embeddings.position_embedding.weight"
"input_layernorm"
"post_attention_layernorm"
"model.norm"
```

与 openpi JAX 默认一致；小 batch 下的数值稳定性依赖这条规则。

## 4. 数值一致性验证

若要确认 PyTorch 实现与 openpi 数值一致：

1. 在 openpi 中对一个固定 observation，导出若干中间点的激活值（SigLIP 后、每层 transformer
   后、action 投影后）。
2. 把同一 observation 喂进本库的模型（用 JAX→PyTorch 转出的权重加载）。
3. 用 `torch.allclose(x_pt, x_jax_as_torch, atol=1e-3)` 对比。

我们的经验是：bf16 下 max-abs-diff 通常 `<1e-3`，fp32 下 `<1e-5`。**本库尚未内置对齐测试
fixture** —— 在 roadmap 上。

## 5. 尚未移植的部分

- **LoRA 微调**（JAX 那边有；PyTorch 这边 config 有占位符，但低秩分解没接上）
- **TPU 专属优化**（sharding 策略、`xla_jit` 路径）
- **推理时的 token 级 prompt streaming**；部署栈目前只做整 prompt 编码 + chunk 级解码

## 6. 反向：PyTorch → JAX

**不支持**。如果你用本库训了一个 checkpoint 想用 openpi 推理，需要自己写转换脚本。
大多数用户用不到。

## 相关文档

- [架构](./architecture.zh-CN.md) — 模型内部细节
- [训练](./training.zh-CN.md) — 如何在转换后的 JAX checkpoint 上做微调
- [openpi (JAX) 仓库](https://github.com/Physical-Intelligence/openpi)
