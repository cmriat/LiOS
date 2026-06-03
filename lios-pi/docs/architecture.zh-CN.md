# 模型架构

[English](./architecture.md) | [中文](./architecture.zh-CN.md)

本文档说明 `src/pi/models_pytorch/` 下的模型架构实现。本库是 [openpi](https://github.com/Physical-Intelligence/openpi)
的 PyTorch 移植版本，同时支持 **Pi0**（连续 state 作为 suffix）和 **Pi0.5**（state 离散化进语言
token + AdaRMS 条件化的 action expert）两种模式。

## 1. 总览

```
                    ┌──────────────────────────────────────┐
                    │ Observation                          │
                    │  • images: {base, left_wrist, right} │
                    │  • prompt (Pi0.5 额外含离散化 state) │
                    │  • state ∈ R^32  (历史 T 帧)         │
                    └──────────────────────────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
        ┌─────────▼─────────┐                ┌──────────▼──────────┐
        │ SigLIP            │                │ Tokenizer           │
        │ (每张图)          │                │ (文本 + 可选 state) │
        └─────────┬─────────┘                └──────────┬──────────┘
                  │ image emb                           │ lang emb
                  └─────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ PaliGemma 2B (VLM)            │  ← prefix 流
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┴────────────────────────┐
            │ 通过联合注意力共享 KV (每层 Q/K/V concat)      │
            └───────────────────────┬────────────────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────┐
        │ Gemma Expert 300M (action expert)                │  ← suffix 流
        │  • 输入: state token (Pi0) 或                    │
        │          Perceiver 压缩后的历史 (Pi0.5)          │
        │  • 流匹配的 noisy actions                        │
        │  • AdaRMS 由 timestep 条件化 (Pi0.5)             │
        └──────────────────────────┬───────────────────────┘
                                   ▼
                       action_out_proj → v_t (流速场)
                                   ▼
                       MSE(u_t = noise - actions, v_t)
```

两个 transformer 流（PaliGemma VLM + Gemma expert）在**注意力层融合**：每一层把两条流的
Q/K/V 在序列维度 concat，一次 attention 调用，然后按位置切回各自的 `o_proj` 和 MLP。这样
action expert 不用复制 KV 就能看到 VLM token。参见
`src/pi/models_pytorch/gemma_pytorch.py::compute_layer_complete`。

## 2. 模型尺寸

| 变体               | width | depth | mlp_dim | heads | kv heads | head_dim |  参数量 |
|--------------------|------:|------:|--------:|------:|---------:|---------:|-------:|
| `gemma_300m`       |  1024 |    18 |    4096 |     8 |        1 |      256 |  ~311M |
| `gemma_2b`         |  2048 |    18 |  16,384 |     8 |        1 |      256 |   ~2B  |
| `dummy` (测试用)   |    64 |     4 |     128 |     8 |        1 |       16 |     —  |

来自 `src/pi/models/gemma.py`。Pi 把它们组合起来：

| 配置                | VLM                 | Action expert       | 总参数 |
|---------------------|---------------------|---------------------|--------|
| `PiConfig` (默认)   | `gemma_2b`          | `gemma_300m`        | ~2.3B  |

Pi 相关的其他维度参数（见 `src/pi/models/pi_config.py`）：

| 字段                    | 默认值  | 说明 |
|-------------------------|--------:|------|
| `action_dim`            |      32 | action 向量维度。少于 32 维的数据集补 0。 |
| `action_horizon`        |      50 | Pi0 默认值。生产环境通常用 10–30。 |
| `max_token_len`         |  48/200 | Pi0 为 48；Pi0.5 为 200（含 state token）。 |
| `state_history_frames`  |       1 | 历史 state 帧数。> `num_latents` 时触发 Perceiver 压缩。 |
| `state_delay_frames`    |       0 | 随机 0..N 帧延迟，模拟真实环境下的 state 采集延时。 |

## 3. 流匹配 (Flow Matching)

动作采样使用 rectified flow matching，不是扩散模型。

- **时间分布**：`t ~ Beta(1.5, 1.0) * 0.999 + 0.001`（偏向小 `t`，即更接近真实动作）。
  实现见 `PI0Pytorch.sample_time`。
- **前向过程**：`x_t = t * noise + (1 - t) * actions`
- **速度目标**：`u_t = noise - actions`
- **损失**：`MSE(u_t, v_t)`，其中 `v_t` 是模型预测的速度场
- **推理**：从 `t=1` 到 `t=0` 做 `num_steps`（默认 10）步 Euler 积分：
  `x_t ← x_t + dt * v_t`，`dt = -1/num_steps`

参见 `src/pi/models_pytorch/pi0_pytorch.py` 中的 `PI0Pytorch.forward` 和
`PI0Pytorch.sample_actions`。

## 4. Pi0 vs Pi0.5

| 方面                    | Pi0                                            | Pi0.5                                                      |
|-------------------------|------------------------------------------------|------------------------------------------------------------|
| State 输入              | 连续，投影后进 expert suffix                   | 离散化后并入语言 token                                     |
| Time 条件               | `concat(action_emb, time_emb) → MLP` 融合      | **AdaRMS**：time MLP → 每层 RMSNorm 的 scale/gate          |
| `state_history_frames`  | 仅支持单帧                                     | 支持 `T ≥ 1`，附带 sinusoidal 时间位置编码                 |
| Perceiver resampler     | 不使用                                         | `state_history_frames > num_latents (=32)` 时使用          |
| `max_token_len`         | 48                                             | 200                                                        |
| 归一化                  | z-score                                        | 分位数（q01 / q99）                                        |

分支由 `PiConfig.pi05`（bool）切换。见 `PI0Pytorch.__init__` 和 `embed_suffix`。

## 5. Perceiver Resampler（Pi0.5 历史压缩）

当启用历史 state（`state_history_frames > 1`）时，Pi0.5 在 state 进入 action expert 之前，
先把长度为 `T` 的 state 嵌入序列压缩为 `M=32` 个 summary token。

```
state[B, T, 32] → state_proj → state_emb[B, T, D]
        + sinusoidal 时间位置编码（反向：index 0 = 最新）
        ↓
PerceiverResampler:
  latents[M, D] (可学) ────────┐
  state_emb[B, T, D] ──────────┴── cross-attn → self-attn → ...  (默认 2 层)
        ↓
compressed[B, M=32, D]
```

来自 `src/pi/models_pytorch/attention_pooling.py` 和 `PI0Pytorch.embed_suffix` 的关键点：

- **Latent 数量**：32（`num_latents=32`）；仅当 `T > num_latents` 时启用压缩
- **层数**：2 层 cross-attention（实测 ">2 收益递减"）
- **cross-attention 之间穿插 self-attention**（`use_self_attn=True`）
- **时间位置编码**为 sinusoidal，`min_period=1.0`，`max_period=T`；index `0` 对应**最新**帧

## 6. AdaRMS 条件化（Pi0.5）

Flow matching 的 timestep 通过 **adaptive RMSNorm** 注入 action expert，而不是与 action
embedding concat。action expert 的每个 RMSNorm 从 `time_mlp(time_emb)` 产生 scale/gate，
取代原本固定的可学 scale。

- VLM 侧：`use_adarms=False`（原版 PaliGemma）
- Action expert 侧：`use_adarms=True`，`adarms_cond_dim = action_expert.width`

config 接线见 `src/pi/models_pytorch/gemma_pytorch.py` 中的
`PaliGemmaWithExpertModel.__init__`；RMSNorm 的实际改造在 `src/pi/models/gemma_/` 下。

## 7. 双流联合注意力

每一层中，两条流各自用独立的权重算 Q/K/V，然后**在序列维度 concat** 做单次 attention 调用：

```python
# 来自 gemma_pytorch.py::compute_layer_complete (简化)
for i, hidden_states in enumerate([vlm_hidden, expert_hidden]):
    q, k, v = qkv_proj(hidden_states)        # 各流的 Q/K/V
    queries.append(q); keys.append(k); values.append(v)

Q = cat(queries, dim=seq)                     # [B, H, T_vlm + T_expert, D]
K = cat(keys, dim=seq)
V = cat(values, dim=seq)

attn_out = attention(Q, K, V, mask)           # 单次 attention 调用

for i, hidden_states in enumerate([vlm_hidden, expert_hidden]):
    out = o_proj(attn_out[:, slice_i])        # 各流的 o_proj + MLP
```

2D 注意力 mask 遵循 openpi 约定（VLM prefix 用 prefix-LM + state / time / action token
有专门规则）。mask 布局见 `make_att_2d_masks` 和 `embed_suffix`。

## 8. 推理流程

`sample_actions`（离线）和部署栈（`scripts/deployment/inference.ipynb`）走同一套流程：

1. **预处理 observation**：图像 resize 到 224×224，归一化到 `[-1, 1]`，prompt 分词
   （Pi0.5 还会加离散化的 state token）。
2. **嵌入 prefix**（image + lang）：通过 PaliGemma 一次 forward 建立 KV cache。
3. **初始化** `x_t = noise ~ N(0, I)`，形状 `[B, action_horizon, action_dim=32]`，`t = 1.0`。
4. **Euler 积分**：迭代 `num_steps` 次，
   - 用 `embed_suffix` 嵌入 `(state, x_t, t)`
   - 仅在 action expert 上跑一次 forward（prefix 复用 KV cache）
   - `x_t ← x_t + dt * v_t`，`t ← t + dt`，`dt = -1/num_steps`
5. **输出**：`t = 0` 时的 `x_t` —— 反归一化后即长度为 `action_horizon` 的动作段。

单卡 A100 / H100 级 GPU 上，典型端到端延迟约 **80–120 ms**（`num_steps=10`，`action_horizon=30`）。

## 9. 显存占用

按 `gemma_2b + gemma_300m`、bfloat16 训练、FSDP 分片到 N 张 GPU、
`batch_size_per_gpu = 24` 估算：

| 组件                    | 单卡 (bf16) |
|-------------------------|-------------|
| 模型参数（分片）        | ~4.6 GB / N |
| 优化器状态（AdamW，fp32）| ~9.2 GB / N |
| 激活值 + KV cache       | ~20–35 GB   |
| 合计                    | ~25–45 GB   |

`history_frames > 50` 或 `action_horizon > 30` 时建议开启 gradient checkpointing
（`gradient_checkpointing_enable`）。

## 相关文档

- [训练](./training.zh-CN.md) — FSDP 训练流程、queue dataloader、profiling、断点续训
- [数据集](./datasets.zh-CN.md) — `repack_transform`、norm_stats、LeRobot 接入
- [部署](./deployment.zh-CN.md) — WebRTC + WebSocket 推理栈
- [从 openpi 移植](./porting-from-openpi.zh-CN.md) — JAX → PyTorch 差异
