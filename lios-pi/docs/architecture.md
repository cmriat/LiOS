# Architecture

[English](./architecture.md) | [中文](./architecture.zh-CN.md)

This document describes the model architecture as implemented in `src/pi/models_pytorch/`.
The library is a PyTorch port of [openpi](https://github.com/Physical-Intelligence/openpi)
and supports both **Pi0** (continuous state suffix) and **Pi0.5** (discrete state in language
tokens + AdaRMS-conditioned action expert).

## 1. Overview

```
                    ┌──────────────────────────────────────┐
                    │ Observation                          │
                    │  • images: {base, left_wrist, right} │
                    │  • prompt (+ discrete state, Pi0.5)  │
                    │  • state ∈ R^32  (history T frames)  │
                    └──────────────────────────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
        ┌─────────▼─────────┐                ┌──────────▼──────────┐
        │ SigLIP            │                │ Tokenizer           │
        │ (per image)       │                │ (text + opt. state) │
        └─────────┬─────────┘                └──────────┬──────────┘
                  │ image emb                           │ lang emb
                  └─────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ PaliGemma 2B (VLM)            │  ← prefix stream
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┴────────────────────────┐
            │ KV cache shared with action expert via         │
            │ joint attention (Q/K/V concat per layer)       │
            └───────────────────────┬────────────────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────┐
        │ Gemma Expert 300M (action expert)                │  ← suffix stream
        │  • input: state tokens (Pi0) or                  │
        │           Perceiver-compressed history (Pi0.5)   │
        │  • flow-matching noisy actions                   │
        │  • AdaRMS conditioned on timestep (Pi0.5)        │
        └──────────────────────────┬───────────────────────┘
                                   ▼
                       action_out_proj → v_t (flow vel.)
                                   ▼
                       MSE(u_t = noise - actions, v_t)
```

The two transformer streams (PaliGemma VLM + Gemma expert) are **fused at the attention
layer**: each layer concatenates Q/K/V from both streams along the sequence dimension,
runs a single attention call, then splits back to per-stream `o_proj` and MLP. This lets
the action expert attend to VLM tokens without copying KV across modules. See
`src/pi/models_pytorch/gemma_pytorch.py::compute_layer_complete`.

## 2. Model sizes

| Variant            | width | depth | mlp_dim | heads | kv heads | head_dim | params |
|--------------------|------:|------:|--------:|------:|---------:|---------:|-------:|
| `gemma_300m`       |  1024 |    18 |    4096 |     8 |        1 |      256 |  ~311M |
| `gemma_2b`         |  2048 |    18 |  16,384 |     8 |        1 |      256 |   ~2B  |
| `dummy` (testing)  |    64 |     4 |     128 |     8 |        1 |       16 |     —  |

Defaults from `src/pi/models/gemma.py`. Pi config combines them:

| Config              | VLM                 | Action expert       | Total |
|---------------------|---------------------|---------------------|-------|
| `PiConfig` (default)| `gemma_2b`          | `gemma_300m`        | ~2.3B |

Other Pi-specific dims (see `src/pi/models/pi_config.py`):

| Field                   | Default | Notes |
|-------------------------|--------:|-------|
| `action_dim`            |      32 | Padded action vector. Datasets with fewer joints zero-pad. |
| `action_horizon`        |      50 | Pi0 default. Production runs typically use 10–30. |
| `max_token_len`         |  48/200 | 48 for Pi0, 200 for Pi0.5 (state tokens included). |
| `state_history_frames`  |       1 | T historical states. Triggers Perceiver compression when > num_latents. |
| `state_delay_frames`    |       0 | Random 0..N frame delay simulating real-world state acquisition latency. |

## 3. Flow matching

Action sampling uses rectified flow matching, not diffusion.

- **Time distribution**: `t ~ Beta(1.5, 1.0) * 0.999 + 0.001` (biased toward small `t`,
  i.e. clearer actions). Implementation: `PI0Pytorch.sample_time`.
- **Forward process**: `x_t = t * noise + (1 - t) * actions`
- **Velocity target**: `u_t = noise - actions`
- **Loss**: `MSE(u_t, v_t)` where `v_t` is the model's predicted velocity field
- **Inference**: Euler integration from `t=1` to `t=0` over `num_steps` (default 10) steps:
  `x_t ← x_t + dt * v_t`, with `dt = -1/num_steps`

See `PI0Pytorch.forward` and `PI0Pytorch.sample_actions` in
`src/pi/models_pytorch/pi0_pytorch.py`.

## 4. Pi0 vs Pi0.5

| Aspect                  | Pi0                                          | Pi0.5                                                    |
|-------------------------|----------------------------------------------|----------------------------------------------------------|
| State input             | Continuous, projected into expert suffix     | Discretized & included in language tokens                |
| Time conditioning       | `concat(action_emb, time_emb) → MLP` fusion  | **AdaRMS**: time MLP → RMSNorm scale/gate per layer      |
| `state_history_frames`  | Single frame only                            | Supports `T ≥ 1` with sinusoidal temporal positional emb |
| Perceiver resampler     | Not used                                     | Used when `state_history_frames > num_latents (=32)`     |
| `max_token_len`         | 48                                           | 200                                                      |
| Normalization           | z-score                                      | Quantile (q01 / q99)                                     |

The branch is selected by `PiConfig.pi05` (bool). See `PI0Pytorch.__init__` and
`embed_suffix`.

## 5. Perceiver Resampler (Pi0.5 history compression)

When historical state is enabled (`state_history_frames > 1`), Pi0.5 compresses the
`T`-length sequence of state embeddings into `M=32` summary tokens before they reach the
action expert.

```
state[B, T, 32] → state_proj → state_emb[B, T, D]
        + sinusoidal temporal pos. enc. (reversed: index 0 = most recent)
        ↓
PerceiverResampler:
  latents[M, D] (learnable) ──┐
  state_emb[B, T, D] ─────────┴── cross-attn → self-attn → ...  (2 layers by default)
        ↓
compressed[B, M=32, D]
```

Key points from `src/pi/models_pytorch/attention_pooling.py` and
`PI0Pytorch.embed_suffix`:

- **Latent count**: 32 (`num_latents=32`); compression only activates when `T > num_latents`
- **Layer count**: 2 cross-attention layers (empirically ">2 gives diminishing returns")
- **Self-attention between cross-attention** is enabled (`use_self_attn=True`)
- **Temporal positional encoding** is sinusoidal with `min_period=1.0`,
  `max_period=T`; index `0` corresponds to the **most recent** frame

## 6. AdaRMS conditioning (Pi0.5)

The flow-matching timestep is injected into the action expert via **adaptive RMSNorm**
rather than concatenated with the action embedding. Each RMSNorm in the action expert
produces a scale/gate from `time_mlp(time_emb)` instead of a fixed learnable scale.

- VLM side: `use_adarms=False` (vanilla PaliGemma)
- Action expert side: `use_adarms=True`, `adarms_cond_dim = action_expert.width`

See `PaliGemmaWithExpertModel.__init__` in `src/pi/models_pytorch/gemma_pytorch.py` for the
config wiring; the actual RMSNorm modification lives in `src/pi/models/gemma_/`.

## 7. Dual-stream joint attention

For each transformer layer, both streams compute Q/K/V independently with their own
weights, then **concatenate along the sequence dimension** for a single attention call:

```python
# from gemma_pytorch.py::compute_layer_complete (simplified)
for i, hidden_states in enumerate([vlm_hidden, expert_hidden]):
    q, k, v = qkv_proj(hidden_states)        # per-stream Q/K/V
    queries.append(q); keys.append(k); values.append(v)

Q = cat(queries, dim=seq)                     # [B, H, T_vlm + T_expert, D]
K = cat(keys, dim=seq)
V = cat(values, dim=seq)

attn_out = attention(Q, K, V, mask)           # single attention call

for i, hidden_states in enumerate([vlm_hidden, expert_hidden]):
    out = o_proj(attn_out[:, slice_i])        # per-stream o_proj + MLP
```

The 2D attention mask follows the openpi convention (prefix-LM for the VLM prefix +
custom rules for state / time / action tokens). See `make_att_2d_masks` and
`embed_suffix` for the mask layout.

## 8. Inference flow

`sample_actions` (offline) and the deployment stack (`scripts/deployment/inference.ipynb`)
both follow the same recipe:

1. **Preprocess observation**: resize images to 224×224, normalize to `[-1, 1]`, tokenize
   prompt (and discrete state for Pi0.5).
2. **Embed prefix** (image + lang): pass through PaliGemma to build a KV cache.
3. **Initialize** `x_t = noise ~ N(0, I)` of shape `[B, action_horizon, action_dim=32]`,
   `t = 1.0`.
4. **Euler steps**: for `num_steps` iterations,
   - embed `(state, x_t, t)` via `embed_suffix`
   - run a single forward over the action expert (KV cache reused for prefix)
   - `x_t ← x_t + dt * v_t`, `t ← t + dt` with `dt = -1/num_steps`
5. **Output**: `x_t` at `t = 0` — denormalized action chunk of length `action_horizon`.

Typical end-to-end latency on a single A100/H100-class GPU is **80–120 ms** for
`num_steps=10`, `action_horizon=30`.

## 9. Memory footprint

Rough estimates for `gemma_2b + gemma_300m`, bfloat16 training, FSDP-sharded across N
GPUs, `batch_size_per_gpu = 24`:

| Component               | Per GPU (bf16) |
|-------------------------|----------------|
| Model params (sharded)  | ~4.6 GB / N    |
| Optimizer (AdamW, fp32) | ~9.2 GB / N    |
| Activations + KV cache  | ~20–35 GB      |
| Total                   | ~25–45 GB      |

Gradient checkpointing (`gradient_checkpointing_enable`) is available and recommended for
`history_frames > 50` or `action_horizon > 30`.

## See also

- [Training](./training.md) — FSDP recipe, queue dataloader, profiling, resume
- [Datasets](./datasets.md) — `repack_transform`, norm_stats, LeRobot integration
- [Deployment](./deployment.md) — WebRTC + WebSocket inference stack
- [Porting from openpi](./porting-from-openpi.md) — JAX → PyTorch differences
