# Porting from openpi (JAX → PyTorch)

[English](./porting-from-openpi.md) | [中文](./porting-from-openpi.zh-CN.md)

This document explains how this PyTorch library differs from upstream
[openpi (JAX)](https://github.com/Physical-Intelligence/openpi) and how to convert
a JAX checkpoint into a format this library can load.

## 1. What was ported, what was rewritten

| Component             | Status                              | Source                                |
|-----------------------|-------------------------------------|---------------------------------------|
| Model architecture    | Ported, byte-identical math         | `src/pi/models_pytorch/`              |
| PaliGemma / Gemma     | Forked from `transformers==4.55.0`  | `src/pi/models/gemma_/`, `paligemma/` |
| SigLIP                | Used from `transformers` directly   | `src/pi/models/siglip/`               |
| Tokenizer             | SentencePiece, same vocab as openpi | `src/pi/models/tokenizer.py`          |
| AdaRMS                | Added to forked Gemma RMSNorm       | `src/pi/models/gemma_/modeling_gemma.py` |
| Perceiver resampler   | New (no openpi equivalent)          | `src/pi/models_pytorch/attention_pooling.py` |
| Flow matching         | Ported, same Beta(1.5, 1.0) schedule| `src/pi/models_pytorch/pi0_pytorch.py` |
| Data pipeline         | Ported, LeRobot-native              | `src/pi/data.py`                      |
| Normalization         | Ported (z-score + quantile)         | `src/pi/shared/normalize.py`          |
| Training loop         | Rewritten for FSDP                  | `scripts/train/train_pytorch_fsdp.py` |
| Inference             | Rewritten (WebRTC + WebSocket)      | `scripts/deployment/inference.ipynb`  |

The model math (attention, flow matching, prefix-LM masking) is intended to match
openpi exactly. Differences are concentrated in (a) the training infrastructure
(JAX/TPU → PyTorch/FSDP) and (b) the historical-state Perceiver resampler, which is
a new contribution in this library.

## 2. JAX → PyTorch weight conversion

The upstream `pi0_base` and `pi05_base` checkpoints are JAX/orbax format. Convert them
to PyTorch using the procedure documented in openpi:

> [openpi: Converting JAX Models to PyTorch](https://github.com/Physical-Intelligence/openpi/blob/main/README.md#converting-jax-models-to-pytorch)

The high-level steps:

1. Download a JAX checkpoint from GCS:
   ```bash
   gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ./pi05_base_jax
   ```
2. Run openpi's `scripts/convert_jax_model_to_pytorch.py` (in the openpi repo) which
   produces a `model.safetensors` file with PyTorch-style key names.
3. Point this library's `TrainConfig.pytorch_weight_path` at the directory containing
   that `model.safetensors`:
   ```python
   _config.TrainConfig(
       ...,
       pytorch_weight_path="/path/to/pi05_base_pytorch",
   )
   ```

`train_pytorch_fsdp.py` calls `safetensors.torch.load_model(raw_model, ..., strict=False)`
and logs any missing / unexpected keys.

### Key-name differences

The PyTorch model adds a few wrappers that JAX did not have:

| Wrapper                       | Adds prefix                       |
|-------------------------------|-----------------------------------|
| `torch.compile`               | `_orig_mod.`                      |
| FSDP                          | `_fsdp_wrapped_module.`           |
| Activation checkpointing      | `_checkpoint_wrapped_module.`     |
| Old TorchTitan `.module`      | `.module`                         |

When loading a JAX-converted checkpoint into a wrapped model, or vice versa, use
`scripts/deployment/normalizer.py::MetadataNormalizingPlanner` (a
`torch.distributed.checkpoint` planner) to strip these prefixes at load time. The
offline inference CLI (`scripts/inference.py`) and the deployment notebook already
do this.

## 3. Behavioural differences vs upstream openpi

These are **intentional** divergences:

| Topic                       | openpi (JAX)                          | This library (PyTorch)                              |
|-----------------------------|---------------------------------------|-----------------------------------------------------|
| Distributed framework       | JAX sharding (`jit` + `pmap`)         | PyTorch FSDP                                        |
| Mixed precision             | bfloat16 throughout                   | bfloat16 + selected fp32 keep-list (see below)      |
| LoRA                        | Supported                             | **Not yet ported** (config keys exist but commented)|
| Historical state            | Single frame                          | Configurable `state_history_frames` + Perceiver     |
| Training data backend       | TFRecords + RLDS                      | LeRobot (`huggingface/lerobot`)                     |
| Norm stats                  | TF-style                              | LeRobot-native (`norm_stats.json`)                  |
| Inference serving           | Bring-your-own                        | WebRTC + WebSocket reference stack                  |
| Action chunk format         | `[H, action_dim=32]` (same)           | `[H, 32]` (same; padding rules identical)           |

### fp32 keep-list

`PaliGemmaWithExpertModel.to_bfloat16_for_selected_params` keeps the following params
in fp32 even when overall precision is bf16:

```python
"vision_tower.vision_model.embeddings.patch_embedding.weight"
"vision_tower.vision_model.embeddings.patch_embedding.bias"
"vision_tower.vision_model.embeddings.position_embedding.weight"
"input_layernorm"
"post_attention_layernorm"
"model.norm"
```

This matches the openpi JAX defaults and is required for numerical stability on
small batches.

## 4. Verifying parity

If you want to confirm that the PyTorch implementation matches openpi numerically:

1. Save activations from openpi at a few known intermediate points (post-SigLIP, after
   each transformer layer, post-action-projection) for a fixed observation.
2. Run the same observation through this library's model loaded from a
   JAX-converted checkpoint.
3. Compare with `torch.allclose(x_pt, x_jax_as_torch, atol=1e-3)`.

In our experience the max-abs-diff is typically `<1e-3` in bf16 and `<1e-5` in fp32.
We do **not** ship a parity test fixture — this is on the roadmap.

## 5. What is NOT yet ported

- **LoRA fine-tuning** (the JAX paths exist; PyTorch config has placeholders but the
  actual low-rank decomposition is not wired up).
- **TPU-specific optimizations** (sharding strategies, `xla_jit` paths).
- **Token-level prompt streaming** during inference; the deployment stack only does
  full-prompt encode + chunk-level decode.

## 6. Going the other direction

Going PyTorch → JAX is **not** supported. If you trained a checkpoint with this
library and want to use openpi for inference, you would need to write your own
weight conversion script. Most users do not need this.

## See also

- [Architecture](./architecture.md) — model internals
- [Training](./training.md) — how to fine-tune a converted JAX checkpoint
- [openpi (JAX) repository](https://github.com/Physical-Intelligence/openpi)
