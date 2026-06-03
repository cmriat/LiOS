# pi

[English](./README.md) | [中文](./README.zh-CN.md)

A pure-PyTorch implementation of **Pi0** and **Pi0.5** vision-language-action (VLA)
models, ported from [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
with first-class FSDP training, LeRobot dataset support, and a reference WebRTC +
WebSocket inference stack.

## Package manager: pixi

This project uses [**pixi**](https://pixi.sh) for environment and dependency
management. Pixi is a Rust-based, conda-compatible package manager from
[prefix.dev](https://prefix.dev): it reuses the conda ecosystem (so CUDA, PyTorch,
and other native packages "just work") but adds a cargo-style project workflow with
deterministic `pixi.lock` files, per-environment isolation, and fast parallel
resolution. The `pixi.toml` at the repository root pins every dependency this
project needs.

Install pixi (one line, no sudo, no Python required):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

See <https://pixi.sh/latest/installation> for Windows / Homebrew / Nix alternatives.
After installation, every command below should be runnable with `pixi run -e dev …`
without any other setup.

## Quickstart

```bash
# 1. Install
pixi install -e dev
pixi run -e dev lerobot   # separate task: pixi has no --no-deps, see pixi.toml

# 2. Compute normalization statistics for your dataset (one-time)
pixi run -e dev torchrun --standalone --nproc_per_node=8 \
    scripts/train/compute_norm_stats.py pi05_airbot \
    --data.repo_id /abs/path/to/lerobot_dataset

# 3. Train — first edit the `>>> MODIFY THIS SECTION <<<` block at the top of
#    scripts/train/start_example.sh: DATA_ROOT, DATASETS, EXPERIMENT_DIR,
#    CHECKPOINT_BASE_DIR, HF_HOME (and ASSET_ID / PROJECT_NAME / POLICY_CONFIG
#    if you are not targeting the bundled pi05_airbot config). See
#    docs/training.md §3 for the full placeholder table.
zsh scripts/train/start_example.sh my_experiment 8
```

For offline single-shot inference against a trained checkpoint:

```bash
pixi run -e dev python scripts/inference.py \
    --config-name pi05_airbot \
    --checkpoint-dir /path/to/checkpoints/<exp>/step_10000
```

## Documentation

| Topic                                        | Document                                       |
|----------------------------------------------|------------------------------------------------|
| Model architecture, Pi0 vs Pi0.5, model sizes | [docs/architecture.md](./docs/architecture.md) |
| FSDP training, multi-node, resume, profiling | [docs/training.md](./docs/training.md)         |
| Adding a new dataset, norm_stats, delta actions | [docs/datasets.md](./docs/datasets.md)       |
| WebRTC + WebSocket inference stack           | [docs/deployment.md](./docs/deployment.md)     |
| JAX → PyTorch weight conversion, parity      | [docs/porting-from-openpi.md](./docs/porting-from-openpi.md) |

## Pretrained checkpoints

Public JAX checkpoints (must be converted to PyTorch — see
[porting-from-openpi.md](./docs/porting-from-openpi.md)):

- `gs://openpi-assets/checkpoints/pi0_base`
- `gs://openpi-assets/checkpoints/pi05_base`

## Repository layout

```
src/pi/
├── models/                # PaliGemma, Gemma, SigLIP (transformers fork)
├── models_pytorch/        # Pi0 / Pi0.5 models, Perceiver resampler, AdaRMS
├── training/              # config dataclasses, instance_config presets
├── shared/                # normalize, tokenizer, image_tools, download
└── data.py                # LeRobot loader, repack_transform, delta actions

scripts/
├── train/                 # FSDP trainer, queue dataloader, norm stats, profiling
├── deployment/            # WebRTC + WebSocket inference (notebook + protocol README)
└── inference.py           # offline single-shot CLI

assets/                    # norm_stats.json per dataset (asset_id)
tests/                     # unit / operator tests
```

## License & attribution

Apache-2.0. Includes code adapted from
[openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence),
[transformers](https://github.com/huggingface/transformers) (HuggingFace),
[torchtitan](https://github.com/pytorch/torchtitan) (PyTorch),
and uses [LeRobot](https://github.com/huggingface/lerobot) as the dataset backend.
See [NOTICE](./NOTICE) for the full attribution.

## Contributing

See [AGENTS.md](./AGENTS.md) for repository guidelines (code style, commit format,
PR requirements).
