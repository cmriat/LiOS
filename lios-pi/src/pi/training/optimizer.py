# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Optimizer utilities for training."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule:
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6


@dataclasses.dataclass(frozen=True)
class AdamW:
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0
