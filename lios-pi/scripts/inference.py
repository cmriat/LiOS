"""Offline inference example for the Pi (pi0 / pi05) PyTorch policy.

Loads a trained FSDP checkpoint, pulls one batch from a LeRobot dataset, runs the
flow-matching action sampler, and prints the predicted action chunk. This is the
single-shot, scriptable counterpart to scripts/deployment/inference_standalone.ju.py,
which instead serves a live robot over websockets.

Example:
    python scripts/inference.py \\
        --config-name pi05_airbot \\
        --checkpoint-dir /path/to/checkpoints/<exp>/step_10000 \\
        --repo-id /path/to/lerobot_dataset
"""

import dataclasses
import logging

import numpy as np
import torch
import torch.distributed.checkpoint
import tyro

import pi.data
import pi.models_pytorch.pi0_pytorch
from pi.models.model import Observation
from pi.training.instance_config import get_config

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    config_name: str
    """Preset training config name, e.g. pi05_airbot or pi05_robotwin."""
    checkpoint_dir: str
    """Trained FSDP checkpoint directory (torch.distributed.checkpoint format)."""
    repo_id: str | None = None
    """Optional override for the dataset path baked into the preset config."""
    num_steps: int = 10
    """Number of flow-matching denoising steps."""
    device: str = "cuda:0"


def _to_torch(item, device):
    """Recursively move a numpy/tensor tree onto `device` as torch tensors."""
    if isinstance(item, dict):
        return {key: _to_torch(value, device) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return type(item)(_to_torch(value, device) for value in item)
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item).to(device)
    if torch.is_tensor(item):
        return item.to(device)
    return item  # leave scalars / strings untouched


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    config = get_config(args.config_name)

    # Build the model on the target device and load the trained checkpoint in place.
    with torch.device(args.device):
        model = pi.models_pytorch.pi0_pytorch.PI0Pytorch(config.model)
    torch.distributed.checkpoint.load(model.state_dict(), checkpoint_id=args.checkpoint_dir)
    model.eval()
    logger.info("Loaded checkpoint from %s", args.checkpoint_dir)

    # Pull one batch from the dataset; norm stats come from config.assets_dirs.
    loader = pi.data.SimpleLeRobotLoader(config, repo_id=args.repo_id, batch_size=1)
    sample = _to_torch(next(iter(loader)), args.device)

    observation = Observation.from_dict(sample)
    bsize = observation.state.shape[0]
    noise = model.sample_noise(
        (bsize, config.model.action_horizon, config.model.action_dim),
        device=args.device,
    )
    pred_actions = model.sample_actions(
        device=args.device, observation=observation, noise=noise, num_steps=args.num_steps
    )

    logger.info("Predicted action chunk shape: %s", tuple(pred_actions.shape))
    print(pred_actions.detach().cpu().float().numpy())


if __name__ == "__main__":
    main(tyro.cli(Args))
