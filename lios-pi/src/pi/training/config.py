# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Simplified training configuration for FSDP training.

This module contains only the configuration classes needed for the FSDP training pipeline.
"""

import logging
import pathlib
import dataclasses
from typing import Any, Literal, Sequence

import tyro
import etils.epath as epath

import pi.models.model as _model
import pi.shared.download as _download
import pi.models.pi_config as pi_config
import pi.shared.normalize as _normalize
import pi.training.optimizer as _optimizer


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration for training and data loading.

    This config carries dataset-specific parameters like repo_id, norm_stats,
    action/state sequence keys, and normalization settings.
    """

    # LeRobot repo id
    repo_id: str = tyro.MISSING
    # Action and state sequence keys
    action_sequence_keys: Sequence[str] = ("actions",)
    state_sequence_keys: Sequence[str] = ("state",)
    # Asset id. If not provided, the repo id will be used.
    asset_id: str | None = None
    # Whether to apply extra delta transform
    apply_delta_transform: bool = True
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False
    # Normalization stats (loaded at runtime, optional for config definition)
    norm_stats: dict | None = None
    # Policy name (set at runtime)
    policy_name: str = ""
    test_ep_num: int = 0  # Number of episodes reserved for testing (0 means no test split)

    def load_norm_stats(self, assets_dir: epath.Path) -> dict | None:
        """Load normalization stats from the assets directory."""
        asset_id = self.asset_id or self.repo_id
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Main training configuration."""

    # ==================== Experiment Configuration ====================
    name: tyro.conf.Suppress[str]
    project_name: str = "lios-pi"
    exp_name: str = tyro.MISSING

    # ==================== Model Configuration ====================
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi_config.PiConfig)
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
    all_fp32: bool = False
    frozen: bool = False

    # ==================== Optimizer Configuration ====================
    lr_schedule: _optimizer.CosineDecaySchedule = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.AdamW = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.999  # Set to None to disable EMA.

    # ==================== Data Configuration ====================
    data: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    parent_data_dir: str | None = None  # Parent directory containing multiple datasets
    batch_size: int = 32
    num_workers: int = 2  # Recommended: min(CPU cores, 8)
    shuffle: bool = True

    # ==================== Training Configuration ====================
    seed: int = 42
    num_epochs: int = 50  # Number of training EPOCHS!
    test_step_interval: int | None = (
        None  # Run test evaluation every n STEPS! (None to disable, epoch-end testing always enabled)
    )

    # ==================== Logging and Checkpointing ====================
    log_interval: int = 100  # Run log every n STEPS!
    save_step_interval: int = 5000  # Run save checkpoint every n STEPS!
    save_epoch_interval: int | None = None  # Run save checkpoint every n EPOCHS! (None to disable)
    wandb_enabled: bool = True

    # ==================== Directory Configuration ====================
    # Base directory for config assets (e.g., norm stats)
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints
    checkpoint_base_dir: str = "./checkpoints"

    # ==================== Resume and Overwrite ====================
    # If true, will overwrite the checkpoint directory if it already exists
    overwrite: bool = False
    # If true, will resume training from the last checkpoint
    resume: bool = False

    # ==================== Policy Metadata ====================
    # Used to pass metadata to the policy server during inference
    policy_metadata: dict[str, Any] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint save directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if not self.resume and not self.overwrite:
            raise ValueError(
                "Must set either --overwrite (to start new training) or --resume (to resume from checkpoint)."
            )
