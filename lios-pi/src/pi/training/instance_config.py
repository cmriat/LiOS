# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Training configurations for different datasets and models.

This file contains preset training configurations that can be easily modified
and referenced by the training scripts.
"""

import tyro
import pi.training.config as _config
import pi.models.pi_config as pi_config
import pi.training.optimizer as _optimizer


# Predefined training configurations
_CONFIGS = [
    _config.TrainConfig(
        name="pi05_airbot",
        exp_name="SET_FOR_YOUR_EXPERIMENT",
        model=pi_config.PiConfig(
            pi05=True, action_horizon=10, discrete_state_input=True, state_history_frames=1, state_delay_frames=0
        ),
        data=_config.DatasetConfig(
            repo_id="/path/to/lerobot_dataset",
            asset_id="airbot",
            apply_delta_transform=True,
            action_sequence_keys=("action",),
            state_sequence_keys=("observation.state",),
            test_ep_num=0,
        ),
        parent_data_dir=None,
        save_step_interval=5000,
        save_epoch_interval=1,
        checkpoint_base_dir="/path/to/checkpoints/pi05_airbot",
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5000,
            peak_lr=3e-5,
            decay_steps=30_000 + 10_000,
            decay_lr=1e-5,
        ),
        num_workers=1,
        log_interval=10,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path="/path/to/pi05_base_pytorch",
        overwrite=True,
        resume=False,
        num_epochs=50,
        test_step_interval=1000,
    ),
    _config.TrainConfig(
        name="pi05_robotwin",
        exp_name="SET_FOR_YOUR_EXPERIMENT",
        model=pi_config.PiConfig(
            pi05=True, action_horizon=10, discrete_state_input=True, state_history_frames=200, state_delay_frames=0
        ),
        pytorch_weight_path="/path/to/pi05_base_pytorch",
        overwrite=True,
        resume=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5000,
            peak_lr=3e-5,
            decay_steps=30_000 + 10_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        data=_config.DatasetConfig(
            repo_id="/path/to/lerobot_dataset",
            asset_id=None,  # None: repo_id is an absolute path; norm_stats live alongside it
            apply_delta_transform=True,
            action_sequence_keys=("action",),
            state_sequence_keys=("state",),
            test_ep_num=5,  # Reserve 10 episodes for testing
        ),
        batch_size=64,
        num_workers=0,
        shuffle=True,
        seed=42,
        num_epochs=175,
        log_interval=10,
        save_step_interval=1000,
        save_epoch_interval=5,
        checkpoint_base_dir="/path/to/checkpoints/pi05_robotwin",
        test_step_interval=1000,
    ),
]


########################################################


def cli():
    """CLI entrypoint using tyro that allows selecting and overriding configs."""
    configs_dict = {config.name: config for config in _CONFIGS}
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in configs_dict.items()})


def get_config(config_name: str) -> _config.TrainConfig:
    """Get a config by name."""
    configs_dict = {config.name: config for config in _CONFIGS}
    return configs_dict[config_name]
