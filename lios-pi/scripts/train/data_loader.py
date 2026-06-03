"""Distributed LeRobot dataloader with CUDA prefetching and image augmentation."""

import os
import glob
import logging
import pathlib
import dataclasses
from typing import Any, List

import numpy as np
import torch
from utils import init_dist as _init_dist, validate_shared_fields as _validate_shared_fields

import pi.models.model as _model  # noqa: E402
import pi.training.config as _config  # noqa: E402

# Import training configs
import pi.training.instance_config as train_config
from pi.data import MultiLeRobotLoader, _stack_tree  # noqa: E402
from pi.training.config import DatasetConfig  # noqa: E402
from pi.models_pytorch.preprocessing_pytorch import (
    IMAGE_KEYS,
    IMAGE_RESOLUTION,
    hsv_to_rgb_torch,
    rgb_to_hsv_torch,
    adjust_contrast_torch,
    adjust_brightness_torch,
)


def worker_init_fn(worker_id: int):
    # Limit each worker to a single PyTorch thread — avoids oversubscription
    # when the DataLoader uses multiple workers per GPU.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _get_local_world_size(fallback: int) -> int:
    """Best-effort detection of processes per node."""
    env_value = os.environ.get("LOCAL_WORLD_SIZE")
    if env_value is not None:
        try:
            value = int(env_value)
            if value > 0:
                return value
        except ValueError:
            logging.warning("Invalid LOCAL_WORLD_SIZE=%s, ignoring.", env_value)
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return fallback


def _build_dataset_configs(configs: List[_config.TrainConfig]) -> List[DatasetConfig]:
    """Build DatasetConfig from TrainConfig list.

    Loads normalization stats and extracts dataset-specific parameters.
    """
    ds_cfgs: List[DatasetConfig] = []
    for cfg in configs:
        # Load norm stats from assets directory
        assets_dir = cfg.assets_dirs
        norm_stats = cfg.data.load_norm_stats(assets_dir)

        if cfg.data.repo_id is None or cfg.data.repo_id == "":
            raise ValueError(f"Repo ID not set for config '{cfg.name}'.")
        if norm_stats is None:
            raise ValueError(f"Normalization stats missing for '{cfg.name}'. Run scripts/compute_norm_stats.py first.")

        # Determine if using quantile normalization (PI0.5 models use quantiles, PI0 uses z-score)
        use_quantile_norm = cfg.model.model_type != _model.ModelType.PI0

        ds_cfgs.append(
            dataclasses.replace(
                cfg.data,
                norm_stats=norm_stats,
                policy_name=cfg.name,
                use_quantile_norm=use_quantile_norm,
            )
        )
    return ds_cfgs


def build_configs_from_parent_dir(
    parent_dir: str | pathlib.Path, template: _config.TrainConfig
) -> List[_config.TrainConfig]:
    """Build a list of TrainConfig by globbing first-level subdirectories.

    Only repo_id differs across configs; all other fields follow the template.
    """
    # NOTE(critical path): use glob for first-level dirs and keep absolute paths for repo_id
    pattern = os.path.join(str(parent_dir), "*/")
    candidates = sorted(glob.glob(pattern))
    subdirs = [p for p in candidates if os.path.isdir(p)]
    if not subdirs:
        raise FileNotFoundError(f"No first-level subdirectories found under: {parent_dir}")

    cfgs: List[_config.TrainConfig] = []
    for d in subdirs:
        abs_d = os.path.abspath(d)
        # Replace only the nested DataConfigFactory.repo_id while keeping other fields intact
        new_data = dataclasses.replace(template.data, repo_id=abs_d)
        new_cfg = dataclasses.replace(template, data=new_data)
        cfgs.append(new_cfg)
        logging.info(f"Built config for repo_id: {abs_d}")
    return cfgs


def preprocess_observation_pytorch_from_dict(
    observation_dict: dict,
    *,
    train: bool = True,
    image_keys: tuple[str, ...] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> dict:
    """dict-in/dict-out variant of preprocess_observation_pytorch."""
    _ = image_resolution  # keep signature aligned with preprocess_observation_pytorch
    images = observation_dict["image"]
    image_masks = observation_dict.get("image_mask", {})
    state = observation_dict["state"]

    for key in images:
        assert hasattr(images[key], "dtype") and images[key].dtype == torch.uint8
        images[key] = images[key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

    if not set(image_keys).issubset(images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(images)}")

    batch_shape = state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = images[key]

        is_channels_first = image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)

        if train:
            image = image / 2.0 + 0.5
            if "wrist" not in key:
                height, width = image.shape[1:3]
                batch_size = image.shape[0]

                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    start_h = torch.randint(0, max_h + 1, (batch_size,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (batch_size,), device=image.device)

                    h_indices = torch.arange(crop_height, device=image.device).view(1, -1, 1)
                    w_indices = torch.arange(crop_width, device=image.device).view(1, 1, -1)

                    h_coords = start_h.view(-1, 1, 1) + h_indices
                    w_coords = start_w.view(-1, 1, 1) + w_indices

                    h_coords = h_coords.expand(batch_size, crop_height, crop_width)
                    w_coords = w_coords.expand(batch_size, crop_height, crop_width)

                    batch_indices = (
                        torch.arange(batch_size, device=image.device)
                        .view(-1, 1, 1)
                        .expand(batch_size, crop_height, crop_width)
                    )
                    image = image[batch_indices, h_coords, w_coords, :]

                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

                angles = torch.rand(batch_size, device=image.device) * 10 - 5
                angles_rad = angles * torch.pi / 180.0

                cos_angles = torch.cos(angles_rad)
                sin_angles = torch.sin(angles_rad)

                grid_x = torch.linspace(-1, 1, width, device=image.device)
                grid_y = torch.linspace(-1, 1, height, device=image.device)
                grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)
                grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)

                cos_a = cos_angles.view(batch_size, 1, 1)
                sin_a = sin_angles.view(batch_size, 1, 1)

                grid_x_rot = grid_x * cos_a - grid_y * sin_a
                grid_y_rot = grid_x * sin_a + grid_y * cos_a

                grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                image = torch.nn.functional.grid_sample(
                    image.permute(0, 3, 1, 2).to(torch.float32),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            batch_size = image.shape[0]

            hue, saturation, value = rgb_to_hsv_torch(image)
            torch.manual_seed(1234)
            brightness_params = (torch.rand(batch_size, device="cpu").to(image.device) * 2 - 1) * 0.3  # [-0.3, 0.3]
            contrast_params = (torch.rand(batch_size, device="cpu").to(image.device) * 2 - 1) * 0.4  # [-0.4, 0.4]
            value = adjust_brightness_torch(value, brightness_params)
            value = adjust_contrast_torch(value, contrast_params)

            image = hsv_to_rgb_torch(hue, saturation, value)
            image = torch.clamp(image, 0, 1)
            image = image * 2.0 - 1.0
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)

        out_images[key] = image
    out_masks = {}
    for key in out_images:
        if key not in image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=state.device)
        else:
            out_masks[key] = image_masks[key]
    out_dict = {
        "actions": observation_dict.get("actions"),
        "image": out_images,
        "image_mask": out_masks,
        "state": state,
        "tokenized_prompt": observation_dict.get("tokenized_prompt"),
        "tokenized_prompt_mask": observation_dict.get("tokenized_prompt_mask"),
        "token_ar_mask": observation_dict.get("token_ar_mask"),
        "token_loss_mask": observation_dict.get("token_loss_mask"),
    }
    return out_dict


def numpy_to_tensor(x):
    """Convert numpy or python scalar to tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (np.bool_, np.int64, np.float32, np.float64, int, float, bool)):
        return torch.tensor(x)
    return x


def collate_and_preprocess(batch_list: List[dict[str, Any]]) -> dict[str, Any]:
    """DataLoader collate_fn.

    - collate batch of dict samples
    - convert numpy → tensor
    - preprocess images, masks, prompt, state (like preprocess_observation_pytorch)
    """
    batch = _stack_tree(batch_list)

    # Run preprocess on the DataLoader worker (CPU). Disable autograd so the
    # collate path does not build a graph.
    with torch.no_grad():
        batch = preprocess_observation_pytorch_from_dict(batch)

    return batch


def tree_map_tensor(fn, x):
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, dict):
        return {k: tree_map_tensor(fn, v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(tree_map_tensor(fn, v) for v in x)
    return x


def tree_apply(fn, x):
    if torch.is_tensor(x):
        fn(x)
    elif isinstance(x, dict):
        for v in x.values():
            tree_apply(fn, v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            tree_apply(fn, v)


class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.iter = None
        self.next_batch = None
        logging.info(f"CUDAPrefetcher initialized on device {self.device}")

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            batch = tree_map_tensor(lambda t: t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t, batch)

        self.next_batch = batch

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        batch = self.next_batch

        # Tell the caching allocator these tensors are used on the current stream
        # so it does not recycle them under a different stream.
        cur_stream = torch.cuda.current_stream(device=self.device)
        tree_apply(lambda t: t.record_stream(cur_stream) if torch.is_tensor(t) else None, batch)

        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)


def create_distributed_dataloader(
    configs: List[_config.TrainConfig],
    *,
    shuffle: bool = True,
    seed: int = 0,
    gpu_rank: int = 1,
) -> dict:
    # Create concatenated dataset and shard with DistributedSampler
    rank, world, _local_rank, _device = _init_dist()
    # Validate that all configs share required fields
    _validate_shared_fields(configs)
    base_model = configs[0].model
    global_batch = int(configs[0].batch_size)
    if gpu_rank <= 0:
        raise ValueError("gpu rank must be positive")
    if global_batch % gpu_rank != 0:
        raise ValueError(f"batch_size {global_batch} must be divisible by gpu rank {gpu_rank}")
    local_batch = global_batch // gpu_rank
    ds_cfgs = _build_dataset_configs(configs)

    logging.info(f"Action horizon: {base_model.action_horizon}, dim: {base_model.action_dim}")
    has_train_test_split = configs[0].data.test_ep_num > 0

    # --- Train dataset / sampler / loader ---
    g = torch.Generator()
    g.manual_seed(seed)
    train_multi_ds = MultiLeRobotLoader(
        datasets=ds_cfgs,
        batch_size=local_batch,
        action_horizon=int(base_model.action_horizon),
        action_dim=int(base_model.action_dim),
        max_token_len=int(base_model.max_token_len),
        discrete_state_input=bool(getattr(base_model, "discrete_state_input", True)),
        apply_delta_transform=bool(getattr(configs[0].data, "apply_delta_transform", True)),
        state_history_frames=int(getattr(base_model, "state_history_frames", 1)),
        state_delay_frames=int(getattr(base_model, "state_delay_frames", 0)),
        mode="train",
    )
    total_train_samples = len(train_multi_ds)
    if rank == 0:
        logging.info(f"Total train samples across datasets: {total_train_samples}")

    local_world_size = _get_local_world_size(world)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_multi_ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=seed, drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_multi_ds,
        batch_size=local_batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None and shuffle),
        num_workers=configs[0].num_workers,
        worker_init_fn=worker_init_fn,
        prefetch_factor=None if configs[0].num_workers == 0 else 2,
        persistent_workers=True,
        generator=g,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_and_preprocess,
    )

    # --- Test dataset / sampler / loader ---
    test_loader = None
    if has_train_test_split:
        test_multi_ds = MultiLeRobotLoader(
            datasets=ds_cfgs,
            batch_size=local_batch,
            action_horizon=int(base_model.action_horizon),
            action_dim=int(base_model.action_dim),
            max_token_len=int(base_model.max_token_len),
            discrete_state_input=bool(getattr(base_model, "discrete_state_input", True)),
            apply_delta_transform=bool(getattr(configs[0].data, "apply_delta_transform", True)),
            state_history_frames=int(getattr(base_model, "state_history_frames", 1)),
            state_delay_frames=int(getattr(base_model, "state_delay_frames", 0)),
            mode="test",
        )
        total_test_samples = len(test_multi_ds)
        if rank == 0:
            logging.info(f"Total test samples across datasets: {total_test_samples}")

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_multi_ds, num_replicas=world, rank=rank, shuffle=False, seed=seed, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_multi_ds,
            batch_size=local_batch,
            sampler=test_sampler,
            shuffle=False,  # Never shuffle test set for consistent evaluation
            num_workers=configs[0].num_workers,
            prefetch_factor=None if configs[0].num_workers == 0 else 2,  # Must be None when num_workers=0
            persistent_workers=False,
            drop_last=False,
            pin_memory=False,
            collate_fn=collate_and_preprocess,
        )

    cuda_prefetcher = CUDAPrefetcher(train_loader, device=torch.device(f"cuda:{_local_rank}"))
    data_loaders = {"train_loader": cuda_prefetcher}
    if test_loader is not None:
        data_loaders["test_loader"] = test_loader
    return data_loaders

