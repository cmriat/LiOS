# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Bare-bones LeRobot data loader without transform abstractions.

Also includes a multi-dataset variant that can iterate and batch
across multiple LeRobot repos using a simple per-dataset config.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import numpy as np
import torch
import einops
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from pi.models import tokenizer as tokenizer_mod
from pi.shared import image_tools
from pi.training import config as _config
from pi.training.config import DatasetConfig


def _repack_transform(policy_name: str, sample: dict) -> dict:
    if "airbot" in policy_name.lower():
        result: dict[str, object] = {
            "observation/cam_env": sample["observation.images.cam_env"],
            "observation/cam_left_wrist": sample["observation.images.cam_left_wrist"],
            "observation/cam_right_wrist": sample["observation.images.cam_right_wrist"],
            "observation/state": sample["observation.state"],
        }
        if "action" in sample:
            result["actions"] = sample["action"]
        if "task" in sample:
            result["prompt"] = sample["task"]
    elif "robotwin" in policy_name.lower():
        result: dict[str, object] = {
            "observation/cam_env": sample["head_image"],
            "observation/cam_left_wrist": sample["left_wrist_image"],
            "observation/cam_right_wrist": sample["right_wrist_image"],
            "observation/state": sample["state"],
        }
        if "action" in sample:
            result["actions"] = sample["action"]
        if "task" in sample:
            result["prompt"] = sample["task"]
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")

    return result


def _make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        _make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        _make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * dim)
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _apply_delta_actions(step: dict, mask: tuple[bool, ...]) -> dict:
    """将绝对动作转换为相对动作(delta actions).

    参考 transforms.DeltaActions 的实现。
    这个转换会在训练时应用，将 actions 从绝对空间转换为相对于 state 的增量。

    Args:
        step: 包含 'state' 和 'actions' 的数据字典
        mask: 布尔掩码，指定哪些动作维度需要转换为 delta

    Returns:
        转换后的数据字典
    """
    if "actions" not in step:
        return step

    state, actions = step["state"], step["actions"]
    mask_array = np.asarray(mask)
    dims = mask_array.shape[-1]

    # If state has multiple frames (history), use only the first frame (most recent state)
    # state shape could be (history_frames, state_dim) or (state_dim,)
    current_state = state[0] if state.ndim > 1 else state

    actions[..., :dims] -= np.expand_dims(np.where(mask_array, current_state[..., :dims], 0), axis=-2)
    step["actions"] = actions

    return step


def _data_inputs(data: dict) -> dict:
    def _to_uint8_image(array: np.ndarray) -> np.ndarray:
        image = np.asarray(array)
        if np.issubdtype(image.dtype, np.floating):
            image = (255.0 * image).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        return image.astype(np.uint8, copy=False)

    result = {
        "state": np.asarray(data["observation/state"]),
        "image": {
            "base_0_rgb": _to_uint8_image(data["observation/cam_env"]),
            "left_wrist_0_rgb": _to_uint8_image(data["observation/cam_left_wrist"]),
            "right_wrist_0_rgb": _to_uint8_image(data["observation/cam_right_wrist"]),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "prompt": data.get("prompt"),
    }
    if "actions" in data:
        result["actions"] = np.asarray(data["actions"])
    return result


def _normalize_array(x: np.ndarray, stats, use_quantiles: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile stats required when use_quantiles=True")
        q01 = stats.q01[..., : x.shape[-1]]
        q99 = stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    mean = stats.mean[..., : x.shape[-1]]
    std = stats.std[..., : x.shape[-1]]
    return (x - mean) / (std + 1e-6)


def _resize_images(data: dict, height: int, width: int) -> dict:
    result = dict(data)
    result["image"] = {key: image_tools.resize_with_pad(img, height, width) for key, img in data["image"].items()}
    return result


def _tokenize_prompt(data: dict, tokenizer, *, discrete_state_input: bool) -> dict:
    result = dict(data)
    prompt = result.pop("prompt", None)
    if prompt is None:
        raise ValueError("Prompt is required for tokenization.")
    if not isinstance(prompt, str):
        prompt = str(prompt if np.isscalar(prompt) else prompt.item())

    # For discrete_state_input: only use the most recent state frame for tokenization with prompt
    # Historical states are preserved in result["state"] for later use by the action head
    # If state has history (shape: (history_frames, state_dim)), extract state[0] (most recent)
    # If state is single frame (shape: (state_dim,)), use it directly
    state_arg = None
    if discrete_state_input:
        state = result["state"]
        if state.ndim > 1:
            # Multi-frame state: state[0] is the most recent frame (t-delay)
            # Note: with delta_timestamps, state[0] = t-state_delay_frames, state[-1] is oldest
            state_arg = state[0]
        else:
            # Single-frame state: use directly
            state_arg = state

    tokens, mask = tokenizer.tokenize(prompt, state_arg)
    result["tokenized_prompt"] = tokens
    result["tokenized_prompt_mask"] = mask
    return result


def _normalize(data: dict, norm_stats: dict, use_quantiles: bool) -> dict:
    result = dict(data)
    if "state" in result and "state" in norm_stats:
        result["state"] = _normalize_array(result["state"], norm_stats["state"], use_quantiles)
    if "actions" in result and "actions" in norm_stats:
        result["actions"] = _normalize_array(result["actions"], norm_stats["actions"], use_quantiles)
    return result


def _pad_state_actions(data: dict, target_dim: int) -> dict:
    def _pad_last_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
        array = np.asarray(array, dtype=np.float64)
        if array.shape[-1] >= target_dim:
            return array
        pad_width = [(0, 0)] * array.ndim
        pad_width[-1] = (0, target_dim - array.shape[-1])
        return np.pad(array, pad_width, constant_values=0.0)

    result = dict(data)
    result["state"] = _pad_last_dim(result["state"], target_dim)
    if "actions" in result:
        result["actions"] = _pad_last_dim(result["actions"], target_dim)
    return result


def _stack_tree(items: list[dict]) -> dict:
    def stack(*values):
        first = values[0]
        if isinstance(first, dict):
            return {key: stack(*[value[key] for value in values]) for key in first}
        if isinstance(first, (list, tuple)):
            packed = [stack(*[value[idx] for value in values]) for idx in range(len(first))]
            return type(first)(packed)
        return torch.stack([torch.as_tensor(value) for value in values], dim=0)

    return stack(*items)


class SimpleLeRobotLoader:
    """Minimal iterator that batches LeRobot samples.

    Note: `data_config` is intentionally not used here. Only the fields that were
    previously read from `data_config` are accepted directly via constructor
    parameters to avoid requiring a full TrainConfig initialization.
    """

    def __init__(
        self,
        config: _config.TrainConfig | None = None,
        *,
        # Fields previously sourced from data_config
        repo_id: str | None = None,
        action_sequence_keys: list[str] | None = None,
        state_sequence_keys: list[str] | None = None,
        norm_stats: dict | None = None,
        use_quantile_norm: bool | None = None,
        policy_name: str | None = None,
        # Fields previously sourced from model/batch config
        batch_size: int | None = None,
        action_horizon: int | None = None,
        action_dim: int | None = None,
        max_token_len: int | None = None,
        discrete_state_input: bool | None = None,
        apply_delta_transform: bool | None = None,
        state_history_frames: int | None = None,
        state_delay_frames: int | None = None,
        test_ep_num: int | None = None,
        mode: str = "train",  # "train" or "test"
    ) -> None:
        # Support both explicit-args path and legacy config path.
        if config is not None:
            # Extract policy name from config if not provided
            if policy_name is None:
                policy_name = config.name

            # Extract non-data fields from TrainConfig.
            batch_size = config.batch_size if batch_size is None else batch_size
            action_horizon = config.model.action_horizon if action_horizon is None else action_horizon
            action_dim = config.model.action_dim if action_dim is None else action_dim
            max_token_len = config.model.max_token_len if max_token_len is None else max_token_len
            if discrete_state_input is None:
                discrete_state_input = getattr(config.model, "discrete_state_input", True)
            if state_history_frames is None:
                state_history_frames = getattr(config.model, "state_history_frames", 1)
            if state_delay_frames is None:
                state_delay_frames = getattr(config.model, "state_delay_frames", 0)

            # Temporarily build data_config to fetch needed fields, but do not store it.
            tmp_dc = config.data.create(config.assets_dirs, config.model)
            repo_id = tmp_dc.repo_id if repo_id is None else repo_id
            action_sequence_keys = (
                list(tmp_dc.action_sequence_keys) if action_sequence_keys is None else action_sequence_keys
            )
            state_sequence_keys = (
                list(tmp_dc.state_sequence_keys) if state_sequence_keys is None else state_sequence_keys
            )
            norm_stats = tmp_dc.norm_stats if norm_stats is None else norm_stats
            use_quantile_norm = tmp_dc.use_quantile_norm if use_quantile_norm is None else use_quantile_norm

            # Read apply_delta_transform from data config
            apply_delta_transform = getattr(config.data, "apply_delta_transform", True)
            if test_ep_num is None:
                test_ep_num = getattr(config.data, "test_ep_num", 0)

        # Validate required fields for explicit construction.
        if repo_id is None:
            raise ValueError("repo_id must be set")
        if norm_stats is None:
            raise ValueError("Normalization stats are required.")
        if action_sequence_keys is None:
            raise ValueError("action_sequence_keys must be set")
        if state_sequence_keys is None:
            raise ValueError("state_sequence_keys must be set")
        if batch_size is None or action_horizon is None or action_dim is None or max_token_len is None:
            raise ValueError("batch_size, action_horizon, action_dim, and max_token_len must be provided")
        if use_quantile_norm is None:
            use_quantile_norm = False
        if discrete_state_input is None:
            discrete_state_input = False
        if apply_delta_transform is None:
            apply_delta_transform = False
        if state_history_frames is None:
            state_history_frames = 1
        if state_delay_frames is None:
            state_delay_frames = 0
        if test_ep_num is None:
            test_ep_num = 0

        # Validate policy_name
        if policy_name is None:
            raise ValueError("policy_name must be provided either via config or as explicit argument")

        # Store compact state; avoid keeping train_config/data_config around.
        self.repo_id = repo_id
        self.policy_name = policy_name
        self.action_sequence_keys = list(action_sequence_keys)
        self.state_sequence_keys = list(state_sequence_keys)
        self.norm_stats = norm_stats
        self.use_quantile_norm = use_quantile_norm
        self.batch_size = int(batch_size)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.max_token_len = int(max_token_len)
        self.discrete_state_input = bool(discrete_state_input)
        self.apply_delta_transform = bool(apply_delta_transform)
        self.state_history_frames = int(state_history_frames)
        self.state_delay_frames = int(state_delay_frames)
        self.test_ep_num = int(test_ep_num)
        self.mode = str(mode)

        if self.mode not in ["train", "test"]:
            raise ValueError(f"mode must be 'train' or 'test', got '{self.mode}'")

        metadata = lerobot_dataset.LeRobotDatasetMetadata(self.repo_id)
        # Use dataset fps to build per-key delta timestamps of length action_horizon.
        delta_timestamps = {
            key: [t / metadata.fps for t in range(self.action_horizon)] for key in self.action_sequence_keys
        }

        # Add state history frames: state at [current-delay, current-delay-1, ..., current-delay-history+1]
        # For history frames, we need negative time offsets going backwards in time
        # Example: state_delay_frames=2, state_history_frames=3
        # We want states at: [t-2, t-3, t-4] (current delayed by 2, then 2 more historical frames)
        if self.state_history_frames > 1 or self.state_delay_frames > 0:
            for state_key in self.state_sequence_keys:
                # Create time offsets for historical states (negative = past frames)
                state_offsets = []
                for i in range(self.state_history_frames):
                    # Negative offset: -state_delay_frames, -state_delay_frames-1, -state_delay_frames-2, ...
                    offset = -(self.state_delay_frames + i)
                    state_offsets.append(offset / metadata.fps)
                delta_timestamps[state_key] = state_offsets

        # Split episodes into train/test sets
        total_ep_num = metadata.total_episodes

        # Validate test_ep_num
        if self.test_ep_num < 0:
            raise ValueError(f"test_ep_num must be non-negative, got {self.test_ep_num}")
        if self.test_ep_num >= total_ep_num:
            raise ValueError(
                f"test_ep_num ({self.test_ep_num}) must be less than total episodes ({total_ep_num}). "
                f"At least 1 episode is required for training."
            )

        train_count = total_ep_num - self.test_ep_num

        # Use fixed random seed for reproducible splits
        rng = np.random.RandomState(42)
        total_ep_idx = rng.permutation(total_ep_num)
        train_ep_idx = total_ep_idx[:train_count]
        test_ep_idx = total_ep_idx[train_count:]

        logging.info(
            f"Dataset split for {self.repo_id}: {len(train_ep_idx)} train episodes, "
            f"{len(test_ep_idx)} test episodes (test_ep_num={self.test_ep_num})"
        )

        # Create dataset based on mode
        if self.mode == "train":
            episodes_to_use = train_ep_idx
        else:  # mode == "test"
            episodes_to_use = test_ep_idx

        self.dataset = lerobot_dataset.LeRobotDataset(
            self.repo_id,
            episodes=episodes_to_use,
            delta_timestamps=delta_timestamps,
        )

        # Fix for episode indexing bug when using episodes parameter
        # The episode_data_index is indexed by filtered episode position, but
        # _get_query_indices receives original episode indices from the data
        if episodes_to_use is not None:
            # Create mapping from original episode index to filtered position
            episode_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(episodes_to_use)}

            # Store original method
            original_get_query_indices = self.dataset._get_query_indices

            # Create patched method that maps episode indices
            def patched_get_query_indices(idx: int, ep_idx: int):
                # Map original episode index to filtered position
                mapped_ep_idx = episode_index_map.get(ep_idx, ep_idx)
                return original_get_query_indices(idx, mapped_ep_idx)

            # Replace the method
            self.dataset._get_query_indices = patched_get_query_indices

        self.tokenizer = tokenizer_mod.PaligemmaTokenizer(self.max_token_len)

    def _transform(self, sample: dict) -> dict:
        step = _repack_transform(self.policy_name, sample)
        step = _data_inputs(step)

        # Apply delta action transform before normalization if enabled
        # This converts absolute actions to delta (relative to current state)
        if self.apply_delta_transform:
            # (6, -1, 6, -1) means 6 joints + 1 gripper per arm
            # True for joint dimensions, False for gripper dimensions
            delta_action_mask = _make_bool_mask(6, -1, 6, -1)
            step = _apply_delta_actions(step, delta_action_mask)

        step = _normalize(step, self.norm_stats, self.use_quantile_norm)
        step = _resize_images(step, 224, 224)
        step = _tokenize_prompt(step, self.tokenizer, discrete_state_input=self.discrete_state_input)
        step = _pad_state_actions(step, self.action_dim)
        return step

    # Random-access API to fetch a single transformed item
    def __getitem__(self, idx: int) -> dict:
        # Now delta_timestamps handles state history and delay automatically
        # The dataset will return historical states based on the delta_timestamps we configured
        sample = self.dataset[idx]

        # _transform handles both single-frame and multi-frame data automatically through numpy broadcasting
        # - _repack_transform: just remaps keys, doesn't care about shape
        # - _normalize: uses _normalize_array which supports broadcasting for (history, dim)
        # - _pad_state_actions: pads last dimension regardless of array.ndim
        transformed = self._transform(sample)

        # If no history frames configured, add time dimension for consistency
        if not (self.state_history_frames > 1 or self.state_delay_frames > 0):
            # transformed["state"] shape: (state_dim,) -> (1, state_dim)
            transformed["state"] = np.expand_dims(transformed["state"], axis=0)
        # else: transformed["state"] already has shape (history_frames, state_dim) from delta_timestamps

        return transformed

    # Dataset length passthrough
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict]:
        batch_size = self.batch_size
        buffer: list[dict] = []

        # Now delta_timestamps handles all temporal queries automatically
        # We can use a simple iteration over the dataset
        for idx in range(len(self.dataset)):
            buffer.append(self[idx])  # Use __getitem__ which handles delta_timestamps properly
            if len(buffer) == batch_size:
                yield _stack_tree(buffer)
                buffer = []
        if buffer:
            yield _stack_tree(buffer)


class MultiLeRobotLoader:
    """Round-robin iterator that batches samples from multiple LeRobot repos.

    The global batching/model params are provided once, while dataset-specific
    parameters are carried by `datasets` (previously passed as non-config args).
    """

    def __init__(
        self,
        *,
        datasets: list[DatasetConfig],
        batch_size: int,
        action_horizon: int,
        action_dim: int,
        max_token_len: int,
        discrete_state_input: bool = False,
        apply_delta_transform: bool = True,
        state_history_frames: int = 1,
        state_delay_frames: int = 0,
        mode: str = "train",  # "train" or "test"
    ) -> None:
        if not datasets:
            raise ValueError("datasets must be a non-empty list")

        self.batch_size = int(batch_size)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.max_token_len = int(max_token_len)
        self.discrete_state_input = bool(discrete_state_input)
        self.apply_delta_transform = bool(apply_delta_transform)
        self.state_history_frames = int(state_history_frames)
        self.state_delay_frames = int(state_delay_frames)
        self.mode = str(mode)

        # Build multiple SimpleLeRobotLoader instances (source of truth).
        self._loaders: list[SimpleLeRobotLoader] = []
        for cfg in datasets:
            loader = SimpleLeRobotLoader(
                None,
                repo_id=cfg.repo_id,
                action_sequence_keys=list(cfg.action_sequence_keys),
                state_sequence_keys=list(cfg.state_sequence_keys),
                norm_stats=cfg.norm_stats,
                use_quantile_norm=cfg.use_quantile_norm,
                policy_name=cfg.policy_name,
                batch_size=self.batch_size,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                max_token_len=self.max_token_len,
                discrete_state_input=self.discrete_state_input,
                apply_delta_transform=self.apply_delta_transform,
                state_history_frames=self.state_history_frames,
                state_delay_frames=self.state_delay_frames,
                test_ep_num=cfg.test_ep_num,
                mode=self.mode,
            )
            self._loaders.append(loader)

        # Precompute index offsets for O(log N) __getitem__ lookup across loaders.
        self._offsets: list[int] = [0]
        total = 0
        for ld in self._loaders:
            total += len(ld)
            self._offsets.append(total)

        self.valid_ptr = None

    # Flattened length across all sub-loaders
    def __len__(self) -> int:  # type: ignore[override]
        return self._offsets[-1]

    # Random-access across concatenated datasets
    def __getitem__(self, idx: int) -> dict:
        try:
            if idx < 0:
                idx = len(self) + idx
            if idx < 0 or idx >= len(self):
                raise IndexError("index out of range")
            # Binary search in prefix sums; returns rightmost insertion point
            import bisect

            i = bisect.bisect_right(self._offsets, idx) - 1
            local_idx = idx - self._offsets[i]
            data = self._loaders[i][local_idx]
            self.valid_ptr = (i, local_idx)  # remember last valid loader index
            return data
        except Exception as e:
            msg = f"Error fetching index {idx}, repo_id {self._loaders[i].repo_id}, local_idx {local_idx}: {type(e).__name__}: {e}"
            logging.error(msg)
            i, local_idx = self.valid_ptr if self.valid_ptr is not None else (0, 0)
            return self._loaders[i][local_idx]  # fall back to the previous valid sample

    def __iter__(self) -> Iterator[dict]:
        # Now delta_timestamps handles all temporal queries automatically
        # Index-based round-robin iteration
        indices = [0 for _ in self._loaders]  # next-sample index per sub-loader
        active_idx = list(range(len(self._loaders)))
        buffer: list[dict] = []
        rr = 0

        while active_idx:
            i = active_idx[rr % len(active_idx)]
            loader = self._loaders[i]

            # Check if this loader is exhausted
            if indices[i] >= len(loader):
                # Drop loader i from the active list once its samples are exhausted.
                active_idx.pop(rr % len(active_idx))
                continue

            # Use loader's __getitem__ which handles delta_timestamps properly
            buffer.append(loader[indices[i]])
            indices[i] += 1
            rr += 1

            if len(buffer) == self.batch_size:
                yield _stack_tree(buffer)
                buffer = []

        if buffer:
            yield _stack_tree(buffer)
