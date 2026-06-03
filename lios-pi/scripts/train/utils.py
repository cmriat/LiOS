"""Utility functions for FSDP training."""

from __future__ import annotations

import os
import glob
import math
import shutil
import logging
import pathlib
import dataclasses
from typing import Tuple

import numpy as np
import torch
import wandb
import torch.distributed as dist
import torch.distributed.checkpoint
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor

import pi.training.config as _config  # noqa: E402

logger = logging.getLogger()


def init_dist(*, backend: str | None = None) -> Tuple[int, int, int, torch.device]:
    """Initialize torch.distributed (env://) with optional backend override."""
    requested_backend = backend.lower() if backend else None
    is_initialized = dist.is_initialized()

    if not is_initialized:
        backend_in_use = requested_backend or "nccl"
        dist.init_process_group(backend=backend_in_use, init_method="env://")
        logging.info(f"Initialized torch.distributed with backend={backend_in_use}")
    else:
        backend_in_use = str(dist.get_backend()).lower()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not is_initialized:
        if backend_in_use == "nccl":
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            else:
                raise RuntimeError("This script requires CUDA when using the NCCL backend.")
        else:
            logger.info("Skipping CUDA device setup because non-NCCL backend is in use.")

    if backend_in_use == "nccl":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    logger.info(f"{rank=},{device=}")
    return rank, world_size, local_rank, device


def validate_shared_fields(configs: list[_config.TrainConfig]) -> None:
    if not configs:
        raise ValueError("configs must be non-empty")
    fields = (
        ("action_horizon", lambda c: c.model.action_horizon),
        ("action_dim", lambda c: c.model.action_dim),
        ("max_token_len", lambda c: c.model.max_token_len),
        ("discrete_state_input", lambda c: getattr(c.model, "discrete_state_input", False)),
        ("batch_size", lambda c: c.batch_size),
        ("pytorch_training_precision", lambda c: c.pytorch_training_precision),
    )
    for name, fn in fields:
        v0 = fn(configs[0])
        for c in configs[1:]:
            if fn(c) != v0:
                raise ValueError(f"All configs must share {name}. Got {v0} vs {fn(c)}")


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


# ------------------------- logging utilities -------------------------
def init_logging():
    """Initialize logging with custom formatter."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def log_memory_usage(device, step, phase="unknown"):
    """Log GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return
    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
    mem_resv = torch.cuda.memory_reserved(device) / 1e9
    mem_free = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1e9
    stats = torch.cuda.memory_stats(device)
    peak_alloc = stats.get("allocated_bytes.all.peak", 0) / 1e9
    peak_resv = stats.get("reserved_bytes.all.peak", 0) / 1e9
    ddp_info = f" | dist: rank={dist.get_rank()}, world={dist.get_world_size()}" if dist.is_initialized() else ""
    logging.info(
        f"Step {step} ({phase}): GPU mem alloc={mem_alloc:.2f}GB, resv={mem_resv:.2f}GB, free={mem_free:.2f}GB, "
        f"peak_alloc={peak_alloc:.2f}GB, peak_resv={peak_resv:.2f}GB{ddp_info}"
    )


def lr_schedule(step, total_steps, config):
    lr_config = config.lr_schedule
    if step < lr_config.warmup_steps:
        init_lr = lr_config.peak_lr / (lr_config.warmup_steps + 1)
        return init_lr + (lr_config.peak_lr - init_lr) * step / lr_config.warmup_steps
    decay_steps = max(lr_config.decay_steps, total_steps + 20_000)
    progress = min(1.0, (step - lr_config.warmup_steps) / max(1, decay_steps - lr_config.warmup_steps))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_config.decay_lr + (lr_config.peak_lr - lr_config.decay_lr) * cos


def fsdp_wrap(model: torch.nn.Module, all_fp32=False) -> torch.nn.Module:
    def _select_mp_policy_bf16() -> MixedPrecisionPolicy | None:
        # Prefer bf16 on Ampere+; otherwise fp16. Inputs will be cast at forward.
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if not all_fp32 else torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=True,
        )

    mp_policy = _select_mp_policy_bf16()

    with torch.no_grad():
        model.to(torch.float32)
    if dist.is_initialized():
        with torch.no_grad():  # critical path: avoid autograd tracking
            for t in model.state_dict().values():
                if torch.is_tensor(t) and t.numel() > 0:
                    dist.broadcast(t, src=0)
    # fully_shard mutates module in-place; create optimizer AFTER this.
    fully_shard(module=model, mp_policy=mp_policy, reshard_after_forward=False)
    return model


# ------------------------- gradient clipping -------------------------
@torch.no_grad()
def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh=None,
) -> torch.Tensor:
    """Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)

    # Group gradients and parameters by device mesh to handle mixed meshes (e.g., EP + non-EP layers)
    mesh_to_grads = {}
    mesh_to_params = {}
    for p in parameters:
        if p.grad is not None:
            if isinstance(p.grad, DTensor):
                mesh_key = str(p.grad.device_mesh)
            else:
                # Regular tensors
                mesh_key = "local"

            if mesh_key not in mesh_to_grads:
                mesh_to_grads[mesh_key] = []
                mesh_to_params[mesh_key] = []
            mesh_to_grads[mesh_key].append(p.grad)
            mesh_to_params[mesh_key].append(p)

    # Compute total norm for each mesh group separately, then combine
    group_norms = []
    for grad_group in mesh_to_grads.values():
        group_norm = torch.nn.utils.get_total_norm(grad_group, norm_type, error_if_nonfinite, foreach)
        if isinstance(group_norm, DTensor):
            group_norm = group_norm.full_tensor()
        group_norms.append(group_norm)

    # Combine norms from different meshes
    if math.isinf(norm_type):
        total_norm = torch.stack(group_norms).max()
    else:
        total_norm_p = sum(norm**norm_type for norm in group_norms)
        total_norm = total_norm_p ** (1.0 / norm_type)

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    # Apply gradient clipping to each mesh group separately using the global total_norm
    for params_in_group in mesh_to_params.values():
        torch.nn.utils.clip_grads_with_norm_(params_in_group, max_norm, total_norm, foreach)

    return total_norm


def build_configs_from_parent_dir(
    parent_dir: str | pathlib.Path, template: _config.TrainConfig
) -> list[_config.TrainConfig]:
    """Build a list of TrainConfig by globbing first-level subdirectories.

    Only repo_id differs across configs; all other fields follow the template.
    """
    # NOTE(critical path): use glob for first-level dirs and keep absolute paths for repo_id
    pattern = os.path.join(str(parent_dir), "*/")
    candidates = sorted(glob.glob(pattern))
    subdirs = [p for p in candidates if os.path.isdir(p)]
    if not subdirs:
        raise FileNotFoundError(f"No first-level subdirectories found under: {parent_dir}")

    cfgs: list[_config.TrainConfig] = []
    for d in subdirs:
        abs_d = os.path.abspath(d)
        # Replace only the nested DataConfigFactory.repo_id while keeping other fields intact
        new_data = dataclasses.replace(template.data, repo_id=abs_d)
        new_cfg = dataclasses.replace(template, data=new_data)
        cfgs.append(new_cfg)
        logging.info(f"Built config for repo_id: {abs_d}")
    return cfgs


def _tree_map_to_device(item, target_device):
    if isinstance(item, dict):
        return {k: _tree_map_to_device(v, target_device) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        converted = [_tree_map_to_device(v, target_device) for v in item]
        return type(item)(converted)
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item).to(target_device)
    if hasattr(item, "to"):
        return item.to(target_device)
    return item


def run_test_evaluation(
    model: torch.nn.Module,
    test_dataloader,
    device: torch.device,
    rank: int,
    is_main: bool,
    epoch: int | None = None,
    global_step: int | None = None,
) -> float | None:
    if (epoch is None and global_step is None) or (epoch is not None and global_step is not None):
        raise ValueError("Exactly one of 'epoch' or 'global_step' must be provided, not both or neither")

    if is_main:
        if epoch is not None:
            logging.info(f"Starting test evaluation at epoch {epoch}")
        else:
            logging.info(f"Starting test evaluation at step {global_step}")
    model_was_training = model.training
    model.eval()
    test_losses: list[float] = []
    with torch.no_grad():
        for test_batch in test_dataloader:
            test_batch = _tree_map_to_device(test_batch, device)
            test_observation_dict = {k: v for k, v in test_batch.items() if k != "actions"}
            test_actions = test_batch["actions"].to(torch.float32)
            test_loss_tensor = model(test_observation_dict, test_actions)
            if isinstance(test_loss_tensor, (list, tuple)):
                test_loss_tensor = torch.stack(list(test_loss_tensor))
            elif not isinstance(test_loss_tensor, torch.Tensor):
                test_loss_tensor = torch.tensor(
                    test_loss_tensor,
                    dtype=torch.float32,
                    device=device,
                )
            test_losses.append(test_loss_tensor.mean().item())
    if test_losses:
        loss_sum = torch.tensor(
            [sum(test_losses), len(test_losses)],
            dtype=torch.float64,
            device=device,
        )
    else:
        loss_sum = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    total_loss, total_batches = loss_sum.tolist()
    test_loss_value = None
    if total_batches > 0:
        test_loss_value = total_loss / total_batches
        if rank == 0:
            if epoch is not None:
                logging.info(f"Finished test evaluation at epoch {epoch} with loss {test_loss_value:.6f}")
            else:
                logging.info(f"Finished test evaluation at step {global_step} with loss {test_loss_value:.6f}")
    if model_was_training:
        model.train()
    return test_loss_value


def fsdp_save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: _config.TrainConfig,
    is_main: bool,
    epoch: int = None,
    step: int = None,
    ema_model=None,
) -> None:
    """Save FSDP model checkpoint.

    Args:
        model: The FSDP-wrapped model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number (used as checkpoint directory name), if None, use step instead
        step: Current step number (used as checkpoint directory name), if None, use epoch instead
        config: Training configuration
        is_main: Whether this is the main process
        ema_model: Optional EMA model to save

    Note: This function no longer checks save_interval internally.
    The caller should decide when to save (e.g., based on epoch).
    """
    # Ensure exactly one of epoch or step is provided
    if (epoch is None and step is None) or (epoch is not None and step is not None):
        raise ValueError("Exactly one of 'epoch' or 'step' must be provided, not both or neither")

    # Use temporary directory for atomic save
    if epoch is not None:
        logging.info(f"Saving FSDP model checkpoint at epoch {epoch}...")
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_epoch{epoch}"
        final_ckpt_dir = config.checkpoint_dir / f"epoch{epoch}"
    else:
        logging.info(f"Saving FSDP model checkpoint at step {step}...")
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_step{step}"
        final_ckpt_dir = config.checkpoint_dir / f"step{step}"

    # Clean up any existing temp directory first
    if is_main:
        shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)
    torch.distributed.barrier()

    # Save to temporary directory
    torch.distributed.checkpoint.save(model.state_dict(), checkpoint_id=tmp_ckpt_dir)

    # Save EMA parameters using distributed checkpoint (sharded across ranks)
    if ema_model is not None:
        ema_tmp_dir = tmp_ckpt_dir / "ema"
        torch.distributed.checkpoint.save(ema_model.shadow, checkpoint_id=ema_tmp_dir)

    # Save RNG states (only main process saves, all processes use same RNG states per rank)
    if is_main:
        rng_state = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()

        rng_path = tmp_ckpt_dir / "rng_state.pth"
        torch.save(rng_state, rng_path)
        logging.info(f"Saved RNG states to {rng_path}")

    # Atomically rename temp directory to final (only main process does the rename)
    torch.distributed.barrier()
    if is_main:
        shutil.rmtree(final_ckpt_dir, ignore_errors=True)
        tmp_ckpt_dir.rename(final_ckpt_dir)
    torch.distributed.barrier()

    if epoch is not None:
        logging.info(f"Saved FSDP model checkpoint at epoch {epoch} -> {final_ckpt_dir}")
    else:
        logging.info(f"Saved FSDP model checkpoint at step {step} -> {final_ckpt_dir}")

    # Save optimizer checkpoint (also use atomic rename)
    tmp_optim_dir = config.checkpoint_dir / "tmp_last_optim"
    optim_ckpt_dir = config.checkpoint_dir / "last_optim"

    if is_main:
        shutil.rmtree(tmp_optim_dir, ignore_errors=True)
    torch.distributed.barrier()
    torch.distributed.checkpoint.save(optimizer.state_dict(), checkpoint_id=tmp_optim_dir)

    torch.distributed.barrier()
    if is_main:
        shutil.rmtree(optim_ckpt_dir, ignore_errors=True)
        tmp_optim_dir.rename(optim_ckpt_dir)
    torch.distributed.barrier()

    if epoch is not None:
        logging.info(f"Saved FSDP optimizer checkpoint at epoch {epoch} -> {optim_ckpt_dir}")
    else:
        logging.info(f"Saved FSDP optimizer checkpoint at step {step} -> {optim_ckpt_dir}")


def resume_from_fsdp_model_checkpoint(model, optimizer, checkpoint_dir: pathlib.Path, ema_model=None) -> int:
    """Load FSDP model and optimizer checkpoints from directory.

    Only loads from step-based checkpoints (e.g., step0, step10, step100).
    Epoch-based checkpoints are ignored for resume.

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        checkpoint_dir: Directory containing checkpoints (must exist)
        ema_model: Optional EMA model

    Returns:
        int: The next global step to start training from (last_saved_step + 1).
             For example, if step10 was loaded, returns 11.

    Raises:
        FileNotFoundError: If no valid step-based checkpoint is found.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    last_global_step = -1
    last_ckpt_name = None

    for dir in checkpoint_dir.glob("*"):
        if not dir.is_dir():
            continue
        # Skip temporary directories, optimizer dir, and epoch-based checkpoints
        if dir.name.startswith("tmp_") or dir.name == "last_optim" or dir.name.startswith("epoch"):
            continue
        try:
            # Only accept step-based checkpoints (e.g., "step123")
            if dir.name.startswith("step"):
                step = int(dir.name[4:])
            else:
                # Legacy: pure number defaults to step
                step = int(dir.name)

            if step > last_global_step:
                last_global_step = step
                last_ckpt_name = dir.name
        except ValueError:
            continue

    # If no step checkpoint found, raise error
    if last_ckpt_name is None or last_global_step < 0:
        raise FileNotFoundError(
            f"No valid step-based checkpoint found in {checkpoint_dir}. "
            "If you want to start new training, use --overwrite instead of --resume."
        )

    last_model_ckpt_dir = checkpoint_dir / last_ckpt_name

    # Load model state dict
    torch.distributed.checkpoint.load(model.state_dict(), checkpoint_id=last_model_ckpt_dir)
    logging.info(f"Loaded FSDP model checkpoint from {last_model_ckpt_dir}")

    # Load EMA parameters if provided
    if ema_model is not None:
        ema_ckpt_dir = last_model_ckpt_dir / "ema"
        if ema_ckpt_dir.exists():
            torch.distributed.checkpoint.load(ema_model.shadow, checkpoint_id=ema_ckpt_dir)
            logging.info(f"Loaded FSDP EMA model checkpoint from {ema_ckpt_dir}")
        else:
            raise FileNotFoundError(
                f"EMA checkpoint directory not found: {ema_ckpt_dir}. "
                "The checkpoint was saved without EMA but you are trying to resume with EMA enabled. "
                "Either disable EMA (set ema_decay=None) or use a checkpoint that was saved with EMA."
            )

    # Load optimizer state dict
    optim_ckpt_dir = checkpoint_dir / "last_optim"
    if not optim_ckpt_dir.exists():
        raise FileNotFoundError(f"Optimizer checkpoint directory {optim_ckpt_dir} does not exist.")
    torch.distributed.checkpoint.load(optimizer.state_dict(), checkpoint_id=optim_ckpt_dir)
    logging.info(f"Loaded FSDP optimizer checkpoint from {optim_ckpt_dir}")

    # Load RNG states
    rng_path = last_model_ckpt_dir / "rng_state.pth"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(rng_state["torch_rng_state"])
        np.random.set_state(rng_state["numpy_rng_state"])
        if torch.cuda.is_available() and "torch_cuda_rng_state" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda_rng_state"])
        logging.info(f"Loaded RNG states from {rng_path}")
    else:
        logging.warning(
            f"RNG state file not found: {rng_path}. RNG states will not be restored (checkpoint may be from older version)."
        )

    # Return the next step to train (the loaded step has already been trained)
    return last_global_step + 1
