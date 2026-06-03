"""FSDP training entrypoint. See docs/training.md for usage."""

from __future__ import annotations

import os
import time
import shutil
import logging
import platform
from typing import List

import tqdm
import numpy as np
import torch
import wandb
import safetensors.torch
import torch.distributed as dist
from utils import (
    fsdp_wrap,
    init_dist,
    init_wandb,
    lr_schedule,
    init_logging,
    clip_grad_norm_,
    log_memory_usage,
    run_test_evaluation,
    fsdp_save_model_checkpoint,
    build_configs_from_parent_dir,
    resume_from_fsdp_model_checkpoint,
)
from profiling import (
    ProfilingConfig,
    maybe_enable_profiling,
    maybe_enable_memory_snapshot,
)
from data_loader import create_distributed_dataloader

import pi.training.config as _config  # noqa: E402

# Import training configs
import pi.training.instance_config as train_config
import pi.models_pytorch.pi0_pytorch  # noqa: E402
from pi.ema_model import EMAModel


def train_loop(
    configs: List[_config.TrainConfig],
):
    # ========== Initialization ==========
    # Use first config as base for shared settings
    base_config = configs[0]

    # Initialize distributed training and set random seeds
    rank, world, local_rank, device = init_dist()
    torch.manual_seed(base_config.seed + local_rank)
    np.random.seed(base_config.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_config.seed + local_rank)

    is_main = (not dist.is_initialized()) or rank == 0

    # ========== Data Loading Setup ==========
    loaders = create_distributed_dataloader(
        configs,
        shuffle=base_config.shuffle,
        seed=base_config.seed + rank,
        gpu_rank=world,
    )
    train_dataloader = loaders["train_loader"]
    test_dataloader = loaders.get("test_loader")

    # ========== Model Creation ==========
    # Model config dtype -> pytorch training precision
    model_cfg = base_config.model
    object.__setattr__(model_cfg, "dtype", base_config.pytorch_training_precision)

    # Create model on device
    with torch.device(device):
        raw_model = pi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)
        if base_config.frozen:
            for param in raw_model.paligemma_with_expert.paligemma.parameters():
                param.requires_grad = False
            logging.info(f"Created model {raw_model.__class__.__name__} on device {device}, with frozen weights")

    # ========== Checkpoint Directory and Pretrained Weights Loading ==========
    # Prepare checkpoint directory
    if base_config.overwrite and is_main:
        if base_config.checkpoint_dir.exists():
            shutil.rmtree(base_config.checkpoint_dir)
        base_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained weights if provided (before FSDP wrapping)
    if base_config.pytorch_weight_path and (not base_config.resume):
        if is_main:
            model_path = os.path.join(base_config.pytorch_weight_path, "model.safetensors")
            missing, unexpected = safetensors.torch.load_model(raw_model, model_path, strict=False)
            if missing:
                logging.warning(f"Missing keys when loading model: {missing}")
            if unexpected:
                logging.warning(f"Unexpected keys when loading model: {unexpected}")
            logging.info(f"Loaded non-FSDP weights from: {base_config.pytorch_weight_path}")

    raw_model = torch.compile(raw_model, fullgraph=True)
    # Wrap with FSDP (must occur before optimizer creation)
    model = fsdp_wrap(raw_model, base_config.all_fp32)

    # ========== EMA Setup ==========
    if base_config.ema_decay is not None and base_config.ema_decay > 0:
        ema_model = EMAModel(model, decay=base_config.ema_decay)
        if is_main:
            logging.info(f"Initialized EMA with decay={base_config.ema_decay}")
    else:
        ema_model = None

    # ========== Optimizer ==========
    optim_params = (p for p in model.parameters() if p.requires_grad)
    optim = torch.optim.AdamW(
        optim_params,
        lr=base_config.lr_schedule.peak_lr,
        betas=(base_config.optimizer.b1, base_config.optimizer.b2),
        eps=base_config.optimizer.eps,
        weight_decay=base_config.optimizer.weight_decay,
    )

    # ========== Resume from FSDP checkpoint if needed ==========
    global_step = 0
    if base_config.resume:
        if not base_config.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir {base_config.checkpoint_dir} does not exist for resume")
        next_step = resume_from_fsdp_model_checkpoint(model, optim, base_config.checkpoint_dir, ema_model)
        if is_main:
            logging.info(f"Resumed training, starting from step {next_step} (loaded checkpoint step{next_step - 1})")
        global_step = next_step

    # ========== WandB Setup ==========
    if is_main:
        init_wandb(base_config, resuming=base_config.resume, enabled=base_config.wandb_enabled)
        # Define custom metrics with different x-axes
        if base_config.wandb_enabled:
            wandb.define_metric("train_global_step")
            wandb.define_metric("test_global_step")
            wandb.define_metric("train_epoch")
            wandb.define_metric("test_epoch")
            # Metrics that use train_global_step as x-axis
            wandb.define_metric("train/loss_vs_step", step_metric="train_global_step")
            wandb.define_metric("train/learning_rate", step_metric="train_global_step")
            wandb.define_metric("train/grad_norm", step_metric="train_global_step")
            wandb.define_metric("perf/step_total_s", step_metric="train_global_step")
            wandb.define_metric("perf/data_iter_s", step_metric="train_global_step")
            wandb.define_metric("perf/model_fwd_bwd_s", step_metric="train_global_step")
            wandb.define_metric("perf/optimizer_s", step_metric="train_global_step")
            # Metrics that use test_global_step as x-axis
            wandb.define_metric("test/loss_vs_step", step_metric="test_global_step")
            # Metrics that use train_epoch as x-axis
            wandb.define_metric("train/loss_vs_epoch", step_metric="train_epoch")
            # Metrics that use test_epoch as x-axis
            wandb.define_metric("test/loss_vs_epoch", step_metric="test_epoch")

    # ========== Training Loop ==========
    # Enable memory optimizations for large-scale distributed runs
    if world >= 8 and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_wrap")

    # Initialize training loop variables
    num_epochs = base_config.num_epochs
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch

    # Calculate starting epoch from global_step (for resume)
    start_epoch = global_step // steps_per_epoch
    steps_in_current_epoch = global_step % steps_per_epoch
    profiling_config = ProfilingConfig(
        enable_profiling=False,
        save_traces_folder=f"./traces_{base_config.batch_size}",
        profile_freq=10,
        enable_memory_snapshot=False,
        save_memory_snapshot_folder=f"./memory_snapshot_{base_config.batch_size}",
    )

    infos: list[dict] = []
    timing_enabled = is_main and base_config.wandb_enabled
    timing_prev_step_end_s = time.perf_counter() if timing_enabled else 0.0
    timing_use_cuda_events = timing_enabled and torch.cuda.is_available()
    if timing_use_cuda_events:
        timing_model_start = torch.cuda.Event(enable_timing=True)
        timing_model_end = torch.cuda.Event(enable_timing=True)
        timing_optim_start = torch.cuda.Event(enable_timing=True)
        timing_optim_end = torch.cuda.Event(enable_timing=True)

    if is_main:
        logging.info(
            f"Host={platform.node()} world={world} local_rank={local_rank} batch-size={base_config.batch_size} "
            f"epochs={num_epochs} steps_per_epoch={steps_per_epoch}"
        )
        if base_config.resume:
            logging.info(f"Resuming from epoch {start_epoch} (0-indexed), batch {steps_in_current_epoch} within epoch")

    model.train()

    # Check if training is already complete
    if start_epoch >= num_epochs:
        if is_main:
            logging.info(
                f"Training already completed. global_step={global_step}, start_epoch={start_epoch}, "
                f"num_epochs={num_epochs}"
            )
            logging.info("No training will be performed.")
        return
    pbar = tqdm.tqdm(total=total_steps, initial=global_step, desc="FSDP-Train", disable=not is_main)
    with (
        maybe_enable_profiling(profiling_config, global_step=global_step) as torch_profiler,
        maybe_enable_memory_snapshot(profiling_config, global_step=global_step) as memory_profiler,
    ):
        for epoch in range(start_epoch, num_epochs):
            if is_main:
                if epoch == start_epoch and steps_in_current_epoch > 0:
                    logging.info(
                        f"Resuming epoch {epoch} (0-indexed) from batch {steps_in_current_epoch}/{steps_per_epoch}"
                    )
                else:
                    logging.info(f"Starting epoch {epoch} (0-indexed, {epoch}/{num_epochs - 1})")

            # Set epoch for distributed sampler to ensure proper shuffling
            if isinstance(train_dataloader.loader, torch.utils.data.DataLoader):
                train_dataloader.loader.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(train_dataloader):
                if epoch == start_epoch and batch_idx < steps_in_current_epoch:
                    if timing_enabled:
                        timing_prev_step_end_s = time.perf_counter()
                    continue
                observation = {k: v for k, v in batch.items() if k != "actions"}
                actions = batch["actions"].to(torch.float32, non_blocking=True)
                if timing_enabled:
                    timing_data_iter_s = time.perf_counter() - timing_prev_step_end_s

                # Update learning rate based on schedule
                for pg in optim.param_groups:
                    pg["lr"] = lr_schedule(global_step, total_steps, base_config)

                # Forward pass and compute loss
                if timing_use_cuda_events:
                    timing_model_start.record()
                elif timing_enabled:
                    timing_model_start_s = time.perf_counter()
                losses = model(observation, actions)
                if isinstance(losses, (list, tuple)):
                    losses = torch.stack(list(losses))
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, dtype=torch.float32, device=device)

                loss = losses.mean()
                loss.backward()

                total_grad_norm = clip_grad_norm_(model.parameters(), max_norm=base_config.optimizer.clip_gradient_norm)
                if timing_use_cuda_events:
                    timing_model_end.record()
                elif timing_enabled:
                    timing_model_fwd_bwd_s = time.perf_counter() - timing_model_start_s

                if timing_use_cuda_events:
                    timing_optim_start.record()
                elif timing_enabled:
                    timing_optim_start_s = time.perf_counter()
                optim.step()
                if timing_use_cuda_events:
                    timing_optim_end.record()
                elif timing_enabled:
                    timing_optimizer_s = time.perf_counter() - timing_optim_start_s
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()
                global_step += 1

                if ema_model is not None:
                    ema_model.update(model)

                optim.zero_grad(set_to_none=True)

                # Update progress bar
                if is_main:
                    loss_value = loss.item()
                    grad_norm_value = total_grad_norm.item()
                    if timing_enabled:
                        step_end_s = time.perf_counter()
                        timing_step_total_s = step_end_s - timing_prev_step_end_s
                        timing_prev_step_end_s = step_end_s
                        if timing_use_cuda_events:
                            timing_model_fwd_bwd_s = timing_model_start.elapsed_time(timing_model_end) / 1000.0
                            timing_optimizer_s = timing_optim_start.elapsed_time(timing_optim_end) / 1000.0
                    pbar.n = global_step
                    pbar.set_postfix(
                        {
                            "loss": f"{loss_value:.4f}",
                            "lr": f"{optim.param_groups[0]['lr']:.2e}",
                        }
                    )
                    pbar.refresh()

                # Collect training statistics
                if is_main:
                    train_stats = {
                        "train_loss": loss_value,
                        "train_learning_rate": optim.param_groups[0]["lr"],
                        "train_total_grad_norm": grad_norm_value,
                    }
                    if timing_enabled:
                        train_stats.update(
                            {
                                "time_step_total_s": timing_step_total_s,
                                "time_data_iter_s": timing_data_iter_s,
                                "time_model_fwd_bwd_s": timing_model_fwd_bwd_s,
                                "time_optimizer_s": timing_optimizer_s,
                            }
                        )
                    infos.append({**train_stats})

                # Log metrics to wandb at specified intervals
                if is_main and len(infos) >= base_config.log_interval:
                    avg_loss = sum(i["train_loss"] for i in infos) / max(1, len(infos))
                    avg_lr = sum(i["train_learning_rate"] for i in infos) / max(1, len(infos))
                    avg_grad_norm = sum(i["train_total_grad_norm"] for i in infos) / max(1, len(infos))
                    if base_config.wandb_enabled:
                        payload = {
                            "train_global_step": global_step,
                            "train/loss_vs_step": avg_loss,
                            "train/learning_rate": avg_lr,
                            "train/grad_norm": avg_grad_norm,
                        }
                        if timing_enabled:
                            denom = max(1, len(infos))
                            payload.update(
                                {
                                    "perf/step_total_s": sum(i["time_step_total_s"] for i in infos) / denom,
                                    "perf/data_iter_s": sum(i["time_data_iter_s"] for i in infos) / denom,
                                    "perf/model_fwd_bwd_s": sum(i["time_model_fwd_bwd_s"] for i in infos) / denom,
                                    "perf/optimizer_s": sum(i["time_optimizer_s"] for i in infos) / denom,
                                }
                            )
                        wandb.log(payload, step=global_step)
                    infos.clear()

                # Save checkpoint at step intervals
                should_save = global_step % base_config.save_step_interval == 0 or global_step == total_steps - 1
                if should_save:
                    fsdp_save_model_checkpoint(
                        model, optim, base_config, is_main, epoch=None, step=global_step, ema_model=ema_model
                    )
                    if is_main:
                        logging.info(f"Saved checkpoint at step {global_step}")

                # Run test evaluation at step intervals
                if base_config.test_step_interval is not None and test_dataloader is not None:
                    should_test_step = global_step % base_config.test_step_interval == 0
                    if should_test_step:
                        test_loss_value = run_test_evaluation(
                            model, test_dataloader, device, rank, is_main, None, global_step
                        )
                        if test_loss_value is not None and is_main and base_config.wandb_enabled:
                            wandb.log(
                                {"test_global_step": global_step, "test/loss_vs_step": test_loss_value},
                                step=global_step,
                            )

            # Log epoch-level metrics
            if is_main and base_config.wandb_enabled:
                wandb.log(
                    {
                        "train_epoch": epoch,
                        "train/loss_vs_epoch": loss.item(),
                    },
                    step=global_step - 1,
                )

            # # Save checkpoint at epoch intervals
            if base_config.save_epoch_interval is not None:
                should_save_epoch = epoch % base_config.save_epoch_interval == 0 or epoch == num_epochs - 1
                if should_save_epoch:
                    fsdp_save_model_checkpoint(
                        model, optim, base_config, is_main, epoch=epoch, step=None, ema_model=ema_model
                    )
                    if is_main:
                        logging.info(f"Saved checkpoint at epoch {epoch}")

        # # Run test evaluation at end of each epoch
        if test_dataloader is not None:
            test_loss_value = run_test_evaluation(model, test_dataloader, device, rank, is_main, epoch, None)
            if test_loss_value is not None and is_main and base_config.wandb_enabled:
                wandb.log({"test_epoch": epoch, "test/loss_vs_epoch": test_loss_value}, step=global_step - 1)

    if is_main and base_config.wandb_enabled:
        wandb.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> int:
    init_logging()
    # Parse base config from CLI
    base_config = train_config.cli()
    logging.info(f"Training config: {base_config}")

    if base_config.parent_data_dir:
        cfgs = build_configs_from_parent_dir(base_config.parent_data_dir, base_config)
    else:
        cfgs = [base_config]
    if base_config.all_fp32:
        torch.backends.cuda.matmul.allow_tf32 = True  # deprecated in future torch versions
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train_loop(cfgs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
