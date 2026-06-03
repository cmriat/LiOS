# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch profiler and CUDA memory-snapshot context managers for the FSDP trainer.

Self-contained: depends only on the standard library and torch, with local
filesystem paths. Trace/snapshot folders are created on demand.
"""

import time
import pickle
import logging
import contextlib
import dataclasses
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Number of warmup steps before the active step in each profiling cycle.
WARMUP = 1

# How many memory allocation/free ops to record in memory snapshots.
MEMORY_SNAPSHOT_MAX_ENTRIES = 50000


@dataclasses.dataclass
class ProfilingConfig:
    enable_profiling: bool = False
    save_traces_folder: str = "./traces"
    profile_freq: int = 100
    enable_memory_snapshot: bool = False
    save_memory_snapshot_folder: str = "./memory_snapshot"


def _current_rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


@contextlib.contextmanager
def maybe_enable_profiling(config: ProfilingConfig, *, global_step: int = 0):
    if not config.enable_profiling:
        yield None
        return

    trace_dir = Path(config.save_traces_folder)
    rank = _current_rank()

    def trace_handler(prof):
        curr_trace_dir = trace_dir / f"iteration_{prof.step_num}"
        curr_trace_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dumping profiler traces at step %s", prof.step_num)
        begin = time.monotonic()
        prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.pt.trace.json")
        logger.info("Finished dumping profiler traces in %.2f seconds", time.monotonic() - begin)

    logger.info("Profiling active. Traces will be saved at %s", trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    warmup, active = WARMUP, 1
    wait = config.profile_freq - (active + warmup)
    if wait < 0:
        raise ValueError("profile_freq must be greater than or equal to warmup + active")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif torch.xpu.is_available():
        activities.append(torch.profiler.ProfilerActivity.XPU)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=trace_handler,
        with_stack=False,
        record_shapes=False,
    ) as torch_profiler:
        torch_profiler.step_num = global_step
        yield torch_profiler


@contextlib.contextmanager
def maybe_enable_memory_snapshot(config: ProfilingConfig, *, global_step: int = 0):
    if not config.enable_memory_snapshot:
        yield None
        return

    snapshot_dir = Path(config.save_memory_snapshot_folder)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    rank = _current_rank()

    class MemoryProfiler:
        def __init__(self, step_num: int, freq: int):
            torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
            # When resuming training we start from the last step.
            self.step_num = step_num
            self.freq = freq

        def step(self, exit_ctx: bool = False):
            self.step_num += 1
            if not exit_ctx and self.step_num % self.freq != 0:
                return
            if not exit_ctx:
                curr_step = self.step_num
                dir_name = f"iteration_{curr_step}"
            else:
                # Dump as iteration_<n>_exit when OOM aborts the step.
                curr_step = self.step_num - 1
                dir_name = f"iteration_{curr_step}_exit"
            curr_snapshot_dir = snapshot_dir / dir_name
            curr_snapshot_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Dumping memory snapshot at step %s", curr_step)
            begin = time.monotonic()
            with open(f"{curr_snapshot_dir}/rank{rank}_memory_snapshot.pickle", "wb") as output:
                pickle.dump(torch.cuda.memory._snapshot(), output)
            logger.info("Finished dumping memory snapshot in %.2f seconds", time.monotonic() - begin)

    logger.info("Memory profiler active. Snapshot will be saved at %s", snapshot_dir)
    profiler = MemoryProfiler(global_step, config.profile_freq)
    try:
        yield profiler
    except torch.OutOfMemoryError:
        profiler.step(exit_ctx=True)
