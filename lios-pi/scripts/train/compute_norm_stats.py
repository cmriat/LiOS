import pi.training.config as _config
from pi.data import _repack_transform, _data_inputs, _make_bool_mask, _apply_delta_actions
import pi.training.instance_config as train_config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from utils import init_dist as _init_dist
import pi.shared.normalize as normalize
import torch
import tqdm
import numpy as np


class BaseLeRobotDataloader:
    def __init__(self, config: _config.TrainConfig):
        self.config = config
        self.policy_name = config.name
        self.repo_id = config.data.repo_id
        self.apply_delta_transform = config.data.apply_delta_transform
        if self.repo_id is None:
            raise ValueError("Data config must have a repo_id")
        metadata = lerobot_dataset.LeRobotDatasetMetadata(self.repo_id)
        # Use dataset fps to build per-key delta timestamps of length action_horizon.
        delta_timestamps = {
            key: [t / metadata.fps for t in range(config.model.action_horizon)]
            for key in config.data.action_sequence_keys
        }

        self.dataset = lerobot_dataset.LeRobotDataset(self.repo_id, delta_timestamps=delta_timestamps)

    def _transform(self, sample: dict) -> dict:
        step = _repack_transform(self.policy_name, sample)
        step = _data_inputs(step)

        # Apply delta action transform before normalization
        if self.apply_delta_transform:
            # (6, -1, 6, -1) means 6 joints + 1 gripper per arm
            # True for joint dimensions, False for gripper dimensions
            delta_action_mask = _make_bool_mask(6, -1, 6, -1)
            step = _apply_delta_actions(step, delta_action_mask)
        return step

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        transformed = self._transform(sample)
        return transformed

    def __len__(self) -> int:
        return len(self.dataset)


def compute_and_aggregate_stats(data_loader, world: int, rank: int, device) -> dict[str, normalize.NormStats]:
    """Compute normalization statistics and aggregate across all processes.

    Args:
        data_loader: PyTorch DataLoader to iterate over
        world: Total number of processes
        rank: Current process rank
        device: Device to use for computation (cuda or cpu)

    Returns:
        Dictionary of normalization statistics for each key
    """
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats(device=device) for key in keys}

    if rank == 0:
        iterator = tqdm.tqdm(data_loader, total=len(data_loader), desc="Computing stats")
    else:
        iterator = data_loader

    for batch in iterator:
        for key in keys:
            # Keep data on GPU if using GPU device
            value = batch[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            # Move to the target device (GPU or CPU)
            if device is not None:
                value = value.to(device)
            stats[key].update(value)

    # Synchronize statistics across all processes
    if world > 1:
        import torch.distributed as dist

        for key in keys:
            stat = stats[key]
            # If using GPU, tensors are already on device; otherwise convert from numpy
            if device is not None and device != "cpu":
                count_tensor = torch.tensor(stat._count, dtype=torch.float64, device=device)
                mean_tensor = stat._mean
                mean_of_squares_tensor = stat._mean_of_squares
                min_tensor = stat._min
                max_tensor = stat._max
            else:
                # CPU mode: convert from numpy to GPU for distributed ops
                count_tensor = torch.tensor(stat._count, dtype=torch.float64).cuda()
                mean_tensor = torch.from_numpy(stat._mean).cuda()
                mean_of_squares_tensor = torch.from_numpy(stat._mean_of_squares).cuda()
                min_tensor = torch.from_numpy(stat._min).cuda()
                max_tensor = torch.from_numpy(stat._max).cuda()

            # All-reduce count (sum across all processes)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            # Weighted average for mean and mean_of_squares
            # First, weight by local count
            local_count = stat._count
            mean_tensor *= local_count
            mean_of_squares_tensor *= local_count

            # Sum across processes
            dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mean_of_squares_tensor, op=dist.ReduceOp.SUM)

            # Divide by total count to get global average
            mean_tensor /= count_tensor
            mean_of_squares_tensor /= count_tensor

            # Synchronize min and max first to get global range
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)

            # Realign histograms to global bin edges before aggregation
            vector_length = len(stat._histograms)

            if device is not None and device != "cpu":
                # GPU mode: work with tensors
                global_bin_edges = [
                    torch.linspace(
                        min_tensor[i] - 1e-10, max_tensor[i] + 1e-10, stat._num_quantile_bins + 1, device=device
                    )
                    for i in range(vector_length)
                ]

                # Redistribute local histograms to global bin edges
                for i in range(vector_length):
                    old_edges = stat._bin_edges[i]
                    new_edges = global_bin_edges[i]

                    # Create a weighted redistribution of old bins to new bins
                    # Using old bin centers to redistribute counts
                    old_centers = (old_edges[:-1] + old_edges[1:]) / 2
                    new_hist = torch.zeros(stat._num_quantile_bins, device=device)

                    # Vectorized binning - make tensors contiguous to avoid warning
                    old_centers = old_centers.contiguous()
                    new_edges_left = new_edges[:-1].contiguous()
                    indices = torch.searchsorted(new_edges_left, old_centers, right=False)
                    indices = torch.clamp(indices, 0, len(new_edges) - 2)
                    new_hist.scatter_add_(0, indices, stat._histograms[i])

                    stat._histograms[i] = new_hist
                    stat._bin_edges[i] = new_edges

                # Aggregate histograms (now aligned)
                for i in range(len(stat._histograms)):
                    dist.all_reduce(stat._histograms[i], op=dist.ReduceOp.SUM)

                # Update stats with aggregated values
                stat._count = int(count_tensor.item())
                stat._mean = mean_tensor
                stat._mean_of_squares = mean_of_squares_tensor
                stat._min = min_tensor
                stat._max = max_tensor
            else:
                # CPU mode: convert to numpy
                global_min = min_tensor.cpu().numpy()
                global_max = max_tensor.cpu().numpy()

                global_bin_edges = [
                    np.linspace(global_min[i] - 1e-10, global_max[i] + 1e-10, stat._num_quantile_bins + 1)
                    for i in range(vector_length)
                ]

                # Redistribute local histograms to global bin edges
                for i in range(vector_length):
                    old_edges = stat._bin_edges[i]
                    new_edges = global_bin_edges[i]

                    # Create a weighted redistribution of old bins to new bins
                    # Using old bin centers to redistribute counts
                    old_centers = (old_edges[:-1] + old_edges[1:]) / 2
                    new_hist = np.zeros(stat._num_quantile_bins)

                    for center, count in zip(old_centers, stat._histograms[i]):
                        # Find which new bin this old bin center falls into
                        # Use side='right' to match np.histogram behavior
                        bin_idx = np.searchsorted(new_edges, center, side="right") - 1
                        bin_idx = np.clip(bin_idx, 0, len(new_edges) - 2)
                        new_hist[bin_idx] += count

                    stat._histograms[i] = new_hist
                    stat._bin_edges[i] = new_edges

                # Aggregate histograms (now aligned)
                for i in range(len(stat._histograms)):
                    hist_tensor = torch.from_numpy(stat._histograms[i]).cuda()
                    dist.all_reduce(hist_tensor, op=dist.ReduceOp.SUM)
                    stat._histograms[i] = hist_tensor.cpu().numpy()

                # Update stats with aggregated values
                stat._count = int(count_tensor.item())
                stat._mean = mean_tensor.cpu().numpy()
                stat._mean_of_squares = mean_of_squares_tensor.cpu().numpy()
                stat._min = global_min
                stat._max = global_max

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    return norm_stats


def main():
    config = train_config.cli()
    rank, world, _local_rank, device = _init_dist()
    batch_size = int(config.batch_size)
    if batch_size % world != 0:
        raise ValueError(f"batch_size {batch_size} must be divisible by world_size {world}")
    local_batch_size = batch_size // world
    dataset = BaseLeRobotDataloader(config)

    g = torch.Generator()
    g.manual_seed(config.seed)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=False, seed=config.seed
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=False,  # Must be False when using sampler
        num_workers=12,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=g,
    )

    # Compute and aggregate statistics using GPU
    norm_stats = compute_and_aggregate_stats(data_loader, world, rank, device)

    # Only rank 0 saves the stats
    if rank == 0:
        output_path = config.data.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)

    # Clean up distributed process group
    if world > 1:
        import torch.distributed as dist

        # Specify device_id to avoid warning
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index if device.index is not None else 0])
        else:
            dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

    # python scripts/train/compute_norm_stats.py pi05_robotwin --data.repo_id /path/to/lerobot_dataset
