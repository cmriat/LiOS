# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Running statistics (mean / std / quantiles) for dataset normalization."""

import json
import pathlib

import numpy as np
import pydantic
import numpydantic
import torch


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors.

    Args:
        device: Device to perform computations on. If None, uses CPU (numpy).
                If 'cuda' or torch.device, uses GPU (torch tensors).
    """

    def __init__(self, device=None):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly
        self._device = device
        self._use_gpu = device is not None and (device != "cpu")

    def update(self, batch: np.ndarray | torch.Tensor) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch: An array/tensor where all dimensions except the last are batch dimensions.
                   Can be numpy array or torch tensor.
        """
        # Convert to appropriate format based on device
        if self._use_gpu:
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch).to(self._device)
            elif not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch, device=self._device)
            elif batch.device != self._device:
                batch = batch.to(self._device)
        else:
            if isinstance(batch, torch.Tensor):
                batch = batch.cpu().numpy()
            else:
                batch = np.asarray(batch)

        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape

        if self._use_gpu:
            self._update_gpu(batch, num_elements, vector_length)
        else:
            self._update_cpu(batch, num_elements, vector_length)

    def _update_cpu(self, batch: np.ndarray, num_elements: int, vector_length: int) -> None:
        """CPU version of update using numpy."""
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def _update_gpu(self, batch: torch.Tensor, num_elements: int, vector_length: int) -> None:
        """GPU version of update using torch tensors."""
        if self._count == 0:
            self._mean = torch.mean(batch, dim=0)
            self._mean_of_squares = torch.mean(batch**2, dim=0)
            self._min = torch.min(batch, dim=0)[0]
            self._max = torch.max(batch, dim=0)[0]
            self._histograms = [torch.zeros(self._num_quantile_bins, device=self._device) for _ in range(vector_length)]
            self._bin_edges = [
                torch.linspace(
                    self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1, device=self._device
                )
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.numel():
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = torch.max(batch, dim=0)[0]
            new_min = torch.min(batch, dim=0)[0]
            max_changed = torch.any(new_max > self._max)
            min_changed = torch.any(new_min < self._min)
            self._max = torch.maximum(self._max, new_max)
            self._min = torch.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = torch.mean(batch, dim=0)
        batch_mean_of_squares = torch.mean(batch**2, dim=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        if self._use_gpu:
            variance = self._mean_of_squares - self._mean**2
            stddev = torch.sqrt(torch.maximum(torch.tensor(0.0, device=self._device), variance))
            q01, q99 = self._compute_quantiles([0.01, 0.99])
            # Convert to numpy for output
            return NormStats(
                mean=self._mean.cpu().numpy(), std=stddev.cpu().numpy(), q01=q01.cpu().numpy(), q99=q99.cpu().numpy()
            )
        else:
            variance = self._mean_of_squares - self._mean**2
            stddev = np.sqrt(np.maximum(0, variance))
            q01, q99 = self._compute_quantiles([0.01, 0.99])
            return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        if self._use_gpu:
            for i in range(len(self._histograms)):
                old_edges = self._bin_edges[i]
                new_edges = torch.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1, device=self._device)

                # Redistribute the existing histogram counts to the new bins
                # Use bin centers instead of left edges for better redistribution
                old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])

                # Manual histogram computation since torch.histogram doesn't support CUDA
                new_hist = torch.zeros(self._num_quantile_bins, device=self._device)
                # Make tensors contiguous to avoid warning
                old_centers = old_centers.contiguous()
                new_edges_left = new_edges[:-1].contiguous()
                indices = torch.searchsorted(new_edges_left, old_centers, right=False)
                indices = torch.clamp(indices, 0, len(new_edges) - 2)

                # Use scatter_add_ for efficient accumulation
                new_hist.scatter_add_(0, indices, self._histograms[i])

                self._histograms[i] = new_hist
                self._bin_edges[i] = new_edges
        else:
            for i in range(len(self._histograms)):
                old_edges = self._bin_edges[i]
                new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

                # Redistribute the existing histogram counts to the new bins
                # Use bin centers instead of left edges for better redistribution
                old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])
                new_hist, _ = np.histogram(old_centers, bins=new_edges, weights=self._histograms[i])

                self._histograms[i] = new_hist
                self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray | torch.Tensor) -> None:
        """Update histograms with new vectors."""
        if self._use_gpu:
            for i in range(batch.shape[1]):
                # torch.histogram doesn't support CUDA, use torch.histc or manual binning
                # Use torch.searchsorted for manual binning
                # Make tensors contiguous to avoid warning
                data = batch[:, i].contiguous()
                edges = self._bin_edges[i]
                edges_left = edges[:-1].contiguous()
                # Find which bin each value belongs to
                indices = torch.searchsorted(edges_left, data, right=False)
                # Clamp to valid bin range
                indices = torch.clamp(indices, 0, len(edges) - 2)
                # Count occurrences in each bin
                hist = torch.bincount(indices, minlength=self._num_quantile_bins)
                # Ensure correct size (bincount might return longer array)
                hist = hist[: self._num_quantile_bins]
                self._histograms[i] += hist
        else:
            for i in range(batch.shape[1]):
                hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
                self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        if self._use_gpu:
            for q in quantiles:
                target_count = q * self._count
                q_values = []
                for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                    cumsum = torch.cumsum(hist, dim=0)
                    idx = torch.searchsorted(cumsum, torch.tensor(target_count, device=self._device))
                    q_values.append(edges[idx])
                results.append(torch.stack(q_values))
            return results
        else:
            for q in quantiles:
                target_count = q * self._count
                q_values = []
                for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                    cumsum = np.cumsum(hist)
                    idx = np.searchsorted(cumsum, target_count)
                    q_values.append(edges[idx])
                results.append(np.array(q_values))
            return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
