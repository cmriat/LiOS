# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""PyTorch preprocessing utilities."""

import logging
from collections.abc import Sequence

import torch

from pi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


# ============================================================================
# HSV conversion helper functions (fully aligned with augmax implementation)
# ============================================================================


def rgb_to_hsv_torch(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert RGB to HSV (fully aligned with augmax.utils.rgb_to_hsv implementation).

    Reference: https://en.wikipedia.org/wiki/HSL_and_HSV#Color_conversion_formulae

    Args:
        rgb: shape [batch, height, width, 3], values in [0, 1]

    Returns:
        hue: shape [batch, height, width], values in [0, 1]
        saturation: shape [batch, height, width], values in [0, 1]
        value: shape [batch, height, width], values in [0, 1]
    """
    value = torch.max(rgb, dim=-1)[0]
    range_val = value - torch.min(rgb, dim=-1)[0]

    argmax = torch.argmax(rgb, dim=-1)

    # Calculate second and third channel indices
    second_idx = torch.remainder(argmax + 1, 3)
    third_idx = torch.remainder(argmax + 2, 3)

    # Use gather to get corresponding channel values
    second_channel = torch.gather(rgb, -1, second_idx.unsqueeze(-1)).squeeze(-1)
    third_channel = torch.gather(rgb, -1, third_idx.unsqueeze(-1)).squeeze(-1)

    # Calculate hue (identical to augmax implementation)
    hue = torch.where(
        range_val == 0.0,
        torch.zeros_like(range_val),
        (2 * argmax.float() + (second_channel - third_channel) / (range_val + 1e-10)) / 6.0,
    )

    # Calculate saturation
    saturation = torch.where(value == 0.0, torch.zeros_like(value), range_val / (value + 1e-10))

    return hue, saturation, value


def hsv_to_rgb_torch(hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Convert HSV to RGB (fully aligned with augmax.utils.hsv_to_rgb implementation).

    Reference: https://en.wikipedia.org/wiki/HSL_and_HSV#Color_conversion_formulae

    Args:
        hue: shape [batch, height, width], values in [0, 1]
        saturation: shape [batch, height, width], values in [0, 1]
        value: shape [batch, height, width], values in [0, 1]

    Returns:
        rgb: shape [batch, height, width, 3], values in [0, 1]
    """
    # augmax implementation: n = [5, 3, 1]
    n = torch.tensor([5, 3, 1], dtype=hue.dtype, device=hue.device)
    k = torch.remainder(n.view(1, 1, 1, 3) + hue.unsqueeze(-1) * 6, 6)

    # augmax implementation: f = value - value * saturation * max(0, min(min(k, 4-k), 1))
    f = value.unsqueeze(-1) - value.unsqueeze(-1) * saturation.unsqueeze(-1) * torch.maximum(
        torch.zeros_like(k), torch.minimum(torch.minimum(k, 4 - k), torch.ones_like(k))
    )

    return f


def adjust_brightness_torch(value: torch.Tensor, brightness: torch.Tensor) -> torch.Tensor:
    """Adjust brightness (fully aligned with augmax.functional.colorspace.adjust_brightness).

    augmax implementation:
        if brightness < 0:
            return value * (1.0 + brightness)
        else:
            return value * (1.0 - brightness) + brightness

    Args:
        value: tensor of shape [batch, height, width]
        brightness: tensor of shape [batch] or [batch, 1, 1], range [-strength, +strength]

    Returns:
        adjusted value: [batch, height, width]
    """
    # Ensure brightness has correct shape for broadcasting
    if brightness.dim() == 1:
        brightness = brightness.view(-1, 1, 1)

    return torch.where(brightness < 0.0, value * (1.0 + brightness), value * (1.0 - brightness) + brightness)


def adjust_contrast_torch(value: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
    """Adjust contrast (fully aligned with augmax.functional.colorspace.adjust_contrast).

    augmax implementation:
        slant = tan((contrast + 1.0) * (pi / 4))
        # Uses piecewise function mapping, see:
        # https://www.desmos.com/calculator/yxnm5siet4

    Args:
        value: tensor of shape [batch, height, width]
        contrast: tensor of shape [batch] or [batch, 1, 1], range [-strength, +strength]

    Returns:
        adjusted value: [batch, height, width]
    """
    # Ensure contrast has correct shape for broadcasting
    if contrast.dim() == 1:
        contrast = contrast.view(-1, 1, 1)

    slant = torch.tan((contrast + 1.0) * (torch.pi / 4))

    # Piecewise breakpoints from augmax implementation
    p1 = (slant - slant**2) / (2 * (1 - slant**2))
    p2 = 1 - p1

    # Piecewise function (identical to augmax)
    result = torch.where(
        value < p1, value / slant, torch.where(value > p2, (value / slant) + 1 - 1 / slant, slant * (value - 0.5) + 0.5)
    )

    return result


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]
                batch_size = image.shape[0]

                # Random crop and resize - per-sample randomization
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop with per-sample offsets - vectorized implementation
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Generate independent random crop positions for each sample in batch
                    start_h = torch.randint(0, max_h + 1, (batch_size,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (batch_size,), device=image.device)

                    # Vectorized crop using advanced indexing
                    # Create mesh grids for crop regions
                    h_indices = torch.arange(crop_height, device=image.device).view(1, -1, 1)  # [1, crop_h, 1]
                    w_indices = torch.arange(crop_width, device=image.device).view(1, 1, -1)  # [1, 1, crop_w]

                    # Add start positions for each sample in batch
                    h_coords = start_h.view(-1, 1, 1) + h_indices  # [batch, crop_h, 1]
                    w_coords = start_w.view(-1, 1, 1) + w_indices  # [batch, 1, crop_w]

                    # Broadcast and gather
                    h_coords = h_coords.expand(batch_size, crop_height, crop_width)  # [batch, crop_h, crop_w]
                    w_coords = w_coords.expand(batch_size, crop_height, crop_width)  # [batch, crop_h, crop_w]

                    # Use advanced indexing to crop all samples at once
                    batch_indices = (
                        torch.arange(batch_size, device=image.device)
                        .view(-1, 1, 1)
                        .expand(batch_size, crop_height, crop_width)
                    )
                    image = image[batch_indices, h_coords, w_coords, :]  # [batch, crop_h, crop_w, channels]

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # Random rotation (small angles) - per-sample randomization with batch processing
                # Generate independent random angles for each sample in batch
                angles = torch.rand(batch_size, device=image.device) * 10 - 5  # Random angles between -5 and 5 degrees

                # Convert to radians
                angles_rad = angles * torch.pi / 180.0

                # Create rotation matrices for all samples at once
                cos_angles = torch.cos(angles_rad)  # [batch_size]
                sin_angles = torch.sin(angles_rad)  # [batch_size]

                # Create base meshgrid (shared across batch)
                grid_x = torch.linspace(-1, 1, width, device=image.device)
                grid_y = torch.linspace(-1, 1, height, device=image.device)
                grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")  # [height, width]

                # Expand grids to batch dimension
                grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, height, width]
                grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, height, width]

                # Apply per-sample rotation transformations (vectorized)
                cos_a = cos_angles.view(batch_size, 1, 1)  # [batch, 1, 1]
                sin_a = sin_angles.view(batch_size, 1, 1)  # [batch, 1, 1]

                grid_x_rot = grid_x * cos_a - grid_y * sin_a  # [batch, height, width]
                grid_y_rot = grid_x * sin_a + grid_y * cos_a  # [batch, height, width]

                # Stack grids for grid_sample: [batch, height, width, 2]
                grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                # Apply rotation to entire batch at once using grid_sample
                image = torch.nn.functional.grid_sample(
                    image.permute(0, 3, 1, 2).to(torch.float32),  # [batch, h, w, c] -> [batch, c, h, w]
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [batch, c, h, w] -> [batch, h, w, c]

            # ================================================================
            # Color augmentations - fully aligned with augmax.ColorJitter
            # ================================================================
            batch_size = image.shape[0]

            # Convert to HSV color space
            hue, saturation, value = rgb_to_hsv_torch(image)

            # Generate per-sample independent random parameters (aligned with augmax parameter ranges)
            # JAX config: ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            brightness_params = (torch.rand(batch_size, device=image.device) * 2 - 1) * 0.3  # [-0.3, 0.3]
            contrast_params = (torch.rand(batch_size, device=image.device) * 2 - 1) * 0.4  # [-0.4, 0.4]
            # Note: saturation is not adjusted because augmax implementation has a bug (line 275 has no assignment)

            # Apply brightness and contrast adjustments on the V channel in HSV space
            value = adjust_brightness_torch(value, brightness_params)
            value = adjust_contrast_torch(value, contrast_params)

            # Convert back to RGB
            image = hsv_to_rgb_torch(hue, saturation, value)

            # Clamp to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
