# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Model configuration and loading utilities."""

import abc
import enum
import logging
import dataclasses

import numpy as np
import torch
import safetensors.torch

import pi.models_pytorch.pi0_pytorch as pi0_pytorch

logger = logging.getLogger("pi")


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models.

    Specific models should inherit from this class, and implement the `create` method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        missing_keys, unexpected_keys = safetensors.torch.load_model(model, weight_path, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading model: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")
        return model


import dataclasses
from typing import Dict, Generic, TypeVar, Optional

ArrayT = TypeVar("ArrayT")  # numpy.ndarray | torch.Tensor


@dataclasses.dataclass(frozen=True)
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    Annotations were originally JAXtyping shapes in openpi; we keep the original
    shape hints in trailing comments for reference.
    """

    images: Dict[str, ArrayT]  # was: at.Float[ArrayT, "*b h w c"]
    image_masks: Dict[str, ArrayT]  # was: at.Bool[ArrayT, "*b"]
    state: ArrayT  # was: at.Float[ArrayT, "*b s"]

    tokenized_prompt: Optional[ArrayT] = None  # was: at.Int[ArrayT, "*b l"]
    tokenized_prompt_mask: Optional[ArrayT] = None  # was: at.Bool[ArrayT, "*b l"]
    token_ar_mask: Optional[ArrayT] = None  # was: at.Int[ArrayT, "*b l"]
    token_loss_mask: Optional[ArrayT] = None  # was: at.Bool[ArrayT, "*b l"]

    @classmethod
    def from_dict(cls, data: Dict) -> "Observation[ArrayT]":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        for key in data["image"]:
            if hasattr(data["image"][key], "dtype"):
                if data["image"][key].dtype == np.uint8:
                    data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                    data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> Dict:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Actions = at.Float[ArrayT, "*b ah ad"]
Actions = ArrayT  # Shape: (*batch, action_horizon, action_dim)
