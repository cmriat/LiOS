# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Exponential Moving Average utilities for model training."""

import torch


# ------------------------- EMA utilities -------------------------
class EMAModel:
    """Exponential Moving Average of model parameters for improved generalization.

    This class maintains a shadow copy of the model parameters and updates them
    using exponential moving average during training.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        """Initialize EMA model.

        Args:
            model: The model to track
            decay: EMA decay rate (default: 0.999)
        """
        self.decay = decay
        self.shadow = {}
        self.register(model)

    def register(self, model: torch.nn.Module) -> None:
        """Register model parameters for EMA tracking."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA parameters using foreach operations for performance."""
        model_params = []
        shadow_params = []

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                model_params.append(param)
                shadow_params.append(self.shadow[name])

        if len(model_params) > 0:
            torch._foreach_lerp_(shadow_params, model_params, weight=1.0 - self.decay)

    def state_dict(self) -> dict:
        """Return EMA state dictionary for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Apply EMA parameters to the model (for evaluation/inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restore original model parameters (after evaluation)."""
        # This would require storing original params, which we skip for now
        # since we primarily use EMA for checkpoint saving
        pass
