# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Pi model configuration."""

import dataclasses

from typing_extensions import override

import pi.models.gemma as _gemma
import pi.models.model as _model


@dataclasses.dataclass(frozen=True)
class PiConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = False

    # Number of historical state frames to use as input. 1 = current state only;
    # >1 = include history (compressed by the Perceiver resampler under Pi0.5).
    state_history_frames: int = 1

    # Random state-acquisition delay applied at training time only. 0 disables;
    # N > 0 samples a uniform delay in [0, N] frames to mimic real-world latency.
    state_delay_frames: int = 0

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)

        # Pi0 mode requires continuous state input (discrete states are Pi0.5 only).
        if self.pi05 is False:
            object.__setattr__(self, "discrete_state_input", False)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0
