# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""PyTorch implementation of Pi0 model."""

import math
import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

import pi.models.gemma as _gemma
import pi.models_pytorch.preprocessing_pytorch as _preprocessing
from pi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from pi.models_pytorch.attention_pooling import PerceiverResampler


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
            # When pi05 + state_history_frames > 1, we need state_proj for historical states
            if config.state_history_frames > 1:
                self.state_proj = nn.Linear(32, action_expert_config.width)
                # Add Perceiver Resampler for compressing historical states
                # Compress T=state_history_frames -> M=64 summary tokens
                self.perceiver_resampler = PerceiverResampler(
                    d_model=action_expert_config.width,
                    num_latents=32,  # M=64 summary tokens
                    num_heads=8,
                    num_layers=2,  # cross-attention layer count; >2 gives diminishing returns
                    use_self_attn=True,  # Use self-attention between cross-attention layers
                    ffn_ratio=4,
                    dropout=0.0,
                )
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        self.register_buffer("alpha_t", torch.as_tensor(1.5, dtype=torch.float32), persistent=False)
        self.register_buffer("beta_t", torch.as_tensor(1.0, dtype=torch.float32), persistent=False)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_beta(self, bsize):
        dist = torch.distributions.Beta(self.alpha_t, self.beta_t)
        return dist.sample((bsize,))

    def sample_time(self, bsize, _device):
        time_beta = self.sample_beta(bsize)
        time = time_beta * 0.999 + 0.001
        return time

    def image_embed_func(self, img):
        return self.paligemma_with_expert.embed_image(img)
        # Process language tokens

    def lang_embed_func(self, lang_tokens):
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        return lang_emb * math.sqrt(lang_emb_dim)

    def _embed_images(self, images, img_masks):
        """Vectorized version of _embed_images: one call to image_embed_func."""
        assert isinstance(images, list)
        num_images = len(images)
        B = images[0].shape[0]

        pixel_values = torch.cat(images, dim=0)  # [num_images * B, 3, H, W]
        img_emb_flat = self.image_embed_func(pixel_values)  # [num_images * B, L_img, D]
        _, L_img, D = img_emb_flat.shape

        img_emb_grouped = img_emb_flat.view(num_images, B, L_img, D)
        img_emb_list = [img_emb_grouped[i] for i in range(num_images)]

        img_masks_stack = torch.stack(list(img_masks), dim=0)  # [num_images, B]
        pad_masks_stack = img_masks_stack[:, :, None].expand(num_images, B, L_img)
        pad_masks_list = [pad_masks_stack[i] for i in range(num_images)]
        return img_emb_list, pad_masks_list

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer.

        To prepare for PaliGemma transformer processing.
        """
        embs, pad_masks = self._embed_images(images, img_masks)

        # language embeddings
        lang_emb = self.lang_embed_func(lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # embs  [B, L_i, D]
        total_tokens = sum(e.shape[1] for e in embs)  # = image_emb_total_len + lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)  # [B, total_tokens, D]
        pad_masks = torch.cat(pad_masks, dim=1)  # [B, total_tokens]

        bsize = pad_masks.shape[0]
        att_masks = torch.zeros(
            bsize,
            total_tokens,
            dtype=torch.bool,
            device=pad_masks.device,
        )

        return embs, pad_masks, att_masks

    def state_proj_func(self, state):
        # Convert state to match state_proj dtype for mixed precision training
        target_dtype = self.state_proj.weight.dtype
        state = state.to(dtype=target_dtype)
        return self.state_proj(state)  # (batch, history, state_dim) -> (batch, history, hidden)

    def action_proj_func(self, noisy_actions):
        return self.action_in_proj(noisy_actions.to(dtype=self.action_in_proj.weight.dtype))

    def mlp_func(self, action_time_emb_):
        x = self.action_time_mlp_in(action_time_emb_)
        x = F.silu(x)
        return self.action_time_mlp_out(x)

    def time_mlp_func(self, time_emb):
        x = self.time_mlp_in(time_emb.to(dtype=self.time_mlp_in.weight.dtype))
        x = F.silu(x)  # swish == silu
        x = self.time_mlp_out(x)
        return F.silu(x)

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []

        state_token_len = 0

        # Process state if:
        # 1. pi0 mode (not self.pi05), OR
        # 2. pi05 mode with state_history_frames > 1 (historical states enter action expert)
        use_state_proj = (not self.pi05) or (self.pi05 and self.config.state_history_frames > 1)

        if use_state_proj:
            # Embed state - now supports historical states
            # state shape: (batch, history, state_dim) or (batch, state_dim)
            # Ensure state has history dimension
            if state.dim() == 2:
                # (batch, state_dim) -> (batch, 1, state_dim) for backward compatibility
                state = state.unsqueeze(1)

            bsize, history_frames = state.shape[0], state.shape[1]
            device = state.device
            # Project each historical frame independently using the same weight
            # state_proj: Linear(state_dim, hidden) will be applied to last dim

            state_emb = self.state_proj_func(state)

            # Add temporal positional encoding for historical states only when history_frames > 1
            # When history_frames == 1, we only have the current state, no need for temporal encoding
            if history_frames > 1:
                # state[:, 0, :] is the most recent (current) state
                # state[:, history_frames-1, :] is the oldest state
                # So we use reversed time indices: (history_frames-1, history_frames-2, ..., 1, 0)
                # to reflect that earlier positions in the sequence correspond to more recent states
                time_indices = torch.arange(history_frames - 1, -1, -1, dtype=torch.float32, device=device)

                # Generate sinusoidal positional embeddings for historical states
                # Use similar parameters as action timestep but adjusted for historical indexing
                temporal_emb = create_sinusoidal_pos_embedding(
                    time_indices,
                    state_emb.shape[-1],  # hidden dimension
                    min_period=1.0,  # Adjusted for frame indices
                    max_period=float(history_frames),  # Max period based on history length
                    device=device,
                )
                # temporal_emb shape: (history_frames, hidden)

                # Expand temporal embedding to match batch dimension and add to state embeddings
                temporal_emb = temporal_emb.unsqueeze(0).expand(
                    bsize, -1, -1
                )  # (1, history, hidden) -> (batch, history, hidden)
                state_emb = state_emb + temporal_emb.to(state_emb.dtype)

            # Apply Perceiver Resampler to compress historical states
            # from [batch, T=history_frames, hidden] to [batch, M=num_latents, hidden]
            if (
                self.pi05
                and hasattr(self, "perceiver_resampler")
                and history_frames > self.perceiver_resampler.num_latents
            ):
                # Only apply compression if history_frames > num_latents
                state_emb = self.perceiver_resampler(state_emb)
                compressed_frames = self.perceiver_resampler.num_latents
            else:
                # No compression needed or not available
                compressed_frames = history_frames

            state_token_len = compressed_frames
            # state_emb shape after compression: (batch, M, hidden) where M=num_latents or original history_frames
            embs.append(state_emb)

            # Create masks for compressed state tokens
            state_mask = torch.ones(bsize, compressed_frames, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP

        action_emb = self.action_proj_func(noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers

            action_time_emb = self.mlp_func(action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_func(time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb
        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_token_len = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_token_len, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        total_tokens = embs.shape[1]
        assert total_tokens == state_token_len + action_token_len
        att_1d = torch.zeros(
            total_tokens,
            dtype=embs.dtype,
            device=embs.device,
        )
        offset = 0
        # state part：first token set to 1
        if state_token_len > 0:
            att_1d[offset] = 1.0
            offset += state_token_len

        # action part：first token set to 1
        att_1d[offset] = 1.0
        att_masks = att_1d[None, :].expand(bsize, total_tokens)
        return embs, pad_masks, att_masks, adarms_cond

    def forward_func(self, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out

    def action_out_proj_func(self, suffix_out):
        dtype = suffix_out.dtype
        return F.linear(
            suffix_out, self.action_out_proj.weight.to(dtype=dtype), self.action_out_proj.bias.to(dtype=dtype)
        )

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)."""
        images = list(observation["image"].values())
        img_masks = list(observation["image_mask"].values())
        lang_tokens = observation["tokenized_prompt"]
        lang_masks = observation["tokenized_prompt_mask"]
        state = observation["state"]
        # 1. Preprocessing & Noise Sampling
        with torch.profiler.record_function("sample_time_and_noise"):
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)

            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

        # 2. Embedding Generation
        with torch.profiler.record_function("embed_prefix_and_suffix"):
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # 3. Mask Preparation
        with torch.profiler.record_function("prepare_attention_masks"):
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            # Prepare attention masks
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # 4. Transformer Backbone Forward
        with torch.profiler.record_function("transformer_backbone"):
            suffix_out = self.forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)
            suffix_out = suffix_out[:, -self.config.action_horizon :]
            suffix_out = suffix_out.to(dtype=torch.float32)

        # 5. Action Projection & Loss
        with torch.profiler.record_function("action_out_proj_and_loss"):
            v_t = self.action_out_proj_func(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)."""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # Pi0.5: lang_tokens already include the discretized state.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
