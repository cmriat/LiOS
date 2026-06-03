# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""PyTorch implementation of Gemma model."""

from typing import Literal

import torch
import pytest
from torch import nn
from transformers.models.auto import CONFIG_MAPPING

from pi.models.gemma_ import modeling_gemma
from pi.models.gemma_.modeling_gemma import GemmaForCausalLM
from pi.models.gemma_.configuration_gemma import GemmaConfig
from pi.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = GemmaConfig(
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)  # vlm
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)  # action expert
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def compute_layer_complete(self, layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
        with torch.profiler.record_function(f"layer_{layer_idx}.start"):
            models = [self.paligemma.language_model, self.gemma_expert.model]

            query_states = []
            key_states = []
            value_states = []
            gates = []

            # ======== Q/K/V + input LN ========
            with torch.profiler.record_function("1.input_ln_and_qkv"):
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]

                    with torch.profiler.record_function(f"1.{i}.input_layernorm"):
                        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                    with torch.profiler.record_function(f"1.{i}.qkv_linear"):
                        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

            # ======== concat ========
            with torch.profiler.record_function("2.concat_qkv"):
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

            # ======== rotary emb ========
            with torch.profiler.record_function("3.rotary_embedding"):
                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

            # ======== attention ========
            with torch.profiler.record_function("4.attention_eager_forward"):
                attention_mask = attention_mask.to(query_states.dtype)
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling
                # 1. eager attention
                att_out_eager, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                att_output = att_out_eager

            # ======== reshape heads ========
            with torch.profiler.record_function("5.reshape_attention_output"):
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                batch_size = query_states.shape[0]
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

            # ======== projection + MLP per model ========
            outputs_embeds = []
            start_pos = 0
            with torch.profiler.record_function("6.post_attention_blocks"):
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    # o_proj
                    with torch.profiler.record_function(f"6.{i}.o_proj"):
                        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    with torch.profiler.record_function(f"6.{i}.first_gated_residual"):
                        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
                    after_first_residual = out_emb.clone()  # intentionally left as-is

                    # layernorm
                    with torch.profiler.record_function(f"6.{i}.post_attention_layernorm"):
                        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                            out_emb = out_emb.to(dtype=torch.bfloat16)

                    # MLP
                    with torch.profiler.record_function(f"6.{i}.mlp"):
                        out_emb = layer.mlp(out_emb)

                    # second residual
                    with torch.profiler.record_function(f"6.{i}.second_gated_residual"):
                        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)

                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

            return outputs_embeds

    # Define final norm computation function for gradient checkpointing
    def compute_final_norms(self, models, inputs_embeds, adarms_cond):
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
            outputs_embeds.append(out_emb)
        return outputs_embeds

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )  # last_hidden_state
            prefix_past_key_values = prefix_output.past_key_values  # per-layer K/V cache
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                with torch.profiler.record_function(f"L{layer_idx:02d}.compute_layer_complete"):
                    inputs_embeds = self.compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

            # final norm
            outputs_embeds = self.compute_final_norms(models, inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
