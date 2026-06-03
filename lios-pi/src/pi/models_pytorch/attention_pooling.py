# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Perceiver Resampler for compressing historical state sequences.

This module implements a Perceiver Resampler variant that uses cross-attention
to compress a long sequence of T historical states into M summary tokens.
"""

import math
import torch
import torch.nn as nn


class PerceiverResamplerLayer(nn.Module):
    """Single Perceiver Resampler layer with cross-attention and FFN.

    Args:
        d_model: Hidden dimension size
        num_heads: Number of attention heads
        ffn_ratio: FFN expansion multiplier
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass through one Perceiver layer.

        Args:
            queries: Query tensor [B, M, D] (latents or previous layer output)
            context: Context tensor [B, T, D] (input sequence)

        Returns:
            Updated queries [B, M, D]
        """
        # Cross-attention: Q=queries, K/V=context
        attn_out, _ = self.attn(queries, context, context, need_weights=False)
        h = self.ln1(queries + attn_out)
        out = self.ln2(h + self.ffn(h))
        return out


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for latent tokens to exchange information.

    Args:
        d_model: Hidden dimension size
        num_heads: Number of attention heads
        ffn_ratio: FFN expansion multiplier
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through self-attention layer.

        Args:
            x: Input tensor [B, M, D] (latent tokens)

        Returns:
            Updated latent tokens [B, M, D]
        """
        # Self-attention: Q=K=V=x
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        h = self.ln1(x + attn_out)
        out = self.ln2(h + self.ffn(h))
        return out


class PerceiverResampler(nn.Module):
    """Perceiver Resampler for temporal attention pooling.

    Compresses a sequence of T historical states into M summary tokens using
    multiple layers of cross-attention with optional self-attention layers.

    Args:
        d_model: Hidden dimension size (must match input dimension)
        num_latents: Number of summary tokens M (default: 32)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of cross-attention layers (default: 1)
        use_self_attn: If True, insert a self-attention layer between each pair of
                       cross-attention layers for latent interaction (default: False)
        ffn_ratio: FFN expansion multiplier (default: 4)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_latents: int = 32,
        num_heads: int = 8,
        num_layers: int = 1,
        use_self_attn: bool = False,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_self_attn = use_self_attn
        # Initialize latent queries with truncated normal distribution
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) / math.sqrt(d_model))

        # Create cross-attention layers
        self.cross_attn_layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Create self-attention layers between cross-attention layers
        # If num_layers = 6, we add self-attn between each pair: 5 self-attn layers
        # Structure: Cross → Self → Cross → Self → ... → Cross
        if use_self_attn and num_layers > 1:
            num_self_attn = num_layers - 1
            self.self_attn_layers = nn.ModuleList(
                [
                    SelfAttentionLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        ffn_ratio=ffn_ratio,
                        dropout=dropout,
                    )
                    for _ in range(num_self_attn)
                ]
            )
        else:
            self.self_attn_layers = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through Perceiver Resampler.

        Args:
            X: Input tensor of shape [B, T, d_model] where T is the sequence length
               (e.g., T=400 for historical states)

        Returns:
            Z: Compressed output tensor of shape [B, M, d_model] where M is num_latents
               (e.g., M=32 summary tokens)
        """
        B = X.shape[0]
        # Initialize latents
        queries = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # Apply cross-attention and self-attention layers alternately
        # Structure: Cross → Self → Cross → Self → ... → Self → Cross
        for i, cross_attn_layer in enumerate(self.cross_attn_layers):
            # Apply cross-attention
            queries = cross_attn_layer(queries, X)  # [B, M, D]

            # Apply self-attention after each cross-attention (except the last one)
            if self.use_self_attn and i < self.num_layers - 1:
                queries = self.self_attn_layers[i](queries)  # [B, M, D]

        return queries


