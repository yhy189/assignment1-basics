from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils
from jaxtyping import Float, Int

try:
    from .module import (
        Embedding,
        Linear,
        MultiHeadSelfAttention,
        RMSNorm,
        RoPE,
        SwiGLU,
    )
except ImportError:
    from module import Embedding, Linear, MultiHeadSelfAttention, RMSNorm, RoPE, SwiGLU


class LatentMultiHeadSelfAttention(nn.Module):
    """
    MLA block:
    1) project token states to a smaller latent space
    2) run MHA in latent space
    3) project back to model dimension
    """

    def __init__(
        self,
        d_model: int,
        latent_d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if latent_d_model <= 0:
            raise ValueError("latent_d_model must be positive")
        if latent_d_model % num_heads != 0:
            raise ValueError("latent_d_model must be divisible by num_heads")

        self.down = Linear(d_model, latent_d_model, device=device, dtype=dtype)
        self.up = Linear(latent_d_model, d_model, device=device, dtype=dtype)

        rope = RoPE(rope_theta, latent_d_model // num_heads, max_seq_len, device)
        self.latent_mha = MultiHeadSelfAttention(
            d_model=latent_d_model,
            num_heads=num_heads,
            pos_encoder=rope,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len d_model"],
        token_positions: Int[torch.Tensor, "seq_len"] | None = None,
    ) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        h = self.down(x)
        h = self.latent_mha(h, token_positions)
        h = self.up(h)
        return h


class MLATransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        latent_d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.mla_attn = LatentMultiHeadSelfAttention(
            d_model=d_model,
            latent_d_model=latent_d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.token_positions: Int[torch.Tensor, "max_seq_len"] = torch.arange(
            max_seq_len, device=device
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len d_model"],
    ) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        x_residual = x
        x = self.norm1(x)
        x = self.mla_attn(x, self.token_positions[: x.shape[-2]])
        x = x + x_residual

        x_residual = x
        x = self.norm2(x)
        x = self.swiglu(x)
        x = x + x_residual

        return x


class MLATransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        latent_d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        activation_checkpointing: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MLATransformerBlock(
                    d_model=d_model,
                    latent_d_model=latent_d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device, dtype)
        self.activation_checkpointing = activation_checkpointing

    def forward(
        self,
        token_ids: Int[torch.Tensor, "batch_size sequence_length"],
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embedding(token_ids)

        for block in self.layers:
            if self.activation_checkpointing and self.training:
                x = checkpoint_utils.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm_final(x)
        logits = self.output_embedding(x)
        return logits
