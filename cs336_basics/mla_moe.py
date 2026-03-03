from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils
from jaxtyping import Float, Int

try:
    from .mla import LatentMultiHeadSelfAttention
    from .module import Embedding, Linear, RMSNorm
    from .moe import SwitchFeedForward
except ImportError:
    from mla import LatentMultiHeadSelfAttention
    from module import Embedding, Linear, RMSNorm
    from moe import SwitchFeedForward


class MLAMoETransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        latent_d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        num_experts: int,
        moe_top_k: int,
        moe_fixed_expert_idx: int,
        moe_expert_d_ffs: list[int] | None,
        moe_aux_loss_coef: float,
        moe_z_loss_coef: float,
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
        self.moe = SwitchFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=moe_top_k,
            fixed_expert_idx=moe_fixed_expert_idx,
            expert_d_ffs=moe_expert_d_ffs,
            aux_loss_coef=moe_aux_loss_coef,
            z_loss_coef=moe_z_loss_coef,
            device=device,
            dtype=dtype,
        )
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.token_positions: Int[torch.Tensor, "max_seq_len"] = torch.arange(
            max_seq_len, device=device
        )
        self.last_aux_loss: torch.Tensor | None = None

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
        x = self.moe(x)
        x = x + x_residual

        self.last_aux_loss = self.moe.get_aux_loss()
        return x

    def get_aux_loss(self) -> torch.Tensor:
        if self.last_aux_loss is None:
            return torch.tensor(0.0, device=self.token_positions.device)
        return self.last_aux_loss


class MLAMoETransformerLM(nn.Module):
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
        num_experts: int,
        moe_top_k: int = 1,
        moe_fixed_expert_idx: int = 0,
        moe_expert_d_ffs: list[int] | None = None,
        moe_aux_loss_coef: float = 1e-2,
        moe_z_loss_coef: float = 1e-4,
        activation_checkpointing: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MLAMoETransformerBlock(
                    d_model=d_model,
                    latent_d_model=latent_d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    moe_fixed_expert_idx=moe_fixed_expert_idx,
                    moe_expert_d_ffs=moe_expert_d_ffs,
                    moe_aux_loss_coef=moe_aux_loss_coef,
                    moe_z_loss_coef=moe_z_loss_coef,
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
        self.last_aux_loss: torch.Tensor | None = None

    def forward(
        self,
        token_ids: Int[torch.Tensor, "batch_size sequence_length"],
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embedding(token_ids)

        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        for block in self.layers:
            if self.activation_checkpointing and self.training:
                x = checkpoint_utils.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            aux_loss = aux_loss + block.get_aux_loss().to(torch.float32)

        x = self.norm_final(x)
        logits = self.output_embedding(x)
        self.last_aux_loss = aux_loss
        return logits

    def get_aux_loss(self) -> torch.Tensor:
        if self.last_aux_loss is None:
            return torch.tensor(0.0, device=self.output_embedding.W.device)
        return self.last_aux_loss
