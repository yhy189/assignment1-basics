from __future__ import annotations

import torch
import torch.nn as nn
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


class SwitchFeedForward(nn.Module):
    """Sparse MoE feed-forward layer with one always-on shared expert."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 1,
        fixed_expert_idx: int = 0,
        expert_d_ffs: list[int] | None = None,
        aux_loss_coef: float = 1e-2,
        z_loss_coef: float = 1e-4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2 for MoE")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        if fixed_expert_idx < 0 or fixed_expert_idx >= num_experts:
            raise ValueError("fixed_expert_idx must satisfy 0 <= fixed_expert_idx < num_experts")
        if expert_d_ffs is None:
            expert_d_ffs = [d_ff] * num_experts
        if len(expert_d_ffs) != num_experts:
            raise ValueError("expert_d_ffs length must equal num_experts")
        if any(v <= 0 for v in expert_d_ffs):
            raise ValueError("expert_d_ffs values must be positive")

        self.num_experts = num_experts
        self.top_k = top_k
        self.fixed_expert_idx = fixed_expert_idx
        self.expert_d_ffs = [int(v) for v in expert_d_ffs]
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef

        self.router = Linear(d_model, num_experts, device=device, dtype=dtype)
        self.experts = nn.ModuleList(
            [
                SwiGLU(d_model, self.expert_d_ffs[idx], device=device, dtype=dtype)
                for idx in range(num_experts)
            ]
        )
        self.last_aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len d_model"],
    ) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        batch_size, seq_len, d_model = x.shape
        flat_x = x.reshape(batch_size * seq_len, d_model)

        router_logits = self.router(flat_x)
        router_probs = torch.softmax(router_logits.to(torch.float32), dim=-1).to(flat_x.dtype)

        fixed_idx = self.fixed_expert_idx
        fixed_indices = torch.full(
            (flat_x.shape[0], 1), fixed_idx, device=flat_x.device, dtype=torch.long
        )
        fixed_probs = router_probs[:, fixed_idx : fixed_idx + 1]

        other_k = self.top_k - 1
        if other_k > 0:
            remaining_indices = [idx for idx in range(self.num_experts) if idx != fixed_idx]
            remaining_indices_tensor = torch.tensor(
                remaining_indices, device=flat_x.device, dtype=torch.long
            )
            remaining_probs = router_probs[:, remaining_indices_tensor]
            other_probs, other_local = torch.topk(remaining_probs, k=other_k, dim=-1)
            other_indices = remaining_indices_tensor[other_local]

            topk_indices = torch.cat([fixed_indices, other_indices], dim=1)
            topk_probs = torch.cat([fixed_probs, other_probs], dim=1)
        else:
            topk_indices = fixed_indices
            topk_probs = fixed_probs

        active_k = topk_indices.shape[1]
        topk_probs = topk_probs / torch.clamp(topk_probs.sum(dim=-1, keepdim=True), min=1e-9)

        output = torch.zeros_like(flat_x)
        for expert_idx, expert in enumerate(self.experts):
            for slot_idx in range(active_k):
                token_mask = topk_indices[:, slot_idx] == expert_idx
                if not bool(token_mask.any()):
                    continue
                expert_in = flat_x[token_mask]
                expert_out = expert(expert_in)
                gate = topk_probs[token_mask, slot_idx].unsqueeze(-1).to(expert_out.dtype)
                output[token_mask] += expert_out * gate

        routing_fraction = torch.zeros_like(router_probs)
        routing_fraction.scatter_(1, topk_indices, 1.0 / float(active_k))
        expert_load = routing_fraction.mean(dim=0)
        expert_importance = router_probs.mean(dim=0)
        load_balancing_loss = float(self.num_experts) * torch.sum(expert_importance * expert_load)

        z_loss = torch.logsumexp(router_logits.to(torch.float32), dim=-1).pow(2).mean()

        self.last_aux_loss = load_balancing_loss * self.aux_loss_coef + z_loss * self.z_loss_coef
        return output.reshape(batch_size, seq_len, d_model)

    def get_aux_loss(self) -> torch.Tensor:
        if self.last_aux_loss is None:
            return torch.tensor(0.0, device=self.router.W.device)
        return self.last_aux_loss


class MoETransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
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
        self.rope = RoPE(rope_theta, d_model // num_heads, max_seq_len, device)
        self.mha = MultiHeadSelfAttention(
            d_model,
            num_heads,
            pos_encoder=self.rope,
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
        x = self.mha(x, self.token_positions[: x.shape[-2]])
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


class MoETransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(
                    d_model=d_model,
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
        self.last_aux_loss: torch.Tensor | None = None

    def forward(
        self,
        token_ids: Int[torch.Tensor, "batch_size sequence_length"],
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embedding(token_ids)

        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        for block in self.layers:
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
