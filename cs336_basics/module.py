import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.linear.weight, 0, std, -3 * std, 3 * std)

    def forward(
            self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        return self.linear(x)


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(self.embedding.weight, 0, 1, -3, 3)

    def forward(
            self, token_ids: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        return self.embedding(token_ids)


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.rms_norm = nn.RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        nn.init.ones_(self.rms_norm.weight)

    def forward(
            self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.rms_norm(x)


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.silu = nn.SiLU()

    def forward(
            self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))


class RoPE(nn.Module):
    def __init__(
            self,
            theta: float = 10000.0,
            d_k: int = None,
            max_seq_len: int = None,
            device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.device = device

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor,
    ) -> torch.Tensor:
        if token_positions.dim() == 1:
            token_positions = token_positions.unsqueeze(0).unsqueeze(0)
        elif token_positions.dim() == 2:
            token_positions = token_positions.unsqueeze(1)

        while token_positions.dim() < x.dim() - 1:
            token_positions = token_positions.expand(*x.shape[:-2], -1)

        freqs = einsum(token_positions, self.inv_freq, "... seq_len, d_half -> ... seq_len d_half")

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x_even * cos[..., ::2] - x_odd * sin[..., ::2]
        x_rot[..., 1::2] = x_even * sin[..., ::2] + x_odd * cos[..., ::2]

        return x_rot


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            pos_encoder: nn.Module | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.pos_encoder = pos_encoder
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
            self,
            x: Float[torch.Tensor, "batch_size ... seq_len d_model"],
            token_positions: Int[torch.Tensor, " ... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "batch_size ... seq_len d_model"]:
        batch_size = x.shape[0]
        seq_len = x.shape[-2]

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        if self.pos_encoder is not None and token_positions is not None:
            Q = self.pos_encoder(Q, token_positions)
            K = self.pos_encoder(K, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool))

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=0.0
        )

        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        return self.W_o(attn_out)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            rope_theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rope = RoPE(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device
        )
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)
        self.mha = MultiHeadSelfAttention(
            d_model, num_heads, pos_encoder=self.rope, device=device, dtype=dtype
        )
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.token_positions = torch.arange(max_seq_len, device=device)

    def forward(
            self,
            x: Float[torch.Tensor, "batch_size seq_len d_model"],
    ) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        x_residual = x
        x = self.norm1(x)
        x = self.mha(x, self.token_positions[:x.shape[-2]])
        x = x + x_residual

        x_residual = x
        x = self.norm2(x)
        x = self.swiglu(x)
        x = x + x_residual

        return x


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float = 10000.0,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta, device, dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
            self,
            token_ids: Int[torch.Tensor, "batch_size sequence_length"],
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embedding(token_ids)
        for block in self.layers:
            x = block(x)
        x = self.norm_final(x)
        x = self.output_embedding(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    BATCH = 2
    SEQ_LEN = 16
    VOCAB_SIZE = 1000
    D_MODEL = 128
    NUM_HEADS = 4
    D_FF = 512
    NUM_LAYERS = 2
    MAX_SEQ = 32
    ROPE_THETA = 10000.0

    # 测试 Linear 层
    print("测试 Linear 层")
    linear = Linear(D_MODEL, D_FF, device, dtype)
    x_linear = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)
    out_linear = linear(x_linear)
    print(f"输入形状: {x_linear.shape}")
    print(f"输出形状: {out_linear.shape}")
    print("Linear 测试通过\n")

    # 测试 Embedding 层
    print("测试 Embedding 层")
    emb = Embedding(VOCAB_SIZE, D_MODEL, device, dtype)
    x_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)
    out_emb = emb(x_ids)
    print(f"输入形状: {x_ids.shape}")
    print(f"输出形状: {out_emb.shape}")
    print("Embedding 测试通过\n")

    # 测试 RMSNorm
    print("测试 RMSNorm 层")
    norm = RMSNorm(D_MODEL, device=device, dtype=dtype)
    x_norm = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)
    out_norm = norm(x_norm)
    print(f"输入形状: {x_norm.shape}")
    print(f"输出形状: {out_norm.shape}")
    print("RMSNorm 测试通过\n")

    # 测试 SwiGLU
    print("测试 SwiGLU 前馈层")
    swiglu = SwiGLU(D_MODEL, D_FF, device, dtype)
    x_swiglu = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)
    out_swiglu = swiglu(x_swiglu)
    print(f"输入形状: {x_swiglu.shape}")
    print(f"输出形状: {out_swiglu.shape}")
    print("SwiGLU 测试通过\n")

    # 测试 RoPE
    print("测试 RoPE 位置编码")
    d_k = D_MODEL // NUM_HEADS
    rope = RoPE(ROPE_THETA, d_k=d_k, device=device)
    x_rope = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, d_k, device=device)
    pos = torch.arange(SEQ_LEN, device=device)
    out_rope = rope(x_rope, pos)
    print(f"输入形状: {x_rope.shape}")
    print(f"输出形状: {out_rope.shape}")
    print("RoPE 测试通过\n")

    # 测试 MultiHeadSelfAttention
    print("测试 多头自注意力")
    rope_mha = RoPE(ROPE_THETA, d_k, device=device)
    mha = MultiHeadSelfAttention(D_MODEL, NUM_HEADS, rope_mha, device, dtype)
    x_mha = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)
    pos_mha = torch.arange(SEQ_LEN, device=device)
    out_mha = mha(x_mha, pos_mha)
    print(f"输入形状: {x_mha.shape}")
    print(f"输出形状: {out_mha.shape}")
    print("MHA 测试通过\n")

    # 测试 TransformerBlock
    print("测试 Transformer 块")
    block = TransformerBlock(D_MODEL, NUM_HEADS, D_FF, MAX_SEQ, ROPE_THETA, device, dtype)
    x_block = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)
    out_block = block(x_block)
    print(f"输入形状: {x_block.shape}")
    print(f"输出形状: {out_block.shape}")
    print("TransformerBlock 测试通过\n")

    # 测试 完整 TransformerLM
    print("测试 完整 Transformer 语言模型")
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=MAX_SEQ,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        device=device
    ).to(device)

    test_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)
    with torch.no_grad():
        logits = model(test_ids)

    print(f"输入 token 形状: {test_ids.shape}")
    print(f"输出 logits 形状: {logits.shape}")
