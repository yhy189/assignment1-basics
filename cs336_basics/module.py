import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils
from einops import einsum, rearrange  # 张量维度操作库，简化维度变换和矩阵乘法
from jaxtyping import Float, Int  # 类型注解，明确张量维度含义（提升代码可读性）


# ===================== 1. 自定义线性层（替代torch.nn.Linear） =====================
class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,  # 输入特征维度（如d_model）
            out_features: int,  # 输出特征维度（如d_ff或vocab_size）
            device: torch.device | None = None,  # 张量设备（CPU/GPU）
            dtype: torch.dtype | None = None,  # 张量数据类型（如float32/float16）
    ):
        """
        自定义线性变换层：y = x @ W（无偏置项，符合现代LLM设计）
        初始化采用「Xavier截断正态分布」，避免梯度消失/爆炸
        """
        super().__init__()
        # 定义可学习权重矩阵W：形状[in_features, out_features]，注册为nn.Parameter（可训练）
        self.W: Float[torch.Tensor, "in_features out_features"] = nn.Parameter(
            torch.empty((in_features, out_features), device=device, dtype=dtype)
        )
        # 计算初始化标准差：遵循「2/(in+out)」的Xavier初始化原则
        std = math.sqrt(2 / (in_features + out_features))
        # 截断正态分布初始化：均值0，标准差std，范围[-3*std, 3*std]（限制极端值）
        nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)

    def forward(
            self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        """
        前向传播：执行线性变换 x @ W
        - 输入x：任意形状，最后一维为in_features 输入的特征维度数 （如[batch, seq_len, d_model]）
        - 输出：与x形状一致（除最后一维变为out_features）
        - 用einops.einsum简化矩阵乘法，无需手动reshape
        乘法，只是指定输入维度和乘的矩阵维度还有输出维度
        """
        return einsum(x, self.W, "... d_in, d_in d_out -> ... d_out")
    #


# ===================== 2. 词嵌入层（Token Embedding） =====================
class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,  # 词表大小（token总数）
            embedding_dim: int,  # 嵌入向量维度（即d_model）
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        """
        词嵌入层：将token ID映射为固定维度的向量
        """
        super().__init__()
        # 嵌入矩阵：形状[num_embeddings, embedding_dim]，可训练参数
        self.embedding: Float[torch.Tensor, "num_embeddings embedding_dim"] = (
            nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
            )
        )
        # 截断正态分布初始化嵌入矩阵
        nn.init.trunc_normal_(self.embedding, 0, 1, -3, 3)

    def forward(
            self, token_ids: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        """
        前向传播：根据token ID查表获取嵌入向量
        - 输入token_ids：任意形状的整数张量（如[batch, seq_len]）
        - 输出：形状为[token_ids.shape, embedding_dim]（如[batch, seq_len, d_model]）
        """
        return self.embedding[token_ids]


# ===================== 3. RMSNorm归一化层（替代LayerNorm） =====================
class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,  # 模型维度（最后一维的大小）
            eps: float = 1e-5,  # 防止除零的小值
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        """
        RMSNorm：Root Mean Square Normalization
        相比LayerNorm，移除了均值中心化，仅保留均方根缩放，减少计算量且效果相当
        """
        super().__init__()
        self.eps = eps
        # 可学习的增益参数（scale）：形状[d_model]，初始化为1
        self.gain: Float[torch.Tensor, "d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(
            self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """
        前向传播：计算RMSNorm
        公式：output = (x / RMS(x)) * gain，其中RMS(x) = sqrt(mean(x²) + eps)
        """
        # 保存输入数据类型，计算完成后还原（避免精度损失）
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 步骤1：计算最后一维的平方均值（保持维度，便于广播）
        # torch.pow(x,2)：逐元素平方；mean(dim=-1, keepdim=True)：最后一维求均值
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        # 步骤2：归一化 + 增益缩放
        result = x / rms * self.gain

        # 还原数据类型并返回
        return result.to(in_dtype)


# ===================== 4. SiLU激活函数（Sigmoid Linear Unit） =====================
def silu(
        x: Float[torch.Tensor, "..."],
) -> Float[torch.Tensor, "..."]:
    """
    SiLU激活函数（也叫Swish）：x * sigmoid(x)
    相比ReLU，具有平滑的梯度，提升模型表达能力
    """
    return x * torch.sigmoid(x)


# ===================== 5. SwiGLU前馈网络（替代传统FFN） =====================
class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,  # 模型维度（输入/输出维度）
            d_ff: int,  # 前馈网络中间层维度（通常是d_model的4倍）
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_ff = d_ff
        # 三个线性层：W1/W3将d_model映射到d_ff，W2将d_ff映射回d_model
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(
            self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """
        SwiGLU前向传播：
        公式：W2( silu(W1(x)) * W3(x) )
        相比传统FFN（Linear→ReLU→Linear），提升了模型的非线性表达能力
        """
        return self.W2.forward(silu(self.W1.forward(x)) * self.W3.forward(x))


# ===================== 6. RoPE旋转位置编码（Rotary Position Embedding） =====================
class RoPE(nn.Module):
    # 注册为buffer（非可训练参数），存储预计算的cos/sin值
    cos: Float[torch.Tensor, "max_seq_len d_k_half"]
    sin: Float[torch.Tensor, "max_seq_len d_k_half"]

    def __init__(
            self,
            theta: float,  # RoPE的Θ参数（通常设为10000）
            d_k: int,  # 单个注意力头的query/key维度（d_model/num_heads）
            max_seq_len: int,  # 模型支持的最大序列长度
            device: torch.device | None = None,
    ) -> None:
        """
        RoPE：旋转位置编码
        核心思想：将位置信息编码到query/key的向量旋转中，保持相对位置不变性
        预计算所有位置的cos/sin值，避免前向时重复计算
        """
        super().__init__()
        # 步骤1：计算每个位置、每个维度的旋转角度θ
        # - torch.arange(max_seq_len)：位置索引 [0,1,...,max_seq_len-1]
        # - torch.pow(theta, -torch.arange(0, d_k, 2)/d_k)：维度缩放因子，形状[d_k/2]
        # - einsum广播：位置索引 × 维度因子 → [max_seq_len, d_k/2]
        thetas = einsum(
            torch.arange(max_seq_len),
            torch.pow(theta, -torch.arange(0, d_k, 2) / d_k),
            "index, theta -> index theta",
        )
        # 步骤2：预计算cos和sin值（形状[max_seq_len, d_k/2]）
        cos = torch.cos(thetas).to(device)
        sin = torch.sin(thetas).to(device)
        # 注册为buffer（非可训练，persistent=False表示不保存到checkpoint）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
            self,
            x: Float[torch.Tensor, "... seq_len d_k"],  # 输入（如query/key）：[..., seq_len, d_k]
            token_positions: Float[torch.Tensor, "... seq_len"],  # token位置：[..., seq_len]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        RoPE前向传播：对输入向量执行旋转操作
        核心逻辑：将向量按维度拆分为奇偶两部分，分别执行旋转计算
        """
        # 步骤1：拆分奇偶维度 → [batch, ..., seq_len, d_k/2]
        x_even = x[..., ::2]  # 取偶数索引维度（0,2,4...）
        x_odd = x[..., 1::2]  # 取奇数索引维度（1,3,5...）

        # 步骤2：根据token位置取对应的cos/sin值 → [batch, ..., seq_len, d_k/2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 步骤3：执行旋转计算
        result = torch.empty_like(x)  # 初始化输出张量（与输入形状一致）
        # 偶数维度：x_even*cos - x_odd*sin
        result[..., ::2] = x_even * cos - x_odd * sin
        # 奇数维度：x_even*sin + x_odd*cos
        result[..., 1::2] = x_even * sin + x_odd * cos

        return result


# ===================== 7. Softmax函数（自定义，数值稳定） =====================
def softmax(x: Float[torch.Tensor, " ..."], dim: int):
    """
    数值稳定的Softmax实现：先减去最大值（避免exp溢出），再计算exp和归一化
    """
    x -= x.max(dim, keepdim=True).values  # 最大值置0，防止exp(x)过大
    exp = x.exp()
    return exp / exp.sum(dim, keepdim=True)


# ===================== 8. 缩放点积注意力（Scaled Dot-Product Attention） =====================
def scaled_dot_product_attention(
        Q: Float[torch.Tensor, "batch_size ... queries d_k"],  # Query：[..., q_len, d_k]
        K: Float[torch.Tensor, "batch_size ... keys d_k"],  # Key：[..., k_len, d_k]
        V: Float[torch.Tensor, "batch_size ... values d_v"],  # Value：[..., k_len, d_v]
        mask: Float[torch.Tensor, "queries keys"] | None = None,  # 掩码：[q_len, k_len]
) -> Float[torch.Tensor, "batch_size ... queries d_v"]:
    """
    缩放点积注意力（Transformer核心）
    公式：Attention(Q,K,V) = softmax( (Q@K^T)/√d_k ) @ V
    带掩码（mask）：掩码位置设为-∞，Softmax后概率为0，实现因果/填充掩码
    """
    # 步骤1：计算Q和K的点积得分 → [batch, ..., q_len, k_len]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")

    # 步骤2：应用掩码（因果掩码/填充掩码）
    if mask is not None:
        # 掩码为False的位置设为-∞（Softmax后概率为0）
        scores = scores.masked_fill(mask == False, float("-inf"))

    # 步骤3：缩放（除以√d_k）→ Softmax → 与V相乘
    # K.shape[-1] = d_k，缩放避免点积值过大
    return einsum(
        softmax(scores / math.sqrt(K.shape[-1]), -1),  # Softmax后的注意力权重
        V,  # Value矩阵
        "... queries keys, ... keys d_v -> ... queries d_v",
    )


# ===================== 9. 多头自注意力（Multi-Head Self-Attention） =====================
class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,  # 模型总维度
            num_heads: int,  # 注意力头数（需整除d_model）
            pos_encoder: nn.Module | None = None,  # 位置编码器（如RoPE）
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.pos_encoder = pos_encoder  # 位置编码器（RoPE）

        # 四个线性层：将d_model映射到d_model（拆分后每个头为d_model/num_heads）
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)  # Q投影
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)  # K投影
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)  # V投影
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)  # 输出投影

    def forward(
            self,
            x: Float[torch.Tensor, "batch_size ... seq_len d_model"],  # 输入：[batch, seq_len, d_model]
            token_positions: Int[torch.Tensor, " ... seq_len"] | None = None,  # token位置
    ) -> Float[torch.Tensor, "batch_size ... seq_len d_model"]:
        """
        多头自注意力前向传播：
        1. 线性投影Q/K/V → 2. 拆分多头 → 3. 位置编码（RoPE） → 4. 缩放点积注意力 → 5. 合并多头 → 6. 输出投影
        """
        seq_len = x.shape[-2]

        # 步骤1：线性投影Q/K/V → [batch, seq_len, d_model]
        Q: Float[torch.Tensor, "batch_size ... seq_len h*d_q"] = self.W_q(x)
        K: Float[torch.Tensor, "batch_size ... seq_len h*d_k"] = self.W_k(x)
        V: Float[torch.Tensor, "batch_size ... seq_len h*d_v"] = self.W_v(x)

        # 步骤2：拆分多头 → [batch, ..., num_heads, seq_len, d_k]（d_k = d_model/num_heads）
        Qh: Float[torch.Tensor, "batch_size ... h seq_len d_q"] = rearrange(
            Q, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )
        Kh: Float[torch.Tensor, "batch_size ... h seq_len d_k"] = rearrange(
            K, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )
        Vh: Float[torch.Tensor, "batch_size ... h seq_len d_v"] = rearrange(
            V, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )

        # 步骤3：应用位置编码（RoPE）到Q/K
        if self.pos_encoder is not None and token_positions is not None:
            Qh = self.pos_encoder(Qh, token_positions)
            Kh = self.pos_encoder(Kh, token_positions)

        # 步骤4：创建因果掩码（Causal Mask）→ [seq_len, seq_len]
        # 目的：确保每个token只能关注自身及之前的token（自回归生成）
        # torch.triu：上三角矩阵（diagonal=1表示对角线以上为1），取反后下三角为True
        mask = (
                torch.triu(
                    torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
                    diagonal=1,
                )
                == 0
        )

        # 步骤5：执行缩放点积注意力 → [batch, ..., num_heads, seq_len, d_v]
        attn_out = scaled_dot_product_attention(Qh, Kh, Vh, mask)

        # 步骤6：合并多头 → [batch, seq_len, d_model]
        out: Float[torch.Tensor, "batch_size ... seq_len h*d_v"] = rearrange(
            attn_out, "... h seq_len d -> ... seq_len (h d)", h=self.num_heads
        )

        # 步骤7：输出线性投影 → 返回最终注意力结果
        return self.W_o(out)


# ===================== 10. Transformer块（单个Decoder层） =====================
class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,  # 模型维度
            num_heads: int,  # 注意力头数
            d_ff: int,  # 前馈网络中间维度
            max_seq_len: int,  # 最大序列长度
            rope_theta: float,  # RoPE的Θ参数
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # 初始化RoPE位置编码器（单个头的维度为d_model/num_heads）
        self.rope = RoPE(rope_theta, d_model // num_heads, max_seq_len, device)
        # 初始化SwiGLU前馈网络
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)
        # 初始化多头自注意力（绑定RoPE）
        self.mha = MultiHeadSelfAttention(
            d_model, num_heads, pos_encoder=self.rope, device=device, dtype=dtype
        )
        # 两个RMSNorm归一化层（Pre-Norm架构：先归一化，再注意力/前馈）
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        # 预计算token位置索引 → [max_seq_len]
        self.token_positions: Int[torch.Tensor, "max_seq_len"] = torch.arange(
            max_seq_len, device=device
        )

    def forward(
            self,
            x: Float[torch.Tensor, "batch_size seq_len d_model"],
    ) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        """
        TransformerBlock前向传播（Pre-Norm架构）：
        1. 自注意力残差块：x = x + MHA(RMSNorm(x))
        2. 前馈网络残差块：x = x + SwiGLU(RMSNorm(x))
        """
        # ---------------------- 多头注意力残差块 ----------------------
        x_residual = x  # 保存残差
        x = self.norm1(x)  # Pre-Norm：先归一化
        x = self.mha(x, self.token_positions[: x.shape[-2]])  # 多头自注意力（取当前序列长度的位置）
        x = x + x_residual  # 残差连接

        # ---------------------- 前馈网络残差块 ----------------------
        x_residual = x  # 保存残差
        x = self.norm2(x)  # Pre-Norm：先归一化
        x = self.swiglu(x)  # SwiGLU前馈网络
        x = x + x_residual  # 残差连接

        return x


# ===================== 11. 完整的Transformer语言模型（Decoder-only） =====================
class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,  # 词表大小
            context_length: int,  # 上下文长度（最大序列长度）
            d_model: int,  # 模型维度
            num_layers: int,  # TransformerBlock层数
            num_heads: int,  # 注意力头数
            d_ff: int,  # 前馈网络中间维度
            rope_theta: float,  # RoPE的Θ参数
            activation_checkpointing: bool = False,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # 堆叠多个TransformerBlock（nn.ModuleList管理）
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta, device, dtype
                )
                for _ in range(num_layers)
            ]
        )
        # 词嵌入层（token ID → d_model向量）
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        # 最终归一化层
        self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        # 输出投影层（d_model → vocab_size，预测每个token的得分）
        self.output_embedding = Linear(d_model, vocab_size, device, dtype)
        self.activation_checkpointing = activation_checkpointing

    def forward(
            self,
            token_ids: Int[torch.Tensor, "batch_size sequence_length"],  # 输入token ID：[batch, seq_len]
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:  # 输出logits：[batch, seq_len, vocab_size]
        """
        TransformerLM前向传播（完整流程）：
        token ID → 词嵌入 → 多层TransformerBlock → 最终归一化 → 输出投影 → logits
        """
        # 步骤1：词嵌入 → [batch, seq_len, d_model]
        x = self.token_embedding(token_ids)

        # 步骤2：逐层通过TransformerBlock
        for block in self.layers:
            if self.activation_checkpointing and self.training:
                x = checkpoint_utils.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # 步骤3：最终归一化
        x = self.norm_final(x)

        # 步骤4：输出投影到词表维度 → [batch, seq_len, vocab_size]
        x = self.output_embedding(x)

        return x
