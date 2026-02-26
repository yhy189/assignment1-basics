import math
from typing import Callable, Iterable, Optional

import torch
from jaxtyping import Float, Int  # 类型注解，明确张量维度含义


# ===================== 1. 数值稳定的交叉熵损失函数 =====================
def cross_entropy(
        pred_logits: Float[torch.Tensor, "batch_size vocab_size"],  # 模型输出logits：[批次大小, 词表大小]
        targets: Int[torch.Tensor, "batch_size"],  # 真实标签：[批次大小]（每个样本的目标token ID）
) -> Float[torch.Tensor, ""]:  # 返回批次平均损失（标量）
    """
    数值稳定的交叉熵损失实现（替代torch.nn.CrossEntropyLoss） #todo 交叉熵是对logit算还是对概率分布？
    核心公式（标准交叉熵）：CE = -log(exp(logit_y) / sum(exp(logit_i))) = log(sum(exp(logit_i))) - logit_y
    为避免exp(logit)溢出，先对logits做「减最大值」的数值稳定处理
    """
    # 步骤1：数值稳定化 → 所有logits减去批次内最大值（使最大值为0，exp后不会溢出）
    # x.shape = [batch_size, vocab_size]
    x = pred_logits - pred_logits.max(dim=-1, keepdim=True).values

    # 步骤2：计算交叉熵损失（逐样本）
    # x.exp().sum(dim=-1)：每个样本的exp(logit_i - max_logit)之和 → [batch_size]
    # .log()：取对数 → log(sum(exp(logit_i - max_logit)))
    # x[torch.arange(x.shape[0]), targets]：每个样本目标token对应的x值（即logit_y - max_logit）
    # 最终逐样本损失：log(sum(exp(x))) - x[targets] → 等价于标准交叉熵（推导见备注）
    x = x.exp().sum(dim=-1).log() - x[torch.arange(x.shape[0]), targets]

    # 步骤3：返回批次平均损失（标量）
    return torch.mean(x)


# 备注：数值稳定推导
# 令max_logit = pred_logits.max(dim=-1)
# sum(exp(logit_i)) = sum(exp(logit_i - max_logit) * exp(max_logit)) = exp(max_logit) * sum(exp(logit_i - max_logit))
# log(sum(exp(logit_i))) = max_logit + log(sum(exp(logit_i - max_logit)))
# 交叉熵 = [max_logit + log(sum(exp(x)))] - logit_y = log(sum(exp(x))) - (logit_y - max_logit) = log(sum(exp(x))) - x[targets]


# ===================== 2. AdamW优化器（带权重衰减的Adam） =====================
class AdamW(torch.optim.Optimizer):
    """
    手写版AdamW优化器（修正了传统Adam的权重衰减实现）
    核心区别：AdamW将「权重衰减」和「梯度更新」分开计算，而非合并到梯度中，是大模型训练的标准优化器
    数学公式：
    1. 一阶矩（动量）更新：m_t = β1*m_{t-1} + (1-β1)*g_t
    2. 二阶矩更新：v_t = β2*v_{t-1} + (1-β2)*g_t²
    3. 偏差校正：m_hat = m_t/(1-β1^t), v_hat = v_t/(1-β2^t)
    4. 参数更新：p_t = p_{t-1} - lr * (m_hat/(√v_hat + ε) + weight_decay * p_{t-1})
    """

    def __init__(
            self,
            params,  # 待优化的模型参数（如model.parameters()）
            lr=1e-3,  # 基础学习率
            weight_decay=1e-2,  # 权重衰减系数（L2正则，防止过拟合）
            betas=(0.9, 0.999),  # β1（一阶矩衰减）、β2（二阶矩衰减）
            eps=1e-8  # 防止分母为0的小值
    ):
        # 合法性检查：学习率不能为负
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 保存优化器超参数（供step函数使用）
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        # 调用父类构造函数（torch.optim.Optimizer）
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        """
        执行单次参数更新（核心方法）
        closure：可选的闭包函数，用于重新计算损失（如梯度累加时）
        """
        # 若传入closure，执行并获取损失值（通常用于梯度累加/重计算）
        loss = None if closure is None else closure()

        # 遍历所有参数组（支持不同参数用不同超参数，如分层学习率）
        for group in self.param_groups:
            # 提取当前参数组的超参数
            lr, weight_decay, betas, eps = (
                group["lr"],
                group["weight_decay"],
                group["betas"],
                group["eps"],
            )

            # 遍历当前参数组的每个参数
            for p in group["params"]:
                # 跳过无梯度的参数（如冻结的层）
                if p.grad is None:
                    continue

                # 获取参数的状态字典（保存动量m、二阶矩v、迭代次数t）
                state = self.state[p]
                # 初始化状态（首次更新时）：t=1，m/v为和参数同形状的零张量
                t = state.get("t", 1)  # 迭代次数（初始1）
                m = state.get("m", torch.zeros_like(p))  # 一阶矩（动量）
                v = state.get("v", torch.zeros_like(p))  # 二阶矩

                # 1. 获取参数的梯度（g_t）
                g = p.grad.data

                # 2. 更新一阶矩（动量）：m_t = β1*m_{t-1} + (1-β1)*g_t
                m = betas[0] * m + (1 - betas[0]) * g

                # 3. 更新二阶矩：v_t = β2*v_{t-1} + (1-β2)*g_t²（逐元素平方）
                v = betas[1] * v + (1 - betas[1]) * g ** 2

                # 4. 计算偏差校正后的学习率（合并到lr_t，避免单独计算m_hat/v_hat）
                # lr_t = lr * sqrt(1-β2^t)/(1-β1^t) → 等价于 lr * m_hat / m_t * v_t / sqrt(v_hat)
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                # 5. 梯度更新部分：p -= lr_t * m/(√v + ε)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                # 6. 权重衰减（AdamW核心：单独更新，而非合并到梯度）：p -= lr * weight_decay * p
                p.data -= lr * weight_decay * p.data

                # 7. 更新状态字典（供下一次迭代使用）
                state["t"] = t + 1  # 迭代次数+1
                state["m"] = m  # 保存新的一阶矩
                state["v"] = v  # 保存新的二阶矩

        # 返回损失值（若有）
        return loss


# ===================== 3. 余弦退火学习率调度器（带预热） =====================
def get_lr_cosine_schedule(
        t: int,  # 当前训练步数（迭代次数）
        lr_max: float,  # 最大学习率（预热后的峰值）
        lr_min: float,  # 最小学习率（衰减后的底值）
        T_w: int,  # 预热阶段步数（0~T_w线性上升）
        T_c: int  # 余弦衰减总步数（T_w~T_c余弦下降）
) -> float: #todo 余弦衰减学习率调度是怎么实现的
    """
    余弦退火学习率调度器（Cosine Annealing with Warmup）
    分三个阶段：
    1. 预热阶段（t < T_w）：学习率从0线性上升到lr_max
    2. 余弦衰减阶段（T_w ≤ t ≤ T_c）：学习率从lr_max余弦下降到lr_min
    3. 保持阶段（t > T_c）：学习率固定为lr_min
    """
    # 阶段1：预热 → 线性上升
    if t < T_w:
        return t / T_w * lr_max

    # 阶段2：余弦衰减 → 余弦函数平滑下降
    if t <= T_c:
        # 余弦衰减公式：lr_min + 0.5*(1 + cos(π*(t-T_w)/(T_c-T_w))) * (lr_max - lr_min)
        # 当t=T_w时，cos(0)=1 → lr=lr_max；当t=T_c时，cos(π)=-1 → lr=lr_min
        return lr_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (
                lr_max - lr_min
        )

    # 阶段3：保持最小值
    return lr_min


# ===================== 4. 梯度裁剪（Gradient Clipping） =====================
def gradient_clipping( #todo 梯度裁剪是怎么实现的
        parameters: Iterable[torch.nn.Parameter],  # 待裁剪的模型参数（如model.parameters()）
        max_l2_norm: float,  # 梯度的最大L2范数阈值（如1.0）
) -> None:
    """
    梯度裁剪：限制所有参数梯度的总L2范数，防止梯度爆炸（大模型训练必备）
    核心逻辑：若总梯度范数 > max_l2_norm，则所有梯度乘以 clip_coef = max_l2_norm / (总范数 + ε)
    """
    # 步骤1：计算所有参数梯度的总L2范数（逐参数计算平方和，再累加）
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            # 计算当前参数梯度的平方和，转换为Python数值（避免张量累积）
            total_norm += (p.grad.data ** 2).sum().item()
    # 开平方得到总L2范数
    total_norm = total_norm ** (1.0 / 2)

    # 步骤2：计算裁剪系数（若总范数≤阈值，系数=1，不裁剪；否则<1）
    clip_coef = max_l2_norm / (total_norm + 1e-6)  # +1e-6防止除零

    # 步骤3：执行裁剪（仅当需要裁剪时）
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                # 梯度乘以裁剪系数，缩小梯度值
                p.grad.data *= clip_coef