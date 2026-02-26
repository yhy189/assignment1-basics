import os
import typing

import numpy as np
import numpy.typing as npt
import torch
import wandb  # 训练监控工具：记录损失、学习率、步数等指标
from jaxtyping import Int
from module import TransformerLM  # 自定义的Transformer语言模型
from optimizer import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping  # 自定义优化组件


# ===================== 1. 训练批次生成函数 =====================
def get_batch(
        x: Int[npt.NDArray, "length"],  # 完整的训练token ID序列：[总token数]
        batch_size: int,  # 批次大小（每个批次的样本数）
        context_length: int,  # 上下文长度（每个样本的token数）
        device: str | torch.device,  # 张量设备（CPU/GPU）
) -> tuple[
    Int[torch.Tensor, "batch_size context_length"],  # 输入序列：[batch_size, context_length]
    Int[torch.Tensor, "batch_size context_length"],  # 目标序列：[batch_size, context_length]
]:
    """
    核心函数：从训练数据中随机采样生成一个训练批次
    逻辑：输入序列是x[i:i+context_length]，目标序列是x[i+1:i+context_length+1]（偏移1个token，自回归训练）
    """
    # 步骤1：随机采样batch_size个起始索引（避免越界，索引范围0 ~ x.shape[0]-context_length）
    indices = np.random.randint(0, x.shape[0] - context_length, size=(batch_size,))

    # 步骤2：生成输入序列（每个索引取context_length长度的token）
    inputs = np.stack([x[i: i + context_length] for i in indices])
    # 步骤3：生成目标序列（输入序列向后偏移1个token，对应自回归的"下一个token预测"）
    targets = np.stack([x[i + 1: i + context_length + 1] for i in indices])

    # 步骤4：转换为PyTorch张量（long类型，放到指定设备）
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor


# ===================== 2. Checkpoint保存函数 =====================
def save_checkpoint(
        model: torch.nn.Module,  # 待保存的模型
        optimizer: torch.optim.Optimizer,  # 待保存的优化器（保留训练状态：动量、学习率等）
        iteration: int,  # 当前训练步数（用于断点续训）
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],  # 保存路径/文件句柄
) -> None:
    """
    保存训练 checkpoint：包含模型权重、优化器状态、当前步数，支持断点续训
    """
    # 构建checkpoint字典（需保存的核心信息）
    checkpoint = {
        "model": model.state_dict(),  # 模型可学习参数（权重/偏置）
        "optimizer": optimizer.state_dict(),  # 优化器状态（动量m、二阶矩v、迭代次数等）
        "iteration": iteration,  # 当前训练步数（续训时从该步数开始）
    }
    # 保存checkpoint到指定路径（PyTorch标准方式）
    torch.save(checkpoint, out)


# ===================== 3. Checkpoint加载函数 =====================
def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],  # checkpoint路径/文件句柄
        model: torch.nn.Module,  # 待加载权重的模型
        optimizer: torch.optim.Optimizer,  # 待加载状态的优化器
) -> int:
    """
    加载训练 checkpoint，恢复模型和优化器状态，返回续训的起始步数
    """
    # 加载checkpoint文件
    checkpoint = torch.load(src)
    # 恢复模型权重
    model.load_state_dict(checkpoint["model"])
    # 恢复优化器状态（保证续训时优化器参数和训练前一致）
    optimizer.load_state_dict(checkpoint["optimizer"])
    # 获取续训的起始步数
    iteration = checkpoint["iteration"]
    return iteration


# ===================== 4. 训练配置类（TypedDict） =====================
class TrainConfig(typing.TypedDict):
    """
    训练配置字典：统一管理所有超参数，包含模型、优化器、数据、训练流程四类配置
    TypedDict保证类型安全，IDE可自动补全参数
    """
    # 基础设备/数据类型配置
    device: torch.device  # 训练设备（如cuda:0）
    dtype: torch.dtype  # 模型参数数据类型（如float32/float16）
    # Transformer LM模型配置
    vocab_size: int  # 词表大小
    context_length: int  # 上下文长度（模型支持的最大序列长度）
    d_model: int  # 模型隐藏层维度
    num_layers: int  # TransformerBlock层数
    num_heads: int  # 多头注意力头数
    d_ff: int  # 前馈网络中间层维度
    rope_theta: float  # RoPE位置编码的Θ参数
    # 优化器配置
    lr: float  # 最大学习率（预热后峰值）
    lr_min: float  # 最小学习率（余弦衰减后底值）
    weight_decay: float  # 权重衰减系数（L2正则）
    betas: tuple[float, float]  # AdamW的β1和β2（一阶/二阶矩衰减系数）
    eps: float  # AdamW的ε（防止除零）
    max_grad_norm: float  # 梯度裁剪的最大L2范数阈值
    # 数据配置
    token_ids_path: str | os.PathLike  # 预编码的token ID文件路径（npy格式）
    checkpoint_dir: str | os.PathLike  # checkpoint保存目录
    # 训练流程配置
    batch_size: int  # 批次大小
    total_tokens: int  # 训练的总token数（终止条件）
    validation_interval: int  # 验证间隔（每N步打印损失）
    checkpoint_interval: int  # checkpoint保存间隔（每N步保存一次）
    wandb_project: str  # WandB项目名（用于训练监控）
    wandb_name: str  # WandB实验名（区分不同训练任务）


# ===================== 5. TinyStories数据集专属配置 =====================
TinyStoriesConfig = TrainConfig(
    device=torch.device("cuda"),  # 使用GPU训练
    dtype=torch.float32,  # 单精度浮点（小模型无需混合精度）
    # Transformer LM配置（适配TinyStories小模型）
    vocab_size=10000,  # 词表大小10000
    context_length=256,  # 上下文长度256
    d_model=512,  # 模型维度512
    num_layers=4,  # 4层TransformerBlock
    num_heads=16,  # 16头注意力
    d_ff=1344,  # 前馈网络维度1344
    rope_theta=10000,  # RoPE的Θ=10000（标准值）
    # 优化器配置
    lr=3e-4,  # 最大学习率3e-4
    lr_min=3e-5,  # 最小学习率3e-5
    weight_decay=0.01,  # 权重衰减0.01
    betas=(0.9, 0.999),  # AdamW标准β值
    eps=1e-8,  # AdamW标准ε值
    max_grad_norm=1.0,  # 梯度裁剪阈值1.0
    # 数据配置
    token_ids_path="../data/TinyStoriesV2-GPT4-train/token_ids.npy",  # 预编码token文件
    checkpoint_dir="../data/checkpoints/tiny_stories",  # checkpoint保存目录
    # 训练流程配置
    batch_size=128,  # 批次大小128
    total_tokens=327_680_000,  # 总训练token数（3.2768亿）
    validation_interval=10,  # 每10步打印一次损失
    checkpoint_interval=1000,  # 每1000步保存一次checkpoint
    wandb_project="cs336",  # WandB项目名
    wandb_name="tiny_stories_h100",  # WandB实验名
)


# ===================== 6. 核心训练函数 =====================
def train(config: TrainConfig) -> None:
    """
    完整训练流程：初始化模型/优化器 → 加载数据 → 训练循环 → 保存最终checkpoint
    """
    # 步骤1：初始化Transformer语言模型
    lm = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=config["device"],
        dtype=config["dtype"],
    )

    # 步骤2：初始化AdamW优化器
    adamw = AdamW(
        lm.parameters(),  # 待优化的模型参数
        lr=config["lr"],  # 初始学习率（后续由调度器更新）
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        eps=config["eps"],
    )

    # 步骤3：加载预编码的token ID数据（mmap_mode="r"：内存映射，避免加载全量数据到内存）
    # 适配GB级大文件：仅在访问时读取对应部分，节省内存
    token_ids = np.load(config["token_ids_path"], mmap_mode="r")

    # 步骤4：计算总训练步数（总token数 / 每个批次的token数）
    # 每个批次的token数 = batch_size * context_length
    total_steps = (
            config["total_tokens"] // config["batch_size"] // config["context_length"]
    )
    print(f"Total training steps: {total_steps}")  # 打印总步数，确认训练规模

    # 步骤5：初始化WandB（训练监控）
    wandb.init(
        project=config["wandb_project"],  # 项目名
        name=config["wandb_name"],  # 实验名
        config={**config, "total_steps": total_steps},  # 上传配置到WandB
    )

    # 步骤6：核心训练循环
    for step in range(total_steps):
        # ---------------------- 6.1 生成训练批次 ----------------------
        inputs, targets = get_batch(
            token_ids,
            batch_size=config["batch_size"],
            context_length=config["context_length"],
            device=config["device"],
        )  # inputs/targets形状：[batch_size, context_length]

        # ---------------------- 6.2 前向传播：计算logits和损失 ----------------------
        logits = lm(inputs)  # logits形状：[batch_size, context_length, vocab_size]
        # 损失计算：展平logits和targets（cross_entropy要求输入为[batch*seq_len, vocab_size]）
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # ---------------------- 6.3 反向传播：梯度计算与裁剪 ----------------------
        adamw.zero_grad()  # 清空上一步的梯度（避免累积）
        loss.backward()  # 反向传播计算梯度
        gradient_clipping(lm.parameters(), config["max_grad_norm"])  # 梯度裁剪，防止爆炸

        # ---------------------- 6.4 学习率调度：余弦退火+预热 ----------------------
        # 计算当前步数的学习率（预热步数=total_steps//10，总衰减步数=total_steps）
        lr = get_lr_cosine_schedule(
            step,
            lr_max=config["lr"],
            lr_min=config["lr_min"],
            T_w=total_steps // 10,  # 预热步数：总步数的10%
            T_c=total_steps,  # 余弦衰减总步数
        )
        # 更新优化器的学习率（所有参数组使用相同学习率）
        for param_group in adamw.param_groups:
            param_group["lr"] = lr

        # ---------------------- 6.5 参数更新 ----------------------
        adamw.step()  # 执行单次参数更新

        # ---------------------- 6.6 训练监控与日志 ----------------------
        # 1. 记录损失和学习率到WandB
        wandb.log({"loss": loss.item(), "lr": lr}, step=step)
        # 2. 定期打印损失（验证间隔）
        if (step + 1) % config["validation_interval"] == 0:
            print(f"Step {step + 1}: loss = {loss.item():.4f}")
        # 3. 定期保存checkpoint（checkpoint间隔）
        if (step + 1) % config["checkpoint_interval"] == 0:
            os.makedirs(config["checkpoint_dir"], exist_ok=True)  # 确保目录存在
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"checkpoint_step_{step + 1}.pt"
            )
            save_checkpoint(lm, adamw, step + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # 步骤7：训练结束，保存最终checkpoint
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"checkpoint_final.pt")
    save_checkpoint(
        lm,
        adamw,
        total_steps,
        checkpoint_path,
    )


# ===================== 7. 主函数：启动训练 =====================
if __name__ == "__main__":
    train(TinyStoriesConfig)  # 使用TinyStories配置启动训练