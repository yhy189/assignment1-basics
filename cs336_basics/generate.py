# 导入PyTorch核心库（深度学习框架，用于张量计算和模型推理）
import torch
# 从自定义module模块导入Transformer语言模型类、softmax激活函数
from module import TransformerLM, softmax
# 从自定义tokenizer模块导入分词器类（用于文本<->token ID的转换）
from tokenizer import Tokenizer


def decode(
        tokenizer: Tokenizer,  # 分词器实例：负责文本和token ID的双向转换
        lm: TransformerLM,  # 预训练的Transformer语言模型实例：核心生成模型
        prompt: str,  # 生成文本的起始提示语（如"Once upon a time"）
        stop_token: str,  # 停止生成的标记（如<|endoftext|>），遇到则终止生成
        context_length: int,  # 生成文本的最大上下文长度（token总数上限）
        temperature: float,  # 温度系数：控制生成随机性（值越大越随机，越小越确定）
        top_p: float,  # Top-P（核采样）阈值：控制采样的token范围（0~1）
        device: str | torch.device,  # 模型推理的设备（如"cpu"、"cuda:0"）
) -> str:  # 函数返回值：生成的完整文本字符串
    """
    核心生成函数：基于提示语和语言模型，生成符合要求的文本
    """
    # ===================== 第一步：预处理提示语，转换为模型可识别的token ID =====================
    # 1. 将提示语文本编码为token ID列表（分词器的核心功能：文本→数字序列）
    input_ids = tokenizer.encode(prompt)
    # 2. 转换为PyTorch张量：
    # - dtype=torch.long：token ID是整数，需用长整型
    # - device=device：将张量放到指定设备（CPU/GPU）上
    # - unsqueeze(0)：增加batch维度（模型输入要求[batch_size, seq_len]，这里batch_size=1）
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    # 3. 获取停止标记对应的token ID（仅取第一个token，因为stop_token通常是单个特殊标记）
    stop_token_id = tokenizer.encode(stop_token)[0]

    # ===================== 第二步：逐token生成文本（核心生成循环） =====================
    # torch.no_grad()：禁用梯度计算（推理阶段不需要反向传播，节省内存、加速计算）
    with torch.no_grad():
        # 生成循环：直到token总数达到context_length才停止（初始input_ids是提示语的token数）
        # range(context_length - input_ids.shape[1])：计算还能生成的token数量
        for _ in range(context_length - input_ids.shape[1]):#todo最大长度减去已有的长度#三维张量（对应[批次大小, 序列长度, 向量维度]）
            # 1. 模型前向推理：输入当前token序列，输出logits（未归一化的概率）
            # logits形状：[batch_size, seq_len, vocab_size] → 这里是[1, 当前序列长度, 词表大小]
            logits = lm(input_ids)

            # 2. 温度调节：只取最后一个token的logits（因为生成是逐token的，只预测下一个token）
            # logits[:, -1, :]：取batch维度全部、序列最后一个位置、所有词表维度
            # / temperature：温度越高，logits的差异越小，采样越随机；温度越低，差异越大，采样越确定
            logits = logits[:, -1, :] / temperature

            # 3. 转换为概率分布：对logits做softmax，得到每个token的概率（和为1）
            # probs形状：[1, vocab_size]
            probs = softmax(logits, dim=-1)

            # 4. todo应用Top-P（核采样）：只保留概率累积和≤top_p的token，降低低概率token的采样可能
            if top_p < 1.0:  # 只有top_p<1时才启用核采样，=1时直接采样所有token
                # 4.1 对概率按降序排序，同时获取排序后的token ID
                # sorted_probs：降序排列的概率值；sorted_indices：对应token ID
                sorted_probs, sorted_indices = probs.sort(descending=True)

                # 4.2 计算累积概率：从最大概率开始累加，直到和>top_p
                cumulative_probs = sorted_probs.cumsum(dim=-1)

                # 4.3 找到累积概率超过top_p的token，标记为需要截断（设为0）
                # cutoff是布尔张量：True表示该token的累积概率>top_p，需要舍弃
                cutoff = cumulative_probs > top_p

                # 4.4 强制保留第一个token（避免所有token都被截断，导致无token可采样）
                cutoff[:, 0] = False

                # 4.5 将截断的token概率置为0
                sorted_probs[cutoff] = 0

                # 4.6 重新归一化概率：确保剩余token的概率和为1
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

                # 4.7 从归一化后的概率中采样一个token ID
                # torch.multinomial：按概率采样，返回采样位置；gather：根据位置取对应的token ID
                next_token = sorted_indices.gather(
                    -1, torch.multinomial(sorted_probs, 1)
                )
            else:
                # 不启用Top-P时，直接从原始概率分布中采样下一个token
                next_token = torch.multinomial(probs, 1)

            # 5. 停止条件判断：如果采样到停止标记的token ID，终止生成循环
            if next_token.item() == stop_token_id:
                break

            # 6. 将新生成的token拼接到输入序列后，作为下一次推理的输入
            # input_ids形状从[1, n]变为[1, n+1]
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # ===================== 第三步：解码token ID为自然语言文本 =====================
    # 1. input_ids[0]：取出batch维度的第一个元素（因为batch_size=1）
    # 2. cpu().tolist()：将张量从GPU移到CPU，再转换为Python列表（方便分词器解码）
    # 3. tokenizer.decode：将token ID列表转换为文本字符串
    output = tokenizer.decode(input_ids[0].cpu().tolist())

    # 返回生成的完整文本
    return output


if __name__ == "__main__":
    # 主函数：加载分词器、模型，调用decode函数生成文本并输出

    # 导入os模块（此处未实际使用，可能是预留的路径处理）
    import os

    # 从自定义train模块导入TinyStories数据集的模型配置类/字典
    from train import TinyStoriesConfig

    # ===================== 1. 加载分词器 =====================
    # Tokenizer.from_files：从指定文件加载预训练的BPE分词器（词汇表+合并规则）
    # - bpe_vocab.pkl：BPE词汇表文件（token→ID映射）
    # - bpe_merges.pkl：BPE合并规则文件（如何将字符合并为token）
    # - special_tokens：特殊标记列表（如<|endoftext|>）
    tokenizer = Tokenizer.from_files(
        "../data/TinyStoriesV2-GPT4-train/bpe_vocab.pkl",
        "../data/TinyStoriesV2-GPT4-train/bpe_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    # ===================== 2. 初始化并加载预训练的Transformer语言模型 =====================
    # 加载模型配置（包含词汇表大小、上下文长度、模型维度等关键参数）
    config = TinyStoriesConfig

    # 实例化Transformer语言模型
    lm = TransformerLM(
        vocab_size=config["vocab_size"],  # 词表大小（token总数）
        context_length=config["context_length"],  # 模型支持的最大上下文长度
        d_model=config["d_model"],  # 模型隐藏层维度（核心维度）
        num_layers=config["num_layers"],  # Transformer解码器层数
        num_heads=config["num_heads"],  # 多头注意力的头数
        d_ff=config["d_ff"],  # 前馈网络的中间层维度
        rope_theta=config["rope_theta"],  # RoPE位置编码的theta参数
        device=config["device"],  # 模型运行设备
        dtype=config["dtype"],  # 模型参数的数据类型（如torch.float16）
    )

    # 加载模型检查点（预训练权重文件）
    checkpoint = torch.load(
        "../data/checkpoints/tiny_stories/checkpoint_final.pt",  # 检查点文件路径
        map_location=config["device"],  # 加载时直接映射到指定设备
    )

    # 将预训练权重加载到模型中
    lm.load_state_dict(checkpoint["model"])

    # 将模型设置为评估模式：禁用Dropout、LayerNorm等训练特有的层行为，确保推理稳定
    lm.eval()

    # ===================== 3. 配置生成参数并调用decode函数 =====================
    # 生成的起始提示语
    prompt = "Once upon a time"
    # 停止生成的标记
    stop_token = "<|endoftext|>"

    # 调用decode函数生成文本
    generated_text = decode(
        tokenizer,  # 分词器实例
        lm,  # 语言模型实例
        prompt,  # 起始提示语
        stop_token,  # 停止标记
        context_length=200,  # 最大生成长度（token数）
        temperature=1.0,  # 温度系数（1.0为原始概率）
        top_p=0.9,  # Top-P阈值（保留90%累积概率的token）
        device=config["device"],  # 推理设备
    )

    # 打印生成的文本
    print(generated_text)
    #todo done