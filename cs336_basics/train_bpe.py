from collections import Counter
import re
from typing import Dict, List, Tuple


# ===================== 1. 简化版预分词函数 =====================
# 功能：将文本拆分为基础单元（字母/数字/符号），统计词频
def simple_pre_tokenize(text: str) -> Counter[str]:
    """
    仅按Unicode字符拆分（无特殊标记处理），统计每个基础单元的出现次数
    """
    # 简单正则：匹配所有Unicode字符（单个字符拆分，便于演示BPE合并）
    pattern = r"\p{L}|\p{N}|[^\s\p{L}\p{N}]"
    tokens = re.findall(pattern, text)
    return Counter(tokens)


# ===================== 2. 核心BPE训练函数（简化版） =====================
def simple_train_bpe(
        text: str,  # 训练文本（直接传入字符串，无需文件）
        vocab_size: int  # 目标词表大小（包含256基础字节）
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    简化版BPE训练流程：
    1. 预分词并统计词频
    2. 初始化词表（256个基础字节）
    3. 迭代合并最频繁的字节对，直到词表达标
    """
    # 步骤1：预分词，获取基础词频
    word_counts = simple_pre_tokenize(text)
    if not word_counts:
        raise ValueError("输入文本不能为空")

    # 步骤2：初始化词表（仅包含0-255基础字节，无特殊标记）
    vocab: List[bytes] = [bytes([i]) for i in range(256)]
    merges: List[Tuple[bytes, bytes]] = []  # 存储合并规则

    # 步骤3：将单词转换为字节列表（BPE初始状态）
    # 格式：{单词字符串: {"tokens": 字节列表, "count": 词频}}
    words = {}
    for word_str, count in word_counts.items():
        # 字符串转字节列表（如"h" → [b'h'], "hello" → [b'h', b'e', b'l', b'l', b'o']）
        tokens = [bytes([ord(c)]) for c in word_str]
        words[word_str] = {"tokens": tokens, "count": count}

    # 步骤4：BPE核心合并循环
    while len(vocab) < vocab_size:
        # 子步骤4.1：统计所有相邻字节对的频率
        pair_counts = Counter[Tuple[bytes, bytes]]()
        for word in words.values():
            tokens = word["tokens"]
            count = word["count"]
            # 遍历单词的相邻字节对，累加频率
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += count

        # 无可用合并对，提前终止
        if not pair_counts:
            print(f"无更多可合并的字节对，当前词表大小：{len(vocab)}")
            break

        # 子步骤4.2：找到频率最高的字节对（核心：选最频繁的）
        most_common_pair = pair_counts.most_common(1)[0][0]
        left, right = most_common_pair

        # 子步骤4.3：合并字节对，生成新token
        new_token = left + right
        vocab.append(new_token)
        merges.append((left, right))

        # 子步骤4.4：更新所有单词的字节列表（替换合并对）
        for word in words.values():
            tokens = word["tokens"]
            new_tokens = []
            i = 0
            # 遍历字节列表，替换合并对
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == most_common_pair:
                    new_tokens.append(new_token)
                    i += 2  # 跳过已合并的右字节
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            word["tokens"] = new_tokens

    # 步骤5：转换词表为{ID: 字节}的字典
    vocab_dict = {i: tok for i, tok in enumerate(vocab)}
    return vocab_dict, merges


# ===================== 3. 测试主函数 =====================
if __name__ == "__main__":
    # 测试用小文本（便于观察合并过程）
    test_text = "hello hello world world world"

    # 训练BPE（目标词表大小=256+3，即合并3次）
    vocab, merges = simple_train_bpe(test_text, vocab_size=259)

    # 打印结果
    print("===== 最终词表（新增的3个合并token） =====")
    # 只打印新增的合并token（前256个是基础字节，略过）
    for i in range(256, len(vocab)):
        print(f"Token ID {i}: {vocab[i]} (合并规则：{merges[i - 256]})")

    print("\n===== 合并规则 =====")
    for idx, (left, right) in enumerate(merges):
        print(f"第{idx + 1}次合并：{left} + {right} → {left + right}")