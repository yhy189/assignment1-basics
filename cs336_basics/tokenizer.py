from collections import Counter
import re
from typing import Dict, List, Tuple, Optional


# ===================== 极简版BPE分词器（仅核心功能） =====================
class SimpleBPETokenizer:
    """
    极简版BPE分词器：仅支持基础的文本编码/解码，无特殊标记、无缓存、无多进程
    完全匹配simple_train_bpe的输出格式（vocab/merges）
    """

    def __init__(
            self,
            vocab: Dict[int, bytes],  # 词表：{token ID: 字节序列}
            merges: List[Tuple[bytes, bytes]]  # 合并规则：[(左字节, 右字节), ...]
    ) -> None:
        # 基础配置
        self.vocab = vocab
        self.merges = merges
        # 反向映射：字节序列→token ID（编码时用）
        self.token2id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        # 合并规则索引：字节对→合并顺序（rank越小，优先级越高）
        self.merge2rank: Dict[Tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }
        # 简化版预分词正则：仅拆分单个Unicode字符（和训练代码对齐）
        self.compiled_pattern = re.compile(r"\p{L}|\p{N}|[^\s\p{L}\p{N}]")

    def encode_token(self, word: str) -> List[int]:
        """
        核心：对单个单词执行BPE编码（和训练代码的合并逻辑完全一致）
        """
        # 步骤1：文本→单个字节列表（初始状态）
        # 例如："hello" → [b'h', b'e', b'l', b'l', b'o']
        tokens = [bytes([ord(c)]) for c in word]

        # 步骤2：迭代合并最高优先级的字节对
        def get_best_merge(tokens: List[bytes]) -> Optional[Tuple[bytes, bytes]]:
            """找到当前token列表中优先级最高的可合并字节对"""
            min_rank = float("inf")
            best_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge2rank and self.merge2rank[pair] < min_rank:
                    min_rank = self.merge2rank[pair]
                    best_pair = pair
            return best_pair

        # 循环合并直到无可用合并对
        while True:
            merge_pair = get_best_merge(tokens)
            if merge_pair is None:
                break
            # 执行合并
            left, right = merge_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(left + right)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # 步骤3：字节序列→token ID
        return [self.token2id[token] for token in tokens]

    def encode(self, text: str) -> List[int]:
        """
        完整文本编码：预分词→逐词BPE编码→合并结果
        """
        # 步骤1：预分词（拆分单个字符，和训练代码对齐）
        word_list = self.compiled_pattern.findall(text)
        # 步骤2：逐词编码并合并结果
        token_ids = []
        for word in word_list:
            token_ids += self.encode_token(word)
        return token_ids

    def decode(self, ids: List[int]) -> str:
        """
        解码：token ID→字节序列→Unicode文本
        """
        # 步骤1：ID→字节序列
        bytes_list = [self.vocab[tid] for tid in ids]
        # 步骤2：拼接并解码（替换无效字节，避免报错）
        return b"".join(bytes_list).decode("utf-8", errors="replace")


# ===================== 测试：和简化版训练代码联动验证 =====================
if __name__ == "__main__":
    # ---------------------- 第一步：先运行简化版训练代码，得到vocab和merges ----------------------
    def simple_pre_tokenize(text: str) -> Counter[str]:
        pattern = r"\p{L}|\p{N}|[^\s\p{L}\p{N}]"
        tokens = re.findall(pattern, text)
        return Counter(tokens)


    def simple_train_bpe(text: str, vocab_size: int) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        word_counts = simple_pre_tokenize(text)
        # 初始化词表（256基础字节）
        vocab: List[bytes] = [bytes([i]) for i in range(256)]
        merges: List[Tuple[bytes, bytes]] = []
        # 初始化单词字典
        words = {}
        for word_str, count in word_counts.items():
            tokens = [bytes([ord(c)]) for c in word_str]
            words[word_str] = {"tokens": tokens, "count": count}
        # BPE合并循环
        while len(vocab) < vocab_size:
            # 统计字节对频率
            pair_counts = Counter[Tuple[bytes, bytes]]()
            for word in words.values():
                tokens = word["tokens"]
                count = word["count"]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += count
            if not pair_counts:
                break
            # 找最频繁的字节对
            most_common_pair = pair_counts.most_common(1)[0][0]
            left, right = most_common_pair
            # 合并并更新
            new_token = left + right
            vocab.append(new_token)
            merges.append((left, right))
            # 更新单词
            for word in words.values():
                tokens = word["tokens"]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == most_common_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word["tokens"] = new_tokens
        # 转换为字典
        vocab_dict = {i: tok for i, tok in enumerate(vocab)}
        return vocab_dict, merges


    # ---------------------- 第二步：训练+测试分词器 ----------------------
    # 测试文本（和训练代码一致）
    test_text = "hello hello world world world"
    # 训练BPE（目标词表大小=256+3，合并3次）
    vocab, merges = simple_train_bpe(test_text, vocab_size=259)
    # 初始化精简版分词器
    tokenizer = SimpleBPETokenizer(vocab, merges)

    # 测试编码
    encode_ids = tokenizer.encode(test_text)
    print("===== 编码结果 =====")
    print(f"原始文本：{test_text}")
    print(f"Token IDs：{encode_ids}")

    # 测试解码
    decode_text = tokenizer.decode(encode_ids)
    print("\n===== 解码结果 =====")
    print(f"解码文本：{decode_text}")
    print(f"编码解码是否一致：{test_text == decode_text}")

    # 打印合并规则和新增token（验证匹配）
    print("\n===== 合并规则与新增Token =====")
    for i in range(256, len(vocab)):
        print(f"Token ID {i}：{vocab[i]} → 合并规则：{merges[i - 256]}")