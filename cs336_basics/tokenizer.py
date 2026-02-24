from collections import Counter
import re
import pickle
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Type

# 改用增强版正则库（兼容Unicode，和专业版对齐），若没有可替换为标准re（需注意\p{L}等语法仅regex支持）
try:
    import regex as re
except ImportError:
    import re
    print("提示：未安装regex库，部分Unicode分段功能可能受限，请执行 pip install regex")


class BPETokenizer:
    """
    融合版BPE分词器：保留核心BPE逻辑，整合实用功能点
    核心功能：
    - 基础编码/解码（极简版核心逻辑）
    - 特殊标记处理（如<|endoftext|>）
    - 优化的文本分段正则（更合理的单词/数字/标点拆分）
    - 编码缓存（提升重复token编码速度）
    - 从pickle文件加载词表/合并规则
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],          # 词表：{token ID: 字节序列}
        merges: List[Tuple[bytes, bytes]],# 合并规则：[(左字节, 右字节), ...]
        special_tokens: Optional[List[str]] = None,  # 特殊标记（如["<|endoftext|>"]）
    ) -> None:
        # 基础配置（继承极简版）
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 反向映射（编码用）+ 合并规则优先级（继承极简版）
        self.token2id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge2rank: Dict[Tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }

        # ========== 整合专业版的实用功能：优化的正则分段 ==========
        # 核心正则：更合理的文本分段（支持缩写、多语言字母、数字、标点）
        self.token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_token_pattern = re.compile(self.token_pattern)

        # 特殊标记正则：按长度降序匹配（避免短标记覆盖长标记）
        if self.special_tokens:
            self.special_tokens_pattern = (
                "(" + "|".join([re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]) + ")"
            )
            self.compiled_special_tokens_pattern = re.compile(self.special_tokens_pattern)

    @classmethod
    def from_files(
        cls: Type["BPETokenizer"],
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """
        整合专业版功能：从pickle文件加载词表和合并规则（实用的文件加载能力）
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    # ========== 整合专业版功能：编码缓存（提升重复token编码速度） ==========
    @lru_cache(maxsize=16 * 1024)  # 缓存前16384个token的编码结果
    def encode_token(self, word: str) -> List[int]:
        """
        核心BPE编码逻辑（继承极简版，保持逻辑不变）
        对单个文本片段执行BPE合并，返回token ID列表
        """
        # 文本→单个字节列表（初始状态）
        tokens = [bytes([i]) for i in word.encode("utf-8")]

        # 内部函数：找到优先级最高的可合并字节对
        def get_best_merge(tokens: List[bytes]) -> Optional[Tuple[bytes, bytes]]:
            min_rank = float("inf")
            best_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge2rank and self.merge2rank[pair] < min_rank:
                    min_rank = self.merge2rank[pair]
                    best_pair = pair
            return best_pair

        # 迭代合并（核心逻辑无改动）
        while True:
            merge_pair = get_best_merge(tokens)
            if merge_pair is None:
                break
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

        # 字节序列→token ID
        return [self.token2id[token] for token in tokens]

    def encode(self, text: str) -> List[int]:
        """
        完整编码逻辑（整合专业版：特殊标记处理 + 优化分段）
        """
        token_ids = []

        # 步骤1：处理特殊标记（拆分并保留特殊token）
        if self.special_tokens:
            for segment in self.compiled_special_tokens_pattern.split(text):
                # 特殊标记直接编码，不参与BPE合并
                if segment in self.special_tokens:
                    token_ids.append(self.token2id[segment.encode("utf-8")])
                    continue
                # 非特殊标记按正则分段后编码
                for match in self.compiled_token_pattern.finditer(segment):
                    word = match.group(0)
                    token_ids += self.encode_token(word)
        else:
            # 无特殊标记时，直接按正则分段编码
            for match in self.compiled_token_pattern.finditer(text):
                word = match.group(0)
                token_ids += self.encode_token(word)

        return token_ids

    def decode(self, ids: List[int]) -> str:
        """
        解码逻辑（继承极简版，保持不变）
        """
        bytes_list = [self.vocab[tid] for tid in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")


# ===================== 训练+测试代码（适配融合版分词器） =====================
if __name__ == "__main__":
    # 1. 简化版BPE训练函数（继承极简版，用于生成测试用的vocab/merges）
    def simple_pre_tokenize(text: str) -> Counter[str]:
        """预分词：和分词器的正则分段逻辑对齐"""
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = re.findall(pattern, text)
        return Counter(tokens)

    def simple_train_bpe(text: str, vocab_size: int) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        word_counts = simple_pre_tokenize(text)
        # 初始化256个基础字节
        vocab: List[bytes] = [bytes([i]) for i in range(256)]
        merges: List[Tuple[bytes, bytes]] = []
        # 初始化单词字典
        words = {}
        for word_str, count in word_counts.items():
            tokens = [bytes([i]) for i in word_str.encode("utf-8")]
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
            # 合并最频繁的字节对
            most_common_pair = pair_counts.most_common(1)[0][0]
            new_token = most_common_pair[0] + most_common_pair[1]
            vocab.append(new_token)
            merges.append(most_common_pair)
            # 更新单词的token列表
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

    # 2. 测试流程
    test_text = "hello hello world！世界123<|endoftext|>hello you'll"
    # 训练BPE（词表大小=256+5，合并5次）
    vocab, merges = simple_train_bpe(test_text, vocab_size=261)
    # 初始化融合版分词器（加入特殊标记）
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    # 3. 测试编码/解码
    encode_ids = tokenizer.encode(test_text)
    print("===== 编码结果 =====")
    print(f"原始文本：{test_text}")
    print(f"Token IDs：{encode_ids}")

    decode_text = tokenizer.decode(encode_ids)
    print("\n===== 解码结果 =====")
    print(f"解码文本：{decode_text}")
    print(f"核心内容是否一致：{test_text.replace(' ', '') == decode_text.replace(' ', '')}")

    # 4. 测试文件保存/加载（整合专业版的from_files功能）
    # 保存vocab和merges到pickle文件
    with open("test_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("test_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    # 从文件加载分词器
    tokenizer_from_file = BPETokenizer.from_files(
        "test_vocab.pkl", "test_merges.pkl", special_tokens=["<|endoftext|>"]
    )
    # 验证加载后的编码结果一致
    encode_ids_from_file = tokenizer_from_file.encode(test_text)
    print("\n===== 文件加载验证 =====")
    print(f"加载后编码结果是否一致：{encode_ids == encode_ids_from_file}")