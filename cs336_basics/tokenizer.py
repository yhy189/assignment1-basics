import functools
import pickle
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Type

import regex as re  # 增强版正则库，支持Unicode字符匹配（如\p{L}匹配所有语言的字母）


# ===================== 核心类：BPE分词器 =====================
class Tokenizer:
    """
    轻量级BPE（字节对编码）分词器，支持：
    - 文本→token ID编码 / token ID→文解码
    - 特殊标记（如<|endoftext|>）处理
    - 大文本的内存高效迭代编码
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],          # 词表：{token ID: 字节序列}（如{123: b'hello'}）
        merges: List[Tuple[bytes, bytes]],# BPE合并规则：按合并顺序存储的字节对列表（如[(b'h', b'e'), (b'he', b'l')]）
        special_tokens: Optional[List[str]] = None,  # 特殊标记列表（如["<|endoftext|>"]）
    ) -> None:
        """
        初始化分词器：构建反向映射、合并规则索引、正则表达式（用于文本分段）
        """
        # 保存基础配置
        self.vocab: Dict[int, bytes] = vocab
        self.merges: List[Tuple[bytes, bytes]] = merges
        self.special_tokens: List[str] = special_tokens or []

        # 构建反向映射：字节序列→token ID（编码时用）
        self.token2id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        # 构建合并规则索引：字节对→合并顺序（rank越小，合并优先级越高）
        self.merge2rank: Dict[Tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }

        # ===================== 正则表达式：文本分段规则 =====================
        # 核心正则：匹配文本中的「单词/数字/标点/空格」，保证分段合理性
        # 拆解：
        # - '(?:[sdmt]|ll|ve|re)'：匹配缩写（如it's、you'll、I've）
        # - ?\p{L}+：匹配任意语言的字母（带可选前导空格）
        # - ?\p{N}+：匹配数字（带可选前导空格）
        # - ?[^\s\p{L}\p{N}]+：匹配标点/符号（带可选前导空格）
        # - \s+(?!\S)：匹配末尾空格
        # - \s+：匹配其他空格
        self.token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_token_pattern = re.compile(self.token_pattern)

        # 特殊标记正则：匹配所有特殊token（按长度降序，避免短token匹配覆盖长token）
        if self.special_tokens:
            # 转义特殊字符（如|、<、>），用|拼接，包裹捕获组（split后保留特殊token）
            self.special_tokens_pattern = (
                "("
                + "|".join(
                    [
                        re.escape(f"{token}")
                        for token in sorted(self.special_tokens, key=len, reverse=True)
                    ]
                )
                + ")"
            )
        else:
            self.special_tokens_pattern = "(?!)"  # 匹配空的正则（无特殊token时）
        self.compiled_special_tokens_pattern = re.compile(self.special_tokens_pattern)

    @classmethod
    def from_files(
        cls: Type["Tokenizer"],
        vocab_filepath: str,             # 词表文件路径（pickle序列化的Dict[int, bytes]）
        merges_filepath: str,            # 合并规则文件路径（pickle序列化的List[Tuple[bytes, bytes]]）
        special_tokens: Optional[List[str]] = None,  # 特殊标记列表
    ) -> "Tokenizer":
        """
        类方法：从文件加载预训练的词表和合并规则，初始化分词器
        """
        # 加载词表（二进制读取，因为vocab值是bytes）
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        # 加载合并规则
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        # 实例化分词器
        return cls(vocab, merges, special_tokens)

    @functools.lru_cache(maxsize=16 * 1024)  # 缓存前16384个token的编码结果，提升重复token编码速度
    def encode_token(self, word: str) -> List[int]:
        """
        核心方法：对单个文本片段（如"hello"）执行BPE编码，返回token ID列表
        步骤：1. 文本→字节序列 2. 迭代合并最高优先级的字节对 3. 映射为token ID
        """
        # 步骤1：将文本转换为UTF-8字节序列，拆分为单个字节的列表（BPE的初始状态）
        # 例如："hello" → b'hello' → [b'h', b'e', b'l', b'l', b'o']
        tokens = [bytes([i]) for i in word.encode("utf-8")]

        # 内部函数：找到当前token列表中「优先级最高」的可合并字节对
        def get_merge(tokens: List[bytes]) -> Optional[Tuple[bytes, bytes]]:
            min_rank = float("inf")  # 最小rank=最高优先级（rank越小，合并越早）
            candidate: Optional[Tuple[bytes, bytes]] = None  # 待返回的最优合并对
            # 遍历所有相邻字节对
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                # 若该字节对在合并规则中，且rank更小（优先级更高）
                if pair in self.merge2rank and self.merge2rank[pair] < min_rank:
                    min_rank = self.merge2rank[pair]
                    candidate = pair
            return candidate  # 返回最优合并对（无则返回None）

        # 步骤2：迭代执行BPE合并，直到无可用合并对
        while True:
            merge = get_merge(tokens)  # 获取当前最优合并对
            if merge is None:  # 无合并对，终止循环
                break
            # 执行合并：遍历token列表，替换合并对为新token
            new_tokens: List[bytes] = []
            i = 0
            while i < len(tokens):
                # 匹配到合并对：替换为合并后的新token，i+2跳过下一个token
                if (
                    i < len(tokens) - 1
                    and tokens[i] == merge[0]
                    and tokens[i + 1] == merge[1]
                ):
                    new_tokens.append(merge[0] + merge[1])  # 合并字节对
                    i += 2
                else:
                    # 未匹配：保留原token，i+1
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens  # 更新token列表，进入下一轮合并

        # 步骤3：将合并后的字节序列映射为token ID（查token2id）
        return [self.token2id[token] for token in tokens]

    def encode(self, text: str) -> List[int]:
        """
        完整文本编码：处理特殊标记 + 正则分段 + 逐段BPE编码
        """
        token_ids = []  # 存储最终的token ID序列

        # 步骤1：按特殊标记拆分文本（保留特殊token在拆分结果中）
        # 例如："hello<|endoftext|>world" → ["hello", "<|endoftext|>", "world"]
        for segment in self.compiled_special_tokens_pattern.split(text):
            # 若分段是特殊token：直接编码（不执行BPE）
            if segment in self.special_tokens:
                # 特殊token→字节→token ID
                token_ids.append(self.token2id[segment.encode("utf-8")])
                continue
            # 步骤2：对非特殊token的分段，按token_pattern拆分（单词/数字/标点等）
            for match in self.compiled_token_pattern.finditer(segment):
                word = match.group(0)  # 提取单个文本片段（如"hello"、","、"123"）
                # 对该片段执行BPE编码，结果追加到token_ids
                token_ids += self.encode_token(word)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        内存高效的迭代式编码：对可迭代文本（如文件句柄）逐行编码，返回token ID生成器
        适配：无法一次性加载到内存的大文本文件（如GB级数据集）
        """
        for text in iterable:
            # 对每行文本编码，逐个yield token ID（避免一次性存储所有ID）
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        解码：将token ID序列转换回Unicode文本
        """
        # 步骤1：token ID→字节序列列表
        bytes_list = [self.vocab[token_id] for token_id in ids]
        # 步骤2：拼接字节序列→解码为文本（errors="replace"：替换无效字节，避免解码失败）
        return b"".join(bytes_list).decode("utf-8", errors="replace")


# ===================== 辅助函数：多进程编码相关 =====================
def _encode_with_text(t: str):
    """
    多进程编码的单任务函数：输入文本，返回（文本, token ID列表）
    用于进程池映射，方便更新进度条（通过文本长度计算处理进度）#todo 进程池映射？
    """
    return (t, tokenizer.encode(t))


def _accumulate_iter(iterable: Iterable[str], min_size: int) -> Iterator[str]:
    """
    文本累积函数：将可迭代文本（如逐行读取的文件）累积到至少min_size字符后返回
    作用：减少多进程通信次数（频繁传递小文本会降低效率）
    """
    batch = ""  # 累积的文本批次
    for text in iterable:
        batch += text
        # 累积到最小尺寸，返回批次并清空
        if len(batch) >= min_size:
            yield batch
            batch = ""
    # 处理最后一批（不足min_size的剩余文本）
    if batch:
        yield batch


def _init_worker(tok: Tokenizer):
    """
    进程池初始化函数：为每个子进程设置全局tokenizer
    原因：multiprocessing的进程间无法直接共享复杂对象，需单独初始化
    """
    global tokenizer
    tokenizer = tok


# ===================== 主函数：批量编码大文本文件 =====================
if __name__ == "__main__":
    import array
    import multiprocessing
    import os

    import numpy as np
    import tqdm  # 进度条库

    # 步骤1：初始化分词器（加载预训练的词表和合并规则）
    tokenizer = Tokenizer.from_files(
        "../data/owt_train/bpe_vocab.pkl",          # OpenWebText训练集的BPE词表
        "../data/owt_train/bpe_merges.pkl",         # OpenWebText训练集的BPE合并规则
        special_tokens=["<|endoftext|>"],           # 特殊标记：文本结束符
    )

    # 步骤2：初始化token ID缓冲区（array.array比list更节省内存，"H"=uint16，适配token ID范围）
    token_ids_buf = array.array("H")

    # 步骤3：处理大文本文件（OpenWebText训练集）
    file_path = "../data/owt_train.txt"
    with open(file_path, "r") as f:
        # 获取文件总字符数（用于进度条）
        f.seek(0, os.SEEK_END)
        bytes_len = f.tell()
        f.seek(0)  # 重置文件指针到开头

        # 初始化进度条（显示编码进度）
        with tqdm.tqdm(
            total=bytes_len,
            unit="char",
            desc="Encoding",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            # 步骤4：启动多进程池（8进程），初始化每个进程的tokenizer
            with multiprocessing.Pool(
                processes=8, initializer=_init_worker, initargs=(tokenizer,)
            ) as pool:
                # 步骤5：批量编码
                # 1. _accumulate_iter：将文本累积为128KB批次（减少进程通信）
                # 2. pool.imap：并行执行_encode_with_text，返回（文本, token ID）
                batch_ids = pool.imap(
                    _encode_with_text, _accumulate_iter(f, 128 * 1024)
                )
                # 遍历编码结果，更新缓冲区和进度条
                for text, ids in batch_ids:
                    token_ids_buf.extend(ids)  # 将token ID追加到缓冲区
                    # 更新进度条：按文本的UTF-8字节数（而非字符数）统计
                    pbar.update(len(text.encode("utf-8")))

    # 步骤6：将缓冲区转换为numpy数组（uint16，节省内存），保存为npy文件
    token_ids = np.frombuffer(token_ids_buf, dtype=np.uint16)
    np.save("token_ids.npy", token_ids)

    # 步骤7：计算压缩比（原始文本字节数 / token总数 → 每个token平均编码的字节数）
    print(f"Compression ratio: {bytes_len/(token_ids.size):.2f}")
    exit()

    # ===================== 测试代码（注释掉，用于验证编码/解码功能） =====================
    token_ids = tokenizer.encode("Hello, 世界！<|endoftext|>")
    print("Token IDs:")
    print(token_ids)
    print("Tokens:")
    print([tokenizer.vocab[token_id] for token_id in token_ids])
    text = tokenizer.decode(token_ids)
    print("Decoded text:")
    print(text)
    print("Decoded text:")
    print(text)