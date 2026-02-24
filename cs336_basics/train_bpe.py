from collections import Counter
import os
import pickle
import re
import multiprocessing
from typing import Dict, List, Tuple, BinaryIO, Optional, Union


# ===================== 1. 工具函数：大文件分块（适配多进程） =====================
def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes = b'<|endoftext|>'
) -> List[int]:
    """
    大文件分块：按特殊标记拆分文件，保证每个块的语义完整性（避免拆分文本）
    适配多进程：每个进程处理一个块，提升大语料预分词效率
    """
    # 获取文件总大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # 初始均匀分块
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size  # 最后一个块到文件末尾

    # 调整边界到特殊标记处（保证块完整性）
    mini_chunk_size = 4096  # 每次读取4KB，平衡效率和内存
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_pos = chunk_boundaries[bi]
        file.seek(initial_pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:  # 读到文件末尾
                chunk_boundaries[bi] = file_size
                break
            # 找到特殊标记位置，调整边界
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_pos + found_at
                break
            initial_pos += mini_chunk_size

    # 去重并排序，返回最终边界
    return sorted(set(chunk_boundaries))


# ===================== 2. 预分词函数：优化版（和分词器正则对齐） =====================
def pre_tokenization(
        text: str,
        special_tokens: List[str],
        pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
) -> Counter[str]:
    """
    预分词：
    1. 移除特殊标记（不参与BPE训练）
    2. 按优化正则拆分文本为基础单元（和分词器的分段逻辑完全对齐）
    3. 统计每个单元的词频
    """
    # 构建特殊标记正则（按长度降序，避免短标记覆盖长标记）
    special_pattern = "|".join([re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)])
    # 拆分并丢弃特殊标记
    segments = re.split(special_pattern, text)

    # 统计非特殊标记部分的词频
    word_counts = Counter()
    compiled_pattern = re.compile(pattern)
    for segment in segments:
        if segment.strip():  # 跳过空段
            tokens = compiled_pattern.findall(segment)
            word_counts.update(tokens)
    return word_counts


# ===================== 3. 核心BPE训练函数（融合版） =====================
def train_bpe(
        input_data: Union[str, os.PathLike[str]],  # 训练数据：文本字符串（小文本）或文件路径（大文本）
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
        num_processes: int = 8  # 多进程数（仅大文本生效）
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    融合版BPE训练函数：
    - 支持小文本（直接传字符串）和大文本（传文件路径）
    - 支持特殊标记（加入词表，不参与合并）
    - 小文本：单进程处理；大文本：多进程分块处理
    - 输出格式和BPETokenizer完全兼容
    """
    special_tokens = special_tokens or []
    # 步骤1：初始化词表（特殊标记 + 256个基础字节）
    vocab: List[bytes] = [
        *[tok.encode("utf-8") for tok in special_tokens],  # 特殊标记优先加入
        *[bytes([i]) for i in range(256)]  # 基础字节
    ]
    merges: List[Tuple[bytes, bytes]] = []

    # 步骤2：获取全量语料的词频统计
    word_counts = Counter[str]()

    # 分支1：小文本（直接传字符串）
    if isinstance(input_data, str) and not os.path.exists(input_data):
        word_counts = pre_tokenization(input_data, special_tokens)

    # 分支2：大文本（文件路径）→ 多进程分块处理
    else:
        with multiprocessing.Manager() as manager:
            results = manager.list()  # 进程间共享结果
            processes = []

            with open(input_data, "rb") as f:
                # 按特殊标记分块
                split_token = special_tokens[0].encode("utf-8") if special_tokens else b''
                boundaries = find_chunk_boundaries(f, num_processes, split_token)

                # 为每个块启动进程
                for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                    f.seek(start)
                    # 读取块并解码为文本
                    chunk = f.read(end - start).decode("utf-8", errors="replace")

                    # 进程执行函数
                    def worker(idx, res_list, txt, sp_toks):
                        try:
                            res = pre_tokenization(txt, sp_toks)
                            res_list.append(res)
                        except Exception as e:
                            print(f"进程 {idx} 出错：{e}")

                    p = multiprocessing.Process(target=worker, args=(i, results, chunk, special_tokens))
                    processes.append(p)
                    p.start()

                # 等待所有进程完成
                for p in processes:
                    p.join()

                # 合并所有进程的词频结果
                for res in results:
                    word_counts += res

    # 步骤3：初始化BPE合并所需数据
    # 单词字典：{单词字符串: {"tokens": 字节列表, "count": 词频}}
    words = {}
    for word_str, count in word_counts.items():
        # 跳过已在初始词表中的特殊标记
        word_bytes = word_str.encode("utf-8")
        if word_bytes in vocab:
            continue
        # 转换为基础字节列表（BPE初始状态）
        tokens = [bytes([i]) for i in word_bytes]
        words[word_str] = {"tokens": tokens, "count": count}

    # 步骤4：核心BPE合并循环（简化版，保留核心逻辑）
    while len(vocab) < vocab_size:
        # 子步骤4.1：统计所有字节对的频率
        pair_counts = Counter[Tuple[bytes, bytes]]()
        for word in words.values():
            tokens = word["tokens"]
            count = word["count"]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += count

        # 无可用合并对，提前终止
        if not pair_counts:
            print(f"无更多可合并字节对，当前词表大小：{len(vocab)} < 目标：{vocab_size}")
            break

        # 子步骤4.2：找到频率最高的字节对
        most_common_pair = pair_counts.most_common(1)[0][0]
        left, right = most_common_pair

        # 子步骤4.3：生成新token并更新词表/合并规则
        new_token = left + right
        vocab.append(new_token)
        merges.append((left, right))

        # 子步骤4.4：更新所有单词的字节列表
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

    # 步骤5：转换为{ID: 字节}的字典，返回结果
    vocab_dict = {i: tok for i, tok in enumerate(vocab)}
    return vocab_dict, merges


# ===================== 4. 测试与保存：适配BPETokenizer =====================
if __name__ == "__main__":
    # ---------------------- 测试1：小文本训练（快速验证） ----------------------
    print("===== 测试1：小文本训练 =====")
    test_text = "hello hello world！世界123<|endoftext|>hello you'll"
    # 训练（目标词表=特殊标记(1)+256基础字节+5次合并=262）
    vocab_small, merges_small = train_bpe(
        input_data=test_text,
        vocab_size=262,
        special_tokens=["<|endoftext|>"],
        num_processes=1  # 小文本用单进程
    )
    # 打印新增的合并token
    print("新增合并token（特殊标记+基础字节后）：")
    start_idx = len(["<|endoftext|>"]) + 256  # 特殊标记数+基础字节数
    for i in range(start_idx, len(vocab_small)):
        print(f"Token ID {i}: {vocab_small[i]} → 合并规则：{merges_small[i - start_idx]}")

    # ---------------------- 测试2：和BPETokenizer适配验证 ----------------------
    print("\n===== 测试2：分词器适配验证 =====")
    # 导入之前的BPETokenizer（确保代码在同一文件/可导入）
    try:
        from bpe_tokenizer import BPETokenizer  # 若在同一文件，直接用类名
    except ImportError:
        # 若不在同一文件，这里复制核心类（或确保路径正确）
        class BPETokenizer:  # 简化版，仅用于验证
            def __init__(self, vocab, merges, special_tokens=None):
                self.vocab = vocab
                self.token2id = {v: k for k, v in vocab.items()}
                self.merge2rank = {m: i for i, m in enumerate(merges)}
                self.special_tokens = special_tokens or []
                self.compiled_token_pattern = re.compile(
                    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
                if self.special_tokens:
                    self.special_tokens_pattern = "(" + "|".join(
                        [re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]) + ")"
                    self.compiled_special_tokens_pattern = re.compile(self.special_tokens_pattern)

            def encode(self, text):
                token_ids = []
                if self.special_tokens:
                    for seg in self.compiled_special_tokens_pattern.split(text):
                        if seg in self.special_tokens:
                            token_ids.append(self.token2id[seg.encode("utf-8")])
                            continue
                        for match in self.compiled_token_pattern.finditer(seg):
                            word = match.group(0)
                            token_ids += self._encode_token(word)
                return token_ids

            def _encode_token(self, word):
                tokens = [bytes([i]) for i in word.encode("utf-8")]
                while True:
                    best_pair = None
                    min_rank = float("inf")
                    for i in range(len(tokens) - 1):
                        pair = (tokens[i], tokens[i + 1])
                        if pair in self.merge2rank and self.merge2rank[pair] < min_rank:
                            min_rank = self.merge2rank[pair]
                            best_pair = pair
                    if best_pair is None:
                        break
                    left, right = best_pair
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
                return [self.token2id[t] for t in tokens]

            def decode(self, ids):
                bytes_list = [self.vocab[tid] for tid in ids]
                return b"".join(bytes_list).decode("utf-8", errors="replace")

    # 初始化分词器
    tokenizer = BPETokenizer(vocab_small, merges_small, special_tokens=["<|endoftext|>"])
    # 测试编码解码
    encode_ids = tokenizer.encode(test_text)
    decode_text = tokenizer.decode(encode_ids)
    print(f"原始文本：{test_text}")
    print(f"解码文本：{decode_text}")
    print(f"核心内容一致：{test_text.replace(' ', '') == decode_text.replace(' ', '')}")

    # ---------------------- 测试3：大文本训练（可选） ----------------------
    # 若有大语料文件，取消注释运行
    # print("\n===== 测试3：大文本训练 =====")
    # vocab_large, merges_large = train_bpe(
    #     input_data="../data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    #     num_processes=8
    # )
    # # 保存训练结果（供分词器加载）
    # with open("bpe_vocab.pkl", "wb") as f:
    #     pickle.dump(vocab_large, f)
    # with open("bpe_merges.pkl", "wb") as f:
    #     pickle.dump(merges_large, f)
    # print("大文本训练完成，已保存vocab和merges到pkl文件")