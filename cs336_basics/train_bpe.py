from collections import Counter
import os
import pickle
import regex as re  # 增强版正则，支持Unicode字符匹配
from typing import BinaryIO
import multiprocessing  # 多进程处理，提升大语料预分词效率


# ===================== 1. 语料分块函数：按特殊标记拆分大文件 =====================
def find_chunk_boundaries(
        file: BinaryIO,  # 二进制文件句柄（已打开的语料文件）
        desired_num_chunks: int,  # 期望的分块数量（通常等于进程数）
        split_special_token: bytes,  # 用于拆分的特殊标记字节（如b'<|endoftext|>'）
) -> list[int]:
    """
    核心作用：将超大文本文件按特殊标记拆分为多个独立块，保证每个块的完整性（不拆分文本语义）
    适配多进程处理：每个进程处理一个块，避免跨块的无效合并
    返回值：按字节偏移量排序的分块边界列表（如[0, 10000, 25000, ..., 文件大小]）
    """
    # 合法性检查：拆分标记必须是字节类型（文件以二进制读取）
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # 步骤1：获取文件总字节大小
    file.seek(0, os.SEEK_END)  # 指针移到文件末尾
    file_size = file.tell()  # 获取总字节数
    file.seek(0)  # 重置指针到开头

    # 步骤2：计算初始分块大小（均匀拆分）
    chunk_size = file_size // desired_num_chunks

    # 步骤3：初始化分块边界（均匀分布，最后一个边界为文件末尾）
    # 例如：desired_num_chunks=8 → [0, chunk_size, 2*chunk_size, ..., file_size]
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size  # 确保最后一个边界是文件末尾

    mini_chunk_size = 4096  # 每次向前读取4KB，寻找特殊标记（平衡效率和内存）

    # 步骤4：调整分块边界（确保边界落在特殊标记处，保证块的完整性）
    for bi in range(1, len(chunk_boundaries) - 1):  # 遍历除首尾外的所有初始边界
        initial_position = chunk_boundaries[bi]  # 当前边界的初始字节偏移
        file.seek(initial_position)  # 指针移到初始边界
        while True:
            # 读取4KB的迷你块（避免一次性读取大文件）
            mini_chunk = file.read(mini_chunk_size)

            # 若读到文件末尾：将当前边界设为文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在迷你块中查找特殊标记的位置
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 找到特殊标记：调整边界到特殊标记的起始位置
                chunk_boundaries[bi] = initial_position + found_at
                break
            # 未找到：继续向后读取下一个4KB
            initial_position += mini_chunk_size

    # 步骤5：去重并排序边界（可能返回少于期望数量的块，保证边界唯一）
    return sorted(set(chunk_boundaries))


# ===================== 2. 预分词函数：生成基础词频统计 =====================
def pre_tokenization(
        text: str,  # 待处理的文本片段（分块后的语料）
        special_tokens: list[str],  # 特殊标记列表（如["<|endoftext|>"]）
        pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> Counter[str]:
    """
    核心作用：
    1. 去除文本中的特殊标记（不参与BPE训练）；
    2. 按正则规则拆分文本为基础“单词”（预分词单元）；
    3. 统计每个“单词”的出现次数，为BPE合并提供初始词频。
    返回值：Counter字典（键=预分词单元，值=出现次数）
    """
    # 步骤1：构建特殊标记的正则（按长度降序，避免短标记覆盖长标记）
    special_tokens_pattern = "|".join(
        [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    )
    # 步骤2：初始化词频计数器
    word_counts = Counter[str]()

    # 步骤3：按特殊标记拆分文本，丢弃特殊标记本身
    for segment in re.split(special_tokens_pattern, text):
        # 步骤4：按正则规则拆分非特殊标记的文本片段为基础“单词”
        for match in re.finditer(pattern, segment):
            word = match.group(0)  # 提取单个预分词单元（如"hello"、","、"123"）
            word_counts[word] += 1  # 统计词频
    return word_counts


# ===================== 3. BPE核心训练函数 =====================
def train_bpe(
        input_path: str | os.PathLike[str],  # 训练语料文件路径
        vocab_size: int,  # 目标词表大小（包含特殊标记+256基础字节+合并token）
        special_tokens: list[str],  # 特殊标记列表（需加入词表，不参与合并）
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    BPE分词器核心训练流程：
    1. 初始化词表（特殊标记+256个基础字节）；
    2. 多进程预分词，统计全量语料的词频；
    3. 迭代合并最频繁的字节对，直到词表达到目标大小；
    4. 返回词表（token ID→字节）和合并规则（按合并顺序）。

    返回值：
        vocab: 词表字典 {token ID: 字节序列}
        merges: 合并规则列表 [(左字节, 右字节), ...]（按合并顺序排列）
    """
    # 步骤1：初始化词表
    # - 先加入所有特殊标记（编码为UTF-8字节）
    # - 再加入0-255的基础字节（BPE的最小单元）
    vocab: list[bytes] = [
        *[tok.encode("utf-8") for tok in special_tokens],
        *[bytes([i]) for i in range(256)],
    ]

    # 步骤2：初始化全量语料的词频计数器
    word_counts: Counter[str] = Counter()

    # ---------------------- 2.1 多进程预分词（处理大语料） ----------------------
    num_processes = 8  # 进程数（通常等于CPU核心数）
    with multiprocessing.Manager() as manager:
        # 进程间共享的结果列表（存储每个进程的预分词结果）
        results = manager.list()
        processes: list[multiprocessing.Process] = []

        # 打开语料文件（二进制模式，避免编码问题）
        with open(input_path, "rb") as f:
            # 按特殊标记拆分文件为多个块（保证块完整性）
            boundaries = find_chunk_boundaries(
                f, num_processes, special_tokens[0].encode("utf-8")
            )

            # 为每个块启动一个进程处理
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                f.seek(start)  # 指针移到块起始位置
                # 读取块内容并解码为UTF-8文本
                chunk = f.read(end - start).decode("utf-8")

                # 进程执行函数：对单个块执行预分词，结果存入共享列表
                def worker(idx, res_list, txt, sp_tokens):
                    try:
                        res = pre_tokenization(txt, sp_tokens)
                        res_list.append(res)
                    except Exception as e:
                        print(f"Error in process {idx}: {e}")

                # 启动子进程
                p = multiprocessing.Process(
                    target=worker, args=(i, results, chunk, special_tokens)
                )
                processes.append(p)
                p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 合并所有进程的预分词结果（累加词频）
        for res in results:
            word_counts += res

    # ---------------------- 2.2 初始化BPE合并所需的数据结构 ----------------------
    merges: list[tuple[bytes, bytes]] = []  # 存储合并规则（按合并顺序）
    # 单词字典：键=预分词单元（如"hello"），值=字典（包含tokens=字节列表、count=词频）
    words: dict[str, dict[str, str | list[bytes] | int]] = {}
    # 反向映射：字节对 → 包含该对的单词及次数（加速合并时的单词更新）
    pair2word: dict[tuple[bytes, bytes], Counter[str]] = {}
    # 字节对频率计数器：统计所有字节对的出现次数
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    # 遍历所有预分词单元，初始化单词字典和字节对计数器
    for word_str, count in word_counts.items():
        # 跳过已在初始词表中的token（如特殊标记，无需合并）
        if word_str.encode("utf-8") in vocab:
            continue
        # 将单词转换为基础字节列表（BPE的初始拆分状态）
        # 例如："hello" → b'hello' → [b'h', b'e', b'l', b'l', b'o']
        tokens = [bytes([i]) for i in word_str.encode("utf-8")]
        # 初始化单词字典项
        word = {
            "str": word_str,  # 原始单词字符串
            "tokens": tokens,  # 拆分后的字节列表
            "count": count,  # 词频
        }
        # 统计该单词中的所有相邻字节对，更新全局字节对频率
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
            # 反向映射：记录该字节对出现在哪个单词中，以及次数
            pair2word.setdefault(pair, Counter())[word_str] += 1
        # 将单词加入字典
        words[word_str] = word

    # ---------------------- 2.3 BPE核心合并循环 ----------------------
    # 迭代合并字节对，直到词表达到目标大小
    while len(vocab) < vocab_size:
        # 步骤1：找到当前最频繁的字节对（频率最高，频率相同时按字典序选更大的）
        most_common_pair: tuple[bytes, bytes] | None = None
        most_common_pair_count = 0
        for pair, count in pair_counts.items():
            # 优先级：1. 频率更高 2. 频率相同时字典序更大
            if (
                    not most_common_pair
                    or count > most_common_pair_count
                    or (count == most_common_pair_count and pair > most_common_pair)
            ):
                most_common_pair = pair
                most_common_pair_count = count

        # 无可用合并对（所有单词已拆分为最小单元），提前终止
        if most_common_pair is None:
            print(f"Ran out of pairs to merge at vocab size {len(vocab)} < {vocab_size}")
            break

        # 步骤2：执行合并，生成新token
        left_token, right_token = most_common_pair
        new_token = left_token + right_token  # 合并后的新字节序列
        vocab.append(new_token)  # 新token加入词表
        merges.append((left_token, right_token))  # 记录合并规则

        # 步骤3：更新所有包含该字节对的单词
        # 获取所有包含当前合并对的单词字符串
        word_strs = list(pair2word.get(most_common_pair, {}).keys())
        for word_str in word_strs:
            word = words[word_str]
            tokens: list[bytes] = word["tokens"]  # 单词当前的字节列表
            count: int = word["count"]  # 单词的词频

            # 子步骤3.1：移除该单词中所有旧字节对的频率统计
            for i in range(len(tokens) - 1):
                old_pair = (tokens[i], tokens[i + 1])
                # 全局字节对频率减去该单词的贡献
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] == 0:
                    pair_counts.pop(old_pair)  # 频率为0则删除
                # 更新反向映射：该单词不再包含该旧字节对
                pair2word[old_pair][word_str] -= 1
                if pair2word[old_pair][word_str] == 0:
                    del pair2word[old_pair][word_str]
                if not pair2word[old_pair]:  # 无单词包含该旧字节对，删除映射
                    pair2word.pop(old_pair)

            # 子步骤3.2：执行合并，生成新的字节列表
            new_tokens = []
            i = 0
            while i < len(tokens) - 1:
                # 匹配到当前合并对：替换为新token，跳过下一个字节
                if tokens[i] == left_token and tokens[i + 1] == right_token:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    # 未匹配：保留原字节，继续
                    new_tokens.append(tokens[i])
                    i += 1
            # 处理最后一个未遍历的字节（若有）
            new_tokens += tokens[i:]
            # 更新单词的字节列表
            word["tokens"] = new_tokens

            # 子步骤3.3：统计新字节列表中的相邻字节对，更新频率
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                pair_counts[new_pair] += count
                # 更新反向映射：该单词包含新字节对
                pair2word.setdefault(new_pair, Counter())[word_str] += 1

    # 步骤4：将词表转换为{token ID: 字节}的字典，返回结果
    return {i: tok for i, tok in enumerate(vocab)}, merges


# ===================== 4. 主函数：启动BPE训练并保存结果 =====================
if __name__ == "__main__":
    # 训练BPE分词器（适配TinyStories语料，目标词表大小10000，特殊标记<|endoftext|>）
    # vocab, merges = train_bpe("test.txt", 256 + 1 + 6, ["<|endoftext|>"])  # 测试用
    # vocab, merges = train_bpe("../tests/fixtures/corpus.en", 256 + 1 + 6, ["<|endoftext|>"])  # 测试用
    vocab, merges = train_bpe(
        "../data/TinyStoriesV2-GPT4-train.txt.txt",  # 训练语料路径
        10000,  # 目标词表大小
        ["<|endoftext|>"]  # 特殊标记
    )

    # 保存词表为文本文件（便于人工查看）
    with open("bpe_vocab.txt", "w", encoding="utf-8") as f:
        for i, tok in vocab.items():
            f.write(f"{i}\t{tok}\n")
    # 保存合并规则为文本文件（便于人工查看）
    with open("bpe_merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left} {right}\n")
    # 保存词表为二进制pickle文件（供分词器加载，高效）
    with open("bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    # 保存合并规则为二进制pickle文件（供分词器加载）
    with open("bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # 以下为调试代码（注释掉）
    # print("Vocab:")
    # for i, tok in vocab.items():
    #     print(f"  {i}: {tok}")
    # print("\nMerges:")
    # for left, right in merges:
    #     print(f"  {left} + {right} -> {left + right}")