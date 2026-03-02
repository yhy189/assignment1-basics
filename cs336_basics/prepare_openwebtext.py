from __future__ import annotations

import argparse
import array
import os
from pathlib import Path

import numpy as np
import tiktoken


def _encode_text_file(
    input_path: Path,
    output_path: Path,
    encoding_name: str,
    max_bytes: int | None,
) -> None:
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = array.array("H")

    total_size = input_path.stat().st_size
    report_interval = 512 * 1024 * 1024
    next_report = report_interval

    seen_bytes = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.encode("utf-8")

            if max_bytes is not None and seen_bytes >= max_bytes:
                break

            if max_bytes is not None and seen_bytes + len(raw) > max_bytes:
                raw = raw[: max_bytes - seen_bytes]
                line = raw.decode("utf-8", errors="ignore")

            ids = enc.encode_ordinary(line)
            if ids and max(ids) >= 65536:
                raise ValueError("Token id overflow for uint16 storage. Use a smaller-vocab tokenizer.")

            token_ids.extend(ids)
            seen_bytes += len(raw)

            if seen_bytes >= next_report:
                pct = 100.0 * seen_bytes / (max_bytes or total_size)
                print(f"[{input_path.name}] processed {seen_bytes/1024/1024/1024:.2f} GB ({pct:.1f}%)")
                next_report += report_interval

    arr = np.frombuffer(token_ids, dtype=np.uint16).copy()
    np.save(output_path, arr)

    print(
        f"Saved {output_path} | tokens={arr.size:,} | dtype={arr.dtype} | "
        f"file_size={output_path.stat().st_size/1024/1024/1024:.2f} GB"
    )


def _gunzip_if_needed(data_dir: Path, name: str) -> Path:
    txt_path = data_dir / name
    gz_path = data_dir / f"{name}.gz"

    if txt_path.exists():
        return txt_path

    if not gz_path.exists():
        raise FileNotFoundError(f"Missing both {txt_path} and {gz_path}")

    import gzip
    import shutil

    print(f"Decompressing {gz_path} -> {txt_path}")
    with gzip.open(gz_path, "rb") as src, txt_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    return txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenWebText token ids for baseline training")
    parser.add_argument("--data-dir", type=str, default="/root/autodl-tmp/data/openwebtext")
    parser.add_argument("--train-input", type=str, default="owt_train.txt")
    parser.add_argument("--valid-input", type=str, default="owt_valid.txt")
    parser.add_argument("--train-output", type=str, default="openwebtext_train_tokens.npy")
    parser.add_argument("--valid-output", type=str, default="openwebtext_valid_tokens.npy")
    parser.add_argument("--encoding", type=str, default="gpt2")
    parser.add_argument("--max-train-bytes", type=int, default=None)
    parser.add_argument("--max-valid-bytes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_txt = _gunzip_if_needed(data_dir, args.train_input)
    valid_txt = _gunzip_if_needed(data_dir, args.valid_input)

    train_out = data_dir / args.train_output
    valid_out = data_dir / args.valid_output

    if train_out.exists() and not args.overwrite:
        raise FileExistsError(f"{train_out} exists. Use --overwrite to replace it.")
    if valid_out.exists() and not args.overwrite:
        raise FileExistsError(f"{valid_out} exists. Use --overwrite to replace it.")

    print(f"Encoding train split from {train_txt}")
    _encode_text_file(train_txt, train_out, args.encoding, args.max_train_bytes)

    print(f"Encoding valid split from {valid_txt}")
    _encode_text_file(valid_txt, valid_out, args.encoding, args.max_valid_bytes)


if __name__ == "__main__":
    main()
