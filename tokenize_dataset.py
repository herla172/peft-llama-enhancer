import argparse
import json
import os

import tqdm.auto as tqdm

import datasets
import transformers


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def read_lm_dataformat(path):
    import lm_dataformat
    reader = lm_dataformat.Reader(path)
    yield from reader.stream_data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_format", type=str, default="jsonl")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--shard_size", type=int, default=100000)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)

    all_tokenized = []
    if args.data_format == "jsonl":
        reader = read_jsonl(args.data_path)
    elif args.data_format == "lm_dataformat":
        reader = read_lm_dataformat(args.data_path)
    else:
        raise KeyError(args.data_format)

  