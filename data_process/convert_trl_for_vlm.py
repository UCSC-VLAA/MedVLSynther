#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dotenv
dotenv.load_dotenv()

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import click
import datasets
from datasets import Sequence, Image as HFImage, Value
from transformers import AutoTokenizer

INSTRUCTION_PROMPT = (
    "You will solve a problem/request. You should provide your thoughts within <think> </think> "
    "tags before providing the answer.\nWrite your final answer within <answer> </answer> tags."
)

def tokenize_sample(sample, tokenizer):
    question = sample["question"]
    raw_options = sample["options"]              # 必须是 JSON 字符串
    options = json.loads(raw_options)

    # 选项顺序尽量稳定
    prompt = f"Question: {question}\n\nOptions:"
    for letter in ["A", "B", "C", "D", "E"]:
        if letter in options:
            prompt += f"\n\n{letter}. {options[letter]}"
    # 补齐可能遗漏的键（若数据不止 A-E）
    for letter, option in options.items():
        if letter not in ["A", "B", "C", "D", "E"]:
            prompt += f"\n\n{letter}. {option}"

    prompt = INSTRUCTION_PROMPT + "\n\n" + prompt

    answer_label = sample["answer_label"]
    reasoning = sample["reasoning"]  # 已过滤非空

    response = f"<think> {reasoning.strip()} </think>\n<answer> {answer_label.strip()} </answer>"

    images = sample.get("images", []) or []
    images_prompt = [{"type": "image", "image": img} for img in images]

    message = [
        {"role": "user", "content": images_prompt + [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]
    text = tokenizer.apply_chat_template(message, tokenize=False)

    # 保持空列表（而不是 None），以维持 Feature 一致性
    return {"text": text, "images": images}

@click.command()
@click.option("--local_data_dir", "-d", type=str, default=None,
              help="本地 HF 数据集目录（save_to_disk 产生的路径）。")
@click.option("--is_local", type=bool, default=False, help="是否从本地目录加载（而不是 Hub）。")
@click.option("--dataset_path", type=str, default=None,
              help="远程数据集 repo id（当 is_local=False 时使用）。")
@click.option("--dataset_subset", type=str, default=None, help="远程数据集的 subset/config 名（可选）。")
@click.option("--dataset_split", type=str, default="train", help="数据集 split（默认 train）。")
@click.option("--num_proc", "-n", type=int, default=16, help="并行 map 的进程数。")
@click.option("--tokenizer_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="分词器名称。")
@click.option("--keep_in_memory", type=bool, default=True, help="map 过程中是否常驻内存。")
@click.option("--out_parquet", type=str, required=True, help="输出 Parquet 文件路径。")

def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    print(f"Arguments: {args}")

    # 1) 加载数据集
    if args.is_local:
        if not args.local_data_dir:
            raise ValueError("当 --is_local=true 时必须提供 --local_data_dir。")
        ds = datasets.load_from_disk(Path(args.local_data_dir))
        print(f"Loaded dataset from local dir: {args.local_data_dir}")
    else:
        if not args.dataset_path:
            raise ValueError("当 --is_local=false 时必须提供 --dataset_path。")
        ds = datasets.load_dataset(
            args.dataset_path,
            name=args.dataset_subset,   # ← 修正为 name
            split=args.dataset_split,
        )
        print(f"Loaded dataset from Hub: {args.dataset_path} (name={args.dataset_subset}, split={args.dataset_split})")

    print(f"Dataset size: {len(ds)}")

    # 2) 过滤掉没有 reasoning 或空白 reasoning 的样本
    ds = ds.filter(lambda x: (x.get("reasoning") is not None) and (str(x["reasoning"]).strip() != ""), num_proc=args.num_proc)
    print(f"Filtered (non-empty reasoning) size: {len(ds)}")

    # 3) 构造 text 列
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # 先试跑一条
    _ = tokenize_sample(sample=ds[0], tokenizer=tokenizer)

    ds = ds.map(
        partial(tokenize_sample, tokenizer=tokenizer),
        num_proc=args.num_proc,
        desc="Tokenizing dataset",
        keep_in_memory=args.keep_in_memory,
    )

    # 4) 写 Parquet（把 images 转成 bytes/path 结构）
    #    - 若 images 已经是 HF Image(decode=False) 的 bytes/path 结构则会保持；
    #    - 若是 PIL.Image，需要 cast 成 Sequence(Image(decode=False))。
    try:
        ds_bytes = ds.cast_column("images", Sequence(HFImage(decode=False)))
    except Exception as e:
        print(f"[warn] cast images to bytes/path failed, will try to write as-is: {e}")
        ds_bytes = ds

    # （可选）确保 text、reasoning 是 string 可空列（通常不需要，但更稳）
    try:
        ds_bytes = ds_bytes.cast_column("text", Value("string"))
    except Exception:
        pass
    try:
        ds_bytes = ds_bytes.cast_column("reasoning", Value("string"))
    except Exception:
        pass

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_bytes.to_parquet(str(out_path))
    print(f"[OK] Wrote tokenized dataset (Parquet) -> {out_path}")

if __name__ == "__main__":
    main()