#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, Iterable, List

# 定义答案标签
LABELS = ["A", "B", "C", "D", "E"]

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """逐行读取并解析JSONL文件。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"输入必须是一个 .jsonl 文件: {path}")

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    yield json.loads(s)
                except json.JSONDecodeError:
                    continue

def is_valid_gv(gv: Any) -> bool:
    """检查 'generated_vqa' 字段是否符合预期的格式。"""
    if not isinstance(gv, dict):
        return False
    opts = gv.get("options")
    ans = gv.get("answer")
    if not (isinstance(gv.get("question"), str) and isinstance(opts, dict) and isinstance(ans, str)):
        return False
    if not all(k in opts for k in LABELS):
        return False
    if ans not in LABELS:
        return False
    return True

def calculate_target_counts(k: int) -> Dict[str, int]:
    """计算大小为k的子集中每个标签的目标数量，使其尽可能均匀。"""
    if k < 0:
        raise ValueError("子集规模 K 必须是非负数。")
    base = k // len(LABELS)
    remainder = k % len(LABELS)
    targets = {label: base + (1 if i < remainder else 0) for i, label in enumerate(LABELS)}
    return targets

def main():
    parser = argparse.ArgumentParser(
        description="从一个JSONL文件中均匀抽取一个子集，并保持原始的dataset_index不变。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True, help="输入的JSONL文件路径。")
    parser.add_argument("--output", required=True, help="输出的JSONL文件路径。")
    parser.add_argument("-k", "--size", type=int, required=True, help="期望的子集规模 (K)。")
    parser.add_argument("--seed", type=int, default=42, help="用于随机抽样的种子，以确保结果可复现。")
    args = parser.parse_args()

    random.seed(args.seed)

    print("第一步：正在读取和分组输入文件...")
    records_by_answer: Dict[str, List[Dict]] = {label: [] for label in LABELS}

    total_lines = 0
    valid_records = 0
    for record in iter_jsonl(args.input):
        total_lines += 1
        gv = record.get("generated_vqa")
        if is_valid_gv(gv):
            valid_records += 1
            answer = gv["answer"]
            records_by_answer[answer].append(record)

    print(f"文件扫描完成。共处理 {total_lines} 行，其中有效记录 {valid_records} 条。")
    available_counts = {k: len(v) for k, v in records_by_answer.items()}
    for label, count in available_counts.items():
        print(f"  - {label}: {count} 条")

    print(f"\n第二步：正在为 {args.size} 的子集规模计算目标数量...")
    if args.size > valid_records:
        print(f"错误：请求的子集规模 ({args.size}) 大于文件中的有效记录总数 ({valid_records})。", file=sys.stderr)
        sys.exit(1)

    target_counts = calculate_target_counts(args.size)
    print("目标抽样数量：")
    for label, count in target_counts.items():
        print(f"  - {label}: {count} 条")

    for label in LABELS:
        if target_counts[label] > available_counts[label]:
            print(f"\n错误：无法满足抽样要求。答案 '{label}' 需要 {target_counts[label]} 个样本，但文件中只有 {available_counts[label]} 个可用。", file=sys.stderr)
            sys.exit(1)

    print("\n第三步：正在进行随机抽样...")
    subset = []
    for label in LABELS:
        random.shuffle(records_by_answer[label])
        subset.extend(records_by_answer[label][:target_counts[label]])

    # 对最终的子集再次进行全局随机排序，以打乱答案标签的顺序
    random.shuffle(subset)
    print(f"抽样完成，共选取 {len(subset)} 条记录。")

    print(f"\n第四步：正在将结果写入到 {args.output}...")
    try:
        with open(args.output, "w", encoding="utf-8") as f_out:
            for record in subset:
                # 直接写入原始记录，不修改任何字段
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    except IOError as e:
        print(f"写入文件时出错: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n处理成功！")
    print(f"已从 {args.input} 中抽取 {len(subset)} 条记录并保存到 {args.output}。")
    print("所有记录均保留了其原始的 'dataset_index'。")

if __name__ == "__main__":
    main()
                                    