#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List
from collections import Counter

LABELS = ["A", "B", "C", "D"]

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    continue
    else:
        raise FileNotFoundError(f"Input must be a single .jsonl file: {path}")

def is_valid_gv(gv: Any) -> bool:
    if not isinstance(gv, dict):
        return False
    if not isinstance(gv.get("question"), str):
        return False
    opts = gv.get("options")
    ans  = gv.get("answer")
    if not isinstance(opts, dict) or not isinstance(ans, str):
        return False
    # 必须 A..D 都在，且答案 ∈ A..D
    for k in LABELS:
        if k not in opts or not isinstance(opts[k], str):
            return False
    if ans not in LABELS:
        return False
    return True

def compute_targets(n: int) -> Dict[str, int]:
    base = n // 4
    r = n % 4
    targets = {k: base + (1 if i < r else 0) for i, k in enumerate(LABELS)}
    return targets

def choose_target_label(cur_counts: Dict[str, int], targets: Dict[str, int]) -> str:
    best_label = None
    best_gap = -10**9
    for k in LABELS:
        gap = targets[k] - cur_counts.get(k, 0)
        if gap > best_gap:
            best_gap = gap
            best_label = k
    return best_label

def build_mapping(old_correct: str, new_correct: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    remaining_new = [k for k in LABELS if k != new_correct]
    remaining_old = [k for k in LABELS if k != old_correct]
    mapping[old_correct] = new_correct
    for o, n in zip(remaining_old, remaining_new):
        mapping[o] = n
    return mapping

def remap_options(options: Dict[str, str], mapping: Dict[str, str]) -> Dict[str, str]:
    inv = {v: k for k, v in mapping.items()}  # new -> old
    new_opts = {}
    for k in LABELS:
        old_k = inv[k]
        new_opts[k] = options[old_k]
    return new_opts

def main():
    ap = argparse.ArgumentParser(description="Rebalance correct answer letters (A-E) by relabeling options per sample.")
    ap.add_argument("--input", required=True, help="输入 JSONL（已通过验证的样本）")
    ap.add_argument("--output", required=True, help="输出 JSONL（重排后的）")
    ap.add_argument("--dry_run", action="store_true", help="只打印前后分布，不写文件")
    args = ap.parse_args()

    # ---------- 第 1 遍：统计可重排样本数 ----------
    total = 0
    valid_ids: List[int] = []  # 行号索引（只用于第二遍快速判定）
    initial_counts = Counter()

    for i, rec in enumerate(iter_jsonl(args.input)):
        total += 1
        gv = rec.get("generated_vqa")
        if is_valid_gv(gv):
            valid_ids.append(i)
            initial_counts[gv["answer"]] += 1

    n_valid = len(valid_ids)
    targets = compute_targets(n_valid)

    print("=== Before ===")
    print(f"Total lines: {total}")
    print(f"Valid (relabelable): {n_valid}")
    for k in LABELS:
        v = initial_counts[k]
        print(f"{k}: {v} ({(v/n_valid if n_valid else 0):.2%})")
    print("\nTarget counts:")
    for k in LABELS:
        print(f"{k}: {targets[k]}")

    # ---------- 第 2 遍：逐条重排并写出 ----------
    cur_counts = Counter()
    out_fp = None if args.dry_run else open(args.output, "w", encoding="utf-8")
    # 重新遍历：按顺序把每个“可重排”样本的正确选项送到当前缺口最大的标签
    line_index = -1
    for rec in iter_jsonl(args.input):
        line_index += 1
        gv = rec.get("generated_vqa")
        if is_valid_gv(gv):
            # 选择目标标签
            target = choose_target_label(cur_counts, targets)
            old_ans = gv["answer"]
            mapping = build_mapping(old_ans, target)
            # 应用到 options / answer
            new_options = remap_options(gv["options"], mapping)
            rec["generated_vqa"] = {
                "question": gv["question"],
                "options": new_options,
                "answer": target,
            }
            cur_counts[target] += 1
        # 写出
        if out_fp:
            out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if out_fp:
        out_fp.close()

    print("\n=== After (expected/achieved) ===")
    for k in LABELS:
        print(f"{k}: {cur_counts[k]} (target {targets[k]})  {(cur_counts[k]/n_valid if n_valid else 0):.2%}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
    else:
        print(f"\nWrote rebalanced JSONL -> {args.output}")

if __name__ == "__main__":
    main()