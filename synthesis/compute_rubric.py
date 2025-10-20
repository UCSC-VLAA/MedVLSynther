#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional, Tuple


REQUIRED_TITLES = [
    "Stem Self-contained",
    "Vocabulary Constraint",
    "Diagnosis Leak",
    "Single Correct Option",
    "Option Type Consistency",
    "Clinical Validity",
    "Image–Text Consistency",
]

def iter_jsonl_paths(path_like: str) -> Iterable[Path]:
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".jsonl":
        yield p
        return
    if p.is_dir():
        for f in sorted(p.rglob("*.jsonl")):
            yield f
        return
    # glob
    for f in sorted(Path().glob(path_like)):
        if f.is_file() and f.suffix.lower() == ".jsonl":
            yield f

def load_line(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_rubric(rec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """同时兼容两种结构：
       - 顶层: {"rubric":[...]}
       - 嵌套: {"rubric_verification":{"rubric":[...]}}"""
    if "rubric" in rec and isinstance(rec["rubric"], list):
        return rec["rubric"]
    rv = rec.get("rubric_verification")
    if isinstance(rv, dict) and isinstance(rv.get("rubric"), list):
        return rv["rubric"]
    return None

def compute_scores(items: List[Dict[str, Any]]) -> Tuple[int, int, int, int, Optional[float]]:
    """返回: (pos_weight_sum, pos_score_sum, penalty_sum, final_score, normalized_score)"""
    pos_w = 0
    pos_s = 0
    penalty = 0
    for it in items:
        cat = str(it.get("category", "")).strip().lower()
        w = it.get("weight", 0)
        s = it.get("score", 0)
        try:
            w = int(w)
        except Exception:
            w = 0
        try:
            s = int(s)
        except Exception:
            s = 0

        if cat == "pitfall":
            # 触发时 s 通常是负值；未触发是 0
            if s < 0:
                penalty += (-s)
        else:
            if w > 0:
                pos_w += w
                # 模型偶发越界，做个截断
                if s < 0:
                    s = 0
                if s > w:
                    s = w
                pos_s += s
    final_score = pos_s - penalty
    norm = None if pos_w <= 0 else max(0.0, min(1.0, final_score / pos_w))
    return pos_w, pos_s, penalty, final_score, norm

def check_required(items: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """所有必需 title 都存在 且 score==weight 才算通过。返回(通过与否, 失败原因列表)"""
    # 以 title 小写去索引
    by_title = {}
    for it in items:
        t = str(it.get("title", "")).strip()
        if t:
            by_title[t.lower()] = it

    missing = []
    not_full = []
    for req in REQUIRED_TITLES:
        it = by_title.get(req.lower())
        if it is None:
            missing.append(req)
            continue
        # 分数满分判定（不强制 category=Essential，避免模型误标）
        w = it.get("weight", 0)
        s = it.get("score", 0)
        try:
            w = int(w)
        except Exception:
            w = 0
        try:
            s = int(s)
        except Exception:
            s = 0
        if not (w > 0 and s == w):
            not_full.append(req)

    reasons = []
    if missing:
        reasons.append(f"missing_titles:{';'.join(missing)}")
    if not_full:
        reasons.append(f"essential_not_full:{';'.join(not_full)}")
    return (len(missing) == 0 and len(not_full) == 0), reasons

def percentile(sorted_vals: List[float], p: float) -> float:
    """0..100 百分位（含端）"""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1

def histogram(values: List[float], bins: int = 10) -> List[Tuple[str, int]]:
    """0-1 等宽直方图，返回 [(区间文本, 计数), ...]"""
    if not values:
        return []
    counts = [0] * bins
    for v in values:
        if v < 0:
            i = 0
        elif v >= 1:
            i = bins - 1
        else:
            i = int(v * bins)
        counts[i] += 1
    out = []
    for i, c in enumerate(counts):
        lo = i / bins
        hi = (i + 1) / bins
        out.append((f"[{lo:.1f},{hi:.1f})" if i < bins - 1 else f"[{lo:.1f},1.0]", c))
    return out

def main():
    ap = argparse.ArgumentParser(description="Analyze rubric verification JSONL and compute pass rate & score distribution.")
    ap.add_argument("--input", required=True, help="JSONL 文件/目录/通配")
    ap.add_argument("--fail_out", default=None, help="未通过样本输出 JSONL（含失败原因）")
    ap.add_argument("--per_item_out", default=None, help="每条样本的度量输出 JSONL")
    args = ap.parse_args()

    total = 0
    evaluable = 0
    passed = 0

    pass_norm_scores: List[float] = []
    pass_final_scores: List[int] = []

    fail_fp = open(args.fail_out, "w", encoding="utf-8") if args.fail_out else None
    per_fp = open(args.per_item_out, "w", encoding="utf-8") if args.per_item_out else None

    for jf in iter_jsonl_paths(args.input):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                rec = load_line(line)
                if rec is None:
                    continue

                # 跳过 error
                rv = rec.get("rubric_verification")
                if isinstance(rv, dict) and "error" in rv:
                    # 不可评估
                    continue

                rubric = extract_rubric(rec)
                if not rubric:
                    # 没有 rubric
                    continue

                evaluable += 1
                pos_w, pos_s, penalty, final_score, norm = compute_scores(rubric)
                ok, reasons = check_required(rubric)

                if ok:
                    passed += 1
                    if norm is not None:
                        pass_norm_scores.append(norm)
                    pass_final_scores.append(final_score)
                else:
                    if fail_fp:
                        out = {
                            "dataset_index": rec.get("dataset_index"),
                            "article_accession_id": rec.get("article_accession_id"),
                            "image_id": rec.get("image_id"),
                            "fail_reasons": reasons,
                        }
                        fail_fp.write(json.dumps(out, ensure_ascii=False) + "\n")

                if per_fp:
                    per_fp.write(json.dumps({
                        "dataset_index": rec.get("dataset_index"),
                        "pos_weight_sum": pos_w,
                        "pos_score_sum": pos_s,
                        "penalty_sum": penalty,
                        "final_score": final_score,
                        "normalized_score": norm,
                        "passed": ok,
                    }, ensure_ascii=False) + "\n")

    if fail_fp: fail_fp.close()
    if per_fp: per_fp.close()
    print("========== Summary ==========")
    print(f"Total lines:        {total}")
    print(f"Evaluable (rubric): {evaluable}")
    print(f"Passed (all required met @ full score): {passed}")
    ratio = (passed / evaluable) if evaluable else 0.0
    print(f"Pass ratio:         {ratio:.4f}")

    if pass_norm_scores:
        vals = sorted(pass_norm_scores)
        print("\n-- Score distribution (normalized among PASS) --")
        print(f"Count: {len(vals)}  Mean: {sum(vals)/len(vals):.4f}  Median: {percentile(vals,50):.4f}")
        print(f"P10: {percentile(vals,10):.4f}  P25: {percentile(vals,25):.4f}  P75: {percentile(vals,75):.4f}  P90: {percentile(vals,90):.4f}")
        print("\nHistogram (deciles):")
        for bucket, cnt in histogram(vals, bins=10):
            print(f"{bucket:>11} : {cnt}")
    else:
        print("\nNo passing samples with computable normalized scores.")

if __name__ == "__main__":
    main()