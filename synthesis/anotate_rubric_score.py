#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
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
        yield p; return
    if p.is_dir():
        for f in sorted(p.rglob("*.jsonl")):
            yield f
        return
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
    if "rubric" in rec and isinstance(rec["rubric"], list):
        return rec["rubric"]
    rv = rec.get("rubric_verification")
    if isinstance(rv, dict) and isinstance(rv.get("rubric"), list):        
        return rv["rubric"]
    return None

def compute_scores(items: List[Dict[str, Any]]) -> Tuple[int, int, int, int, Optional[float]]:
    pos_w = 0
    pos_s = 0
    penalty = 0
    for it in items:
        cat = str(it.get("category", "")).strip().lower()
        w = it.get("weight", 0)
        s = it.get("score", 0)
        try: w = int(w)
        except Exception: w = 0
        try: s = int(s)
        except Exception: s = 0

        if cat == "pitfall":
            if s < 0:
                penalty += (-s)
        else:
            if w > 0:
                pos_w += w
                if s < 0: s = 0
                if s > w: s = w
                pos_s += s
    final_score = pos_s - penalty
    norm = None if pos_w <= 0 else max(0.0, min(1.0, final_score / pos_w))
    return pos_w, pos_s, penalty, final_score, norm

def check_required(items: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    by_title = {}
    for it in items:
        t = str(it.get("title", "")).strip()
        if t:
            by_title[t.lower()] = it

    missing, not_full = [], []
    for req in REQUIRED_TITLES:
        it = by_title.get(req.lower())
        if it is None:
            missing.append(req); continue
        w = it.get("weight", 0)
        s = it.get("score", 0)
        try: w = int(w)
        except Exception: w = 0
        try: s = int(s)
        except Exception: s = 0
        if not (w > 0 and s == w):
            not_full.append(req)

    reasons = []
    if missing:  reasons.append(f"missing_titles:{';'.join(missing)}")
    if not_full: reasons.append(f"essential_not_full:{';'.join(not_full)}")
    return (len(missing) == 0 and len(not_full) == 0), reasons

def main():
    ap = argparse.ArgumentParser(description="Append verification pass & score to each JSONL line.")
    ap.add_argument("--input", required=True, help="JSONL 文件/目录/通配")
    ap.add_argument("--output", required=True, help="输出 JSONL（在每行追加 verify_pass / verify_score）")
    ap.add_argument("--score_type", choices=["normalized", "final"], default="normalized",
                    help="写入的得分类型：normalized=0..1（默认），final=整数（正向得分-罚分）")
    args = ap.parse_args()

    out_fp = open(args.output, "w", encoding="utf-8")
    total, annotated = 0, 0

    for jf in iter_jsonl_paths(args.input):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                rec = load_line(line)
                if rec is None:
                    continue

                # 若包含 error，视为不可评估
                rv = rec.get("rubric_verification")
                if isinstance(rv, dict) and "error" in rv:
                    rec["verify_pass"]  = False
                    rec["verify_score"] = None
                    out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    annotated += 1
                    continue

                rubric = extract_rubric(rec)
                if not rubric:
                    rec["verify_pass"]  = False
                    rec["verify_score"] = None
                    out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    annotated += 1
                    continue

                _, _, _, final_score, norm = compute_scores(rubric)
                ok, _ = check_required(rubric)

                rec["verify_pass"] = bool(ok)
                if args.score_type == "final":
                    rec["verify_score"] = None if final_score is None else int(final_score)
                else:
                    rec["verify_score"] = norm  # 0..1 或 None

                out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                annotated += 1

    out_fp.close()
    print(f"[Done] Read {total} lines; wrote {annotated} lines -> {args.output}")

if __name__ == "__main__":
    main()