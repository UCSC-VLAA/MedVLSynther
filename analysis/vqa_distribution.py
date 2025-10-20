#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple
from collections import Counter
import re

DEF_STOPWORDS = set("""
a an the and or of to for from by with without within into onto over under in on at as is are was were be being been
it its they them their this that these those which who whom whose what why how where when
we our you your he she his her i me my mine
can could may might should would will shall do does did done having have has had
no not nor only than then also more most much many few several some any all both either neither
such other another same different each per via using use used
figure image photo fig images caption context provided above following according
""".split())

TOKEN_RE = re.compile(r"[A-Za-z]+")

def iter_jsonl(path_like: str) -> Iterable[Dict[str, Any]]:
    p = Path(path_like)
    files: List[Path] = []
    if p.is_file() and p.suffix.lower() == ".jsonl":
        files = [p]
    elif p.is_dir():
        files = sorted(p.rglob("*.jsonl"))
    else:
        files = [f for f in sorted(Path().glob(path_like)) if f.is_file() and f.suffix.lower() == ".jsonl"]
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                yield rec
def tokenize_en(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())

def safe_gv(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gv = rec.get("generated_vqa")
    if not isinstance(gv, dict):
        return None
    q = gv.get("question")
    opts = gv.get("options")
    if not isinstance(q, str) or not isinstance(opts, dict):
        return None
    return gv

def pct(sorted_vals: List[int], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0: return float(sorted_vals[0])
    if p >= 100: return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))

def flatten_secondary_labels(value) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        v = value.strip()
        if v:
            out.append(v)
        return out
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                for lab in item:
                    if isinstance(lab, str):
                        v = lab.strip()
                        if v:
                            out.append(v)
            elif isinstance(item, str):
                v = item.strip()
                if v:
                    out.append(v)
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Terminal stats for generated_vqa + image_secondary_label distribution."
    )
    ap.add_argument("--input", required=True, help="JSONL path")
    ap.add_argument("--include_options_in_words", action="store_true",
                    help="whether including options in word frequency computation")
    ap.add_argument("--top_k", type=int, default=100, help="the number of top high-frequency words to be printed")
    ap.add_argument("--stopwords", type=str, default=None,
                    help="define new stopword")
    ap.add_argument("--top_labels", type=int, default=50,
                    help="the number of image_secondary_label to be printed")
    ap.add_argument("--labels_lower", action="store_true",
                    help="labels lower")
    args = ap.parse_args()

    stopwords = set(DEF_STOPWORDS)
    if args.stopwords:
        with open(args.stopwords, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    stopwords.add(w)

    total = 0
    used = 0
    ans_counter = Counter()
    word_counter = Counter()
    q_token_lengths: List[int] = []
    label_counter = Counter()
    sample_has_labels = 0

    for rec in iter_jsonl(args.input):
        total += 1
        gv = safe_gv(rec)
        if gv is None:
            continue

        ans = gv.get("answer")
        if isinstance(ans, str) and ans in ["A", "B", "C", "D", "E"]:
            ans_counter[ans] += 1

        # token length
        q = gv.get("question") or ""
        q_tokens = tokenize_en(q)
        q_token_lengths.append(len(q_tokens))

        bag = q_tokens[:]
        if args.include_options_in_words:
            opts = gv.get("options") or {}
            for k in ["A", "B", "C", "D", "E"]:
                t = opts.get(k)
                if isinstance(t, str):
                    bag.extend(tokenize_en(t))
        bag = [w for w in bag if w not in stopwords and len(w) > 1]
        word_counter.update(bag)

        lbls = flatten_secondary_labels(rec.get("image_secondary_label"))
        if lbls:
            sample_has_labels += 1
            if args.labels_lower:
                lbls = [l.lower() for l in lbls]
            label_counter.update(lbls)
        used += 1

    print("=========== Generated VQA (terminal stats) ===========")
    print(f"Total lines scanned : {total}")
    print(f"Samples included    : {used}")

    print("\n-- Correct option distribution --")
    tot_ans = sum(ans_counter.values())
    for k in ["A","B","C","D","E"]:
        v = ans_counter[k]
        r = (v / tot_ans) if tot_ans else 0.0
        print(f"{k}: {v} ({r:.2%})")

    # token length stats
    print("\n-- Question token length (word-count) --")
    if q_token_lengths:
        arr = sorted(q_token_lengths)
        mean = sum(arr) / len(arr)
        print(f"Count: {len(arr)}  Min: {arr[0]}  P25: {pct(arr,25):.1f}  Median: {pct(arr,50):.1f}  "
              f"P75: {pct(arr,75):.1f}  P90: {pct(arr,90):.1f}  Max: {arr[-1]}  Mean: {mean:.1f}")
        bins = [0]*10
        upper = max(1, int(pct(arr, 90)))
        for v in arr:
            idx = min(9, int((v / upper) * 10)) if upper > 0 else 0
            if idx == 10: idx = 9
            bins[idx] += 1
        print("\nHistogram (relative to P90):")
        for i, c in enumerate(bins):
            lo = i/10
            hi = (i+1)/10
            bar = "#" * max(1, int(50 * c / max(1, max(bins))))
            print(f"  [{lo:.1f},{hi:.1f}) : {c:6d} {bar}")
    else:
        print("(no questions)")
    print(f"\n-- Top {args.top_k} words {'(question+options)' if args.include_options_in_words else '(question only)'} --")
    for w, c in word_counter.most_common(args.top_k):
        print(f"{w:>16s} : {c}")

    print(f"\n-- image_secondary_label distribution (top {args.top_labels}) --")
    print(f"samples with labels : {sample_has_labels}")
    total_labels = sum(label_counter.values())
    print(f"total label tokens  : {total_labels}")
    for lab, cnt in label_counter.most_common(args.top_labels):
        pct_lab = (cnt / total_labels) if total_labels else 0.0
        print(f"{lab:>32s} : {cnt:6d} ({pct_lab:.2%})")

if __name__ == "__main__":
    main()