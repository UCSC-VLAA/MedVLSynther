#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys
from pathlib import Path
from collections import defaultdict, Counter

def iter_jsonl_files(path_like: str):
    p = Path(path_like)
    if p.is_file():
        yield p
    elif p.is_dir():
        for q in sorted(p.rglob("*.jsonl")):
            yield q
    else:
        for q in sorted(Path().glob(path_like)):
            if q.is_file() and q.suffix == ".jsonl":
                yield q

def safe_len_images(rec) -> int:
    imgs = rec.get("images")
    if isinstance(imgs, list):
        return len(imgs)
    return 0

def pick_qtype(rec) -> str:
    qt = rec.get("Question Type", None)
    if qt is None:
        qt = rec.get("question_type", None)
    if isinstance(qt, str) and qt.strip():
        return qt.strip()
    return "Unknown"

def main():
    ap = argparse.ArgumentParser(
        description="Count distribution of question types and image counts from JSONL outputs."
    )
    ap.add_argument("input", help="JSONL path, e.g.: results.jsonl; outputs/ ; 'shards/*.jsonl'")
    ap.add_argument("--save-csv", default=None, help="export to csv")
    args = ap.parse_args()

    counts = Counter()
    img_sums = defaultdict(int)
    total = 0

    files = list(iter_jsonl_files(args.input))
    if not files:
        print(f"[ERR] No .jsonl files found from: {args.input}", file=sys.stderr)
        sys.exit(1)

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                qtype = pick_qtype(rec)
                nimg = safe_len_images(rec)
                counts[qtype] += 1
                img_sums[qtype] += nimg
                total += 1

    if total == 0:
        print("No records parsed.")
        return

    rows = []
    for qt, c in counts.most_common():
        imgs = img_sums[qt]
        pct = 100.0 * c / total
        avg = imgs / c if c else 0.0
        rows.append((qt, c, pct, imgs, avg))

    col_w = {
        "type": max(6, max(len(r[0]) for r in rows)),
    }
    hdr = f"{'Type':<{col_w['type']}}  {'#Questions':>10}  {'%':>6}  {'#Images':>8}  {'AvgImgs/Q':>9}"
    print(hdr)
    print("-" * len(hdr))
    for qt, c, pct, imgs, avg in rows:
        print(f"{qt:<{col_w['type']}}  {c:>10d}  {pct:>6.2f}  {imgs:>8d}  {avg:>9.2f}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<{col_w['type']}}  {total:>10d}  {100.00:>6.2f}  {sum(img_sums.values()):>8d}  "
          f"{(sum(img_sums.values())/total if total else 0):>9.2f}")

    if args.save_csv:
        outp = Path(args.save_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as w:
            w.write("type,num_questions,percent,num_images,avg_images_per_question\n")
            for qt, c, pct, imgs, avg in rows:
                qt_csv = '"' + qt.replace('"', '""') + '"'
                w.write(f"{qt_csv},{c},{pct:.6f},{imgs},{avg:.6f}\n")
        print(f"[OK] CSV saved to: {outp}")

if __name__ == "__main__":
    main()
