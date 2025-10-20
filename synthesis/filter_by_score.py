#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from typing import Iterable, Optional, Dict, Any

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

def main():
    ap = argparse.ArgumentParser(description="Filter JSONL by verify_pass and verify_score >= threshold.")
    ap.add_argument("--input", required=True, help="输入 JSONL 文件/目录/通配（需含 verify_pass / verify_score）")
    ap.add_argument("--output", required=True, help="输出 JSONL（只保留通过且得分≥阈值的样本）")
    ap.add_argument("--min_score", type=float, default=0.95, help="得分阈值（默认 0.95；若是 final 分数也照此数值比较）")
    ap.add_argument("--require_pass", action="store_true", help="要求 verify_pass=True（默认不开则只按分数筛）")
    args = ap.parse_args()

    kept, seen = 0, 0
    out_fp = open(args.output, "w", encoding="utf-8")

    for jf in iter_jsonl_paths(args.input):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                rec = load_line(line)
                if rec is None:
                    continue
                seen += 1
                vp = rec.get("verify_pass", None)
                sc = rec.get("verify_score", None)

                # 只在有分数（非 None）时比较
                if sc is None:
                    continue
                if args.require_pass and vp is not True:
                    continue
                if float(sc) < args.min_score:
                    continue

                out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    out_fp.close()
    print(f"[Done] Scanned {seen} lines; kept {kept} lines (min_score={args.min_score}, require_pass={args.require_pass}) -> {args.output}")

if __name__ == "__main__":
    main()