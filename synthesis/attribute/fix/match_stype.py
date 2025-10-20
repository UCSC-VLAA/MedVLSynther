#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

# --------- JSONL streaming ----------
def load_jsonl(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                rec["_line_no"] = ln  # 便于定位
                yield rec
            except Exception:
                continue

# --------- Key (q + image members) ----------
def _norm_q(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _extract_wds_members_from_paths(paths: List[Any]) -> List[str]:
    out = []
    for p in paths:
        if not p:
            continue
        p = str(p).strip().lower()
        m = p.split("::", 1)[1] if "::" in p else os.path.basename(p)
        if m:
            out.append(m)
    return out

def image_key_members(rec: Dict[str, Any]) -> Tuple[str, ...]:
    mem = rec.get("image_wds_member", None)
    if isinstance(mem, list) and mem:
        return tuple(sorted({str(x).lower() for x in mem if x}))

    paths: List[Any] = []
    v = rec.get("image_wds_path", None)
    if isinstance(v, list):
        paths.extend(v)
    elif isinstance(v, str):
        paths.append(v)
    imgs = rec.get("images", None)
    if isinstance(imgs, list):
        for e in imgs:
            if isinstance(e, dict) and e.get("path"):
                paths.append(e["path"])
            elif isinstance(e, str):
                paths.append(e)
    members = _extract_wds_members_from_paths(paths)
    if members:
        return tuple(sorted(set(members)))

    fns = rec.get("image_file_name", None)
    if isinstance(fns, list) and fns:
        return tuple(sorted({str(x).lower() for x in fns if x}))
    if isinstance(fns, str) and fns:
        return (fns.strip().lower(),)

    return tuple()

def extract_question(rec: Dict[str, Any]) -> str:
    return rec.get("question") or (rec.get("generated_vqa") or {}).get("question") or ""

def make_key_qimg(rec: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    qn = _norm_q(extract_question(rec))
    mems = image_key_members(rec)
    return (qn, mems)

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Attach 'Question Type' from NEW jsonl to OLD jsonl via (normalized question + image members). Supports conflict inspection and resolution.")
    ap.add_argument("--new_jsonl", required=True, help="含 'Question Type' 的新 JSONL")
    ap.add_argument("--old_jsonl", required=True, help="将被附加类型的老 JSONL")
    ap.add_argument("--out_jsonl", required=True, help="输出 JSONL")
    ap.add_argument("--expected_types", type=int, default=7, help="期望 Question Type 种类数（默认 7）")
    ap.add_argument("--resolve_strategy", choices=["none", "majority", "first", "last"], default="last",
                    help="冲突消解策略：none=仅报告并退出；majority=少数服从多数；first/last=按出现顺序取第一个/最后一个")
    ap.add_argument("--conflict_dump", default=None, help="若给出，将冲突样本逐条导出到该 JSONL，便于人工检查")
    ap.add_argument("--show_conflicts", type=int, default=10, help="最多在终端展示多少个冲突键的摘要")
    args = ap.parse_args()

    # 1) 扫描 NEW：统计类型、收集冲突
    print(f"[INFO] Scanning NEW jsonl: {args.new_jsonl}")
    total_new = 0
    type_set: set = set()

    # key -> Counter(type) 以及样本列表（便于打印/导出）
    key2type_counts: Dict[Tuple[str, Tuple[str, ...]], Counter] = defaultdict(Counter)
    key2samples: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)

    for rec in load_jsonl(args.new_jsonl):
        total_new += 1
        qtype = rec.get("Question Type") or rec.get("question_type")
        if not qtype:
            continue
        qtype = str(qtype).strip()
        if not qtype:
            continue
        type_set.add(qtype)
        k = make_key_qimg(rec)
        key2type_counts[k][qtype] += 1
        key2samples[k].append(rec)

    print("========== NEW JSONL: Question Type Stats ==========")
    print(f"- total lines scanned : {total_new}")
    print(f"- distinct types      : {len(type_set)}")
    if type_set:
        # 分布（再扫一遍不必要，直接用 key2type_counts 汇总即可）
        total_by_type = Counter()
        for cnt in key2type_counts.values():
            total_by_type.update(cnt)
        for t, c in total_by_type.most_common():
            print(f"  * {t}: {c}")

    if len(type_set) != args.expected_types:
        print(f"[ERROR] Distinct 'Question Type' = {len(type_set)} (expected {args.expected_types}). Abort.")
        sys.exit(3)

    # 找冲突键（一个键对应 >=2 种类型）
    conflicts = [(k, cnt) for k, cnt in key2type_counts.items() if len(cnt) > 1]
    print(f"[INFO] Conflicting keys: {len(conflicts)}")

    # 可选导出冲突样本
    if args.conflict_dump and conflicts:
        outp = Path(args.conflict_dump)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for k, _ in conflicts:
                for rec in key2samples[k]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] Dumped conflict records -> {args.conflict_dump}")

    # 预览若干冲突
    if conflicts:
        print("\n-- Conflict preview --")
        for i, (k, cnt) in enumerate(conflicts[:args.show_conflicts], 1):
            qn, mems = k
            mems_str = ", ".join(mems) if mems else "<no-image-key>"
            print(f"[{i}] q='{qn[:100]}'")
            print(f"    imgs=[{mems_str}]")
            for t, c in cnt.most_common():
                print(f"      - {t}: {c} occurrence(s)")
            # 打印两个样例
            for s in key2samples[k][:2]:
                di = s.get("dataset_index", "NA")
                ln = s.get("_line_no", "NA")
                qt = s.get("Question Type") or s.get("question_type")
                print(f"        example: dataset_index={di}, line={ln}, type='{qt}'")
        print()

    # 2) 若有冲突，根据策略决定是否继续
    if conflicts and args.resolve_strategy == "none":
        print("[ERROR] Found conflicting Question Types. Re-run with --resolve_strategy majority|first|last "
              "or dump conflicts via --conflict_dump for manual fix.")
        sys.exit(2)

    # 3) 生成最终 key->type 映射（已按策略消解）
    def choose_type_for_key(k, cnt: Counter, samples: List[Dict[str, Any]]) -> str:
        if len(cnt) == 1:
            return next(iter(cnt.keys()))
        if args.resolve_strategy == "majority":
            # 票数最高；若并列，取按出现顺序的第一个并列类型
            best = cnt.most_common()
            top_count = best[0][1]
            tied = {t for t, c in best if c == top_count}
            if len(tied) == 1:
                return best[0][0]
            # 并列：按样本出现顺序挑第一个类型
            for s in samples:
                t = (s.get("Question Type") or s.get("question_type") or "").strip()
                if t in tied:
                    return t
        elif args.resolve_strategy == "first":
            # 第一次出现的类型
            for s in samples:
                t = (s.get("Question Type") or s.get("question_type") or "").strip()
                if t:
                    return t
        elif args.resolve_strategy == "last":
            # 最后一次出现的类型
            for s in reversed(samples):
                t = (s.get("Question Type") or s.get("question_type") or "").strip()
                if t:
                    return t
        # 回退（不该走到这里）
        return next(iter(cnt.keys()))

    key2type_final: Dict[Tuple[str, Tuple[str, ...]], str] = {}
    for k, cnt in key2type_counts.items():
        key2type_final[k] = choose_type_for_key(k, cnt, key2samples[k])

    print(f"[INFO] Effective mapping keys: {len(key2type_final)} (after conflict resolution: {args.resolve_strategy})")

    # 4) 扫描 OLD，命中即附加/覆盖 "Question Type"
    print(f"\n[INFO] Applying to OLD jsonl: {args.old_jsonl}")
    total_old = matched = unmatched = 0
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as fout:
        for rec in load_jsonl(args.old_jsonl):
            total_old += 1
            k = make_key_qimg(rec)
            qtype = key2type_final.get(k)
            if qtype:
                rec["Question Type"] = qtype
                matched += 1
            else:
                unmatched += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n========== Summary ==========")
    print(f"- NEW distinct types          : {len(type_set)} (expected={args.expected_types})")
    print(f"- Conflicting keys            : {len(conflicts)} (strategy={args.resolve_strategy})")
    print(f"- OLD total lines             : {total_old}")
    print(f"- Matched & annotated         : {matched}")
    print(f"- Unmatched (left unchanged)  : {unmatched}")
    print(f"[OK] Wrote -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
