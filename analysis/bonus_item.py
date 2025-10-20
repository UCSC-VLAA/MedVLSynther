#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, Counter

# ======== setup: 7 essential items ========
REQUIRED_TITLES = [
    "Stem Self-contained",
    "Vocabulary Constraint",
    "Diagnosis Leak",
    "Single Correct Option",
    "Option Type Consistency",
    "Clinical Validity",
    "Image–Text Consistency",
]
ESSENTIAL_SET = {t.lower() for t in REQUIRED_TITLES}

# ======== IO ========
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
    if not s: return None
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_rubric(rec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if isinstance(rec.get("rubric"), list):
        return rec["rubric"]
    rv = rec.get("rubric_verification")
    if isinstance(rv, dict) and isinstance(rv.get("rubric"), list):
        return rv["rubric"]
    return None

# ======== normalization ========
def norm_title(x) -> str:
    return (str(x) if x is not None else "").strip()

def title_key(x) -> str:
    return norm_title(x).lower()

def norm_category(x) -> str:
    return (str(x) if x is not None else "").strip().lower()

def norm_weight(x):
    try:
        v = float(str(x).strip())
        if math.isfinite(v):
            r = round(v)
            return int(r) if abs(v - r) < 1e-9 else v
    except Exception:
        pass
    return str(x)

# ======== 主逻辑 ========
def main():
    ap = argparse.ArgumentParser(description="check all non-essential items in JSONL, list all (title, weight, category) and check consistency")
    ap.add_argument("--input", required=True, help="JSONL file/path")
    ap.add_argument("--out-combos-csv", default=None, help="optional：export all possible non essential items to CSV")
    ap.add_argument("--show-examples", type=int, default=5, help="show examples")
    args = ap.parse_args()

    total = 0
    evaluable = 0

    # each sample's essential title set
    schema_counter = Counter()                        
    schema_examples: Dict[frozenset, List[Any]] = defaultdict(list)

    title_display: Dict[str, str] = {}                
    title_weights: Dict[str, set] = defaultdict(set)  
    title_cats: Dict[str, set] = defaultdict(set)    

    combo_counter = Counter()                   

    per_record_schema: List[Tuple[Any, frozenset]] = []  # (dataset_index, schema)

    for fp in iter_jsonl_paths(args.input):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                total += 1
                rec = load_line(line)
                if rec is None:
                    continue

                rub = extract_rubric(rec)
                if not rub:
                    continue
                evaluable += 1

                ne_titles_in_rec: List[str] = []
                seen_keys_in_rec = set()

                for it in rub:
                    t = norm_title(it.get("title"))
                    k = title_key(t)
                    if not t or k in ESSENTIAL_SET:
                        continue

                    title_display.setdefault(k, t)

                    w = norm_weight(it.get("weight", 0))
                    c = norm_category(it.get("category", ""))

                    title_weights[k].add(w)
                    title_cats[k].add(c)
                    combo_counter[(k, w, c)] += 1

                    if k not in seen_keys_in_rec:
                        ne_titles_in_rec.append(k)
                        seen_keys_in_rec.add(k)

                schema = frozenset(ne_titles_in_rec)
                schema_counter[schema] += 1
                ds_idx = rec.get("dataset_index", None)
                if len(schema_examples[schema]) < args.show_examples:
                    schema_examples[schema].append(ds_idx)
                per_record_schema.append((ds_idx, schema))

    print("========== Summary ==========")
    print(f"Total lines scanned     : {total}")
    print(f"Records with rubric     : {evaluable}")

    all_titles = sorted([title_display.get(k, k) for k in set(tk for (tk, *_ ) in combo_counter.keys())],
                        key=lambda s: s.lower())
    print(f"\nNon-essential unique titles (global): {len(all_titles)}")
    for t in all_titles:
        print(f"  - {t}")

    print("\n========== Per-Title Consistency (weight/category) ==========")
    uniform_all = True
    for k in sorted(title_display.keys(), key=lambda s: s.lower()):
        disp = title_display[k]
        ws = title_weights[k]
        cs = title_cats[k]
        combos = Counter()
        for (tk, w, c), cnt in combo_counter.items():
            if tk == k:
                combos[(w, c)] += cnt

        is_uniform = (len(ws) == 1 and len(cs) == 1 and len(combos) == 1)
        uniform_all = uniform_all and is_uniform
        status = "OK" if is_uniform else "MIXED"
        print(f"- {disp}:  weights={sorted(list(ws), key=lambda x: (isinstance(x,str), x))}  "
              f"categories={sorted(list(cs))}  combos={len(combos)}  ==> {status}")

        if not is_uniform:
            for (w, c), cnt in combos.most_common():
                print(f"    * (weight={w}, category={c}) -> {cnt} record(s)")

    print("\n========== Per-Record Schema (set of non-essential titles) ==========")
    print(f"Distinct schemas (by title set, ignoring weight/category): {len(schema_counter)}")
    top = schema_counter.most_common()
    if not top:
        print("No non-essential items found in any record.")
    else:
        for i, (sch, cnt) in enumerate(top[:10], 1):
            disp_titles = [title_display.get(k, k) for k in sorted(list(sch))]
            ex = schema_examples.get(sch, [])[:args.show_examples]
            print(f"[{i}] count={cnt}  titles={disp_titles}")
            if ex:
                print(f"     examples dataset_index={ex}")

        schema_uniform = (len(schema_counter) == 1)
        print(f"\nSchema uniform across records?  {'YES' if schema_uniform else 'NO'}")

        if not schema_uniform:
            base_schema = top[0][0]
            base_titles = set(base_schema)
            print("\nDifferences relative to the most common schema:")
            shown = 0
            for ds_idx, sch in per_record_schema:
                if sch == base_schema:
                    continue
                missing = [title_display.get(k, k) for k in sorted(list(base_titles - sch))]
                extra   = [title_display.get(k, k) for k in sorted(list(sch - base_titles))]
                print(f"- dataset_index={ds_idx}  missing={missing}  extra={extra}")
                shown += 1
                if shown >= args.show_examples:
                    break

    if args.out_combos_csv:
        import csv
        with open(args.out_combos_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["title","weight","category","records_with_this_combo"])
            for (k,wg,cat), cnt in sorted(combo_counter.items(),
                                          key=lambda kv: (title_display.get(kv[0][0], kv[0][0]).lower(), str(kv[0][1]), kv[0][2])):
                w.writerow([title_display.get(k, k), wg, cat, cnt])
        print(f"\n[OK] Wrote combo details -> {args.out_combos_csv}")

if __name__ == "__main__":
    main()
