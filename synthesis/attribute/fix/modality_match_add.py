# /opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_filtered_mean9670_13k_rebalanced_subset10k_type.jsonl

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge modality / anatomy from NEW jsonl into OLD jsonl by key=(normalized question + image members).

- NEW jsonl: contains top-level fields: 'modality' and/or 'anatomy' (plus images/question)
- OLD jsonl: the large metadata jsonl to be updated
- Output: OLD jsonl with 'modality'/'anatomy' attached (others unchanged)

Keying:
  (normalized question, sorted tuple of image members)
  image members are extracted from: image_wds_member OR image_wds_path / images[].path OR image_file_name

Conflicts:
  A key may map to multiple modalities and/or anatomies in NEW.
  We detect conflicts separately for modality and anatomy, and resolve by strategy:
    - none:  report and abort
    - majority: pick the most frequent label (ties broken by first appearance)
    - first:   pick the first appearance
    - last:    pick the last appearance   [default]

Write policy:
  - By default, only fill when OLD value is missing/empty (None / '' / 'null' / 'None' / 'NaN').
  - Use --force to overwrite even non-empty values.

Usage:
  python merge_mod_ana_by_qimg.py \
    --new_jsonl /path/to/new.jsonl \
    --old_jsonl /path/to/old.jsonl \
    --out_jsonl /path/to/merged.jsonl \
    [--resolve_mod last] [--resolve_ana last] [--force] \
    [--conflict_dump_mod mod_conflicts.jsonl] \
    [--conflict_dump_ana ana_conflicts.jsonl] \
    [--show_conflicts 10]
"""

import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

# ---------------- JSONL streaming ----------------
def load_jsonl(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                rec["_line_no"] = ln  # for debugging
                yield rec
            except Exception:
                continue

# ---------------- Key helpers ----------------
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
    # 1) image_wds_member
    mem = rec.get("image_wds_member", None)
    if isinstance(mem, list) and mem:
        return tuple(sorted({str(x).lower() for x in mem if x}))

    # 2) image_wds_path / images[].path
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

    # 3) image_file_name
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

# ---------------- Utils ----------------
def _is_missing(v: Any) -> bool:
    """Treat None / '' / 'null' / 'None' / 'NaN' (case-insensitive) as missing."""
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s in ("", "none", "null", "nan")
    return False

def _choose_label_for_key(
    strategy: str, cnt: Counter, samples: List[Dict[str, Any]], field_name: str
) -> str:
    """
    Pick a single label for a key given counts and sample order.
    field_name: 'modality' or 'anatomy'
    """
    if len(cnt) == 1:
        return next(iter(cnt.keys()))
    if strategy == "majority":
        best = cnt.most_common()
        top_count = best[0][1]
        tied = {t for t, c in best if c == top_count}
        if len(tied) == 1:
            return best[0][0]
        # tie -> first appearance
        for s in samples:
            t = (s.get(field_name) or "").strip()
            if t in tied:
                return t
    elif strategy == "first":
        for s in samples:
            t = (s.get(field_name) or "").strip()
            if t:
                return t
    elif strategy == "last":
        for s in reversed(samples):
            t = (s.get(field_name) or "").strip()
            if t:
                return t
    # Fallback (shouldn't hit)
    return next(iter(cnt.keys()))

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Merge modality/anatomy from NEW jsonl into OLD jsonl by (question + image members).")
    ap.add_argument("--new_jsonl", required=True, help="NEW jsonl with modality/anatomy")
    ap.add_argument("--old_jsonl", required=True, help="OLD jsonl to be updated")
    ap.add_argument("--out_jsonl", required=True, help="Output path")

    ap.add_argument("--resolve_mod", choices=["none", "majority", "first", "last"], default="none",
                    help="Conflict strategy for modality (default: last)")
    ap.add_argument("--resolve_ana", choices=["none", "majority", "first", "last"], default="none",
                    help="Conflict strategy for anatomy  (default: last)")

    ap.add_argument("--conflict_dump_mod", default=None, help="Dump modality-conflict samples to this jsonl (optional)")
    ap.add_argument("--conflict_dump_ana", default=None, help="Dump anatomy-conflict samples to this jsonl (optional)")
    ap.add_argument("--show_conflicts", type=int, default=10, help="How many conflict keys to preview")

    ap.add_argument("--force", action="store_true", help="Overwrite even if OLD has non-empty values (default: fill missing only)")
    args = ap.parse_args()

    # 1) Scan NEW and build per-key counters/samples
    print(f"[INFO] Scanning NEW jsonl: {args.new_jsonl}")
    total_new = 0

    mod_key2cnt: Dict[Tuple[str, Tuple[str, ...]], Counter] = defaultdict(Counter)
    mod_key2samples: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    ana_key2cnt: Dict[Tuple[str, Tuple[str, ...]], Counter] = defaultdict(Counter)
    ana_key2samples: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)

    mod_set = Counter()
    ana_set = Counter()

    for rec in load_jsonl(args.new_jsonl):
        total_new += 1
        k = make_key_qimg(rec)
        mod = (rec.get("modality") or "").strip()
        ana = (rec.get("anatomy")  or "").strip()

        if mod:
            mod_key2cnt[k][mod] += 1
            mod_key2samples[k].append(rec)
            mod_set.update([mod])
        if ana:
            ana_key2cnt[k][ana] += 1
            ana_key2samples[k].append(rec)
            ana_set.update([ana])

    # Report distinct sets
    print("\n========== NEW: label stats ==========")
    print(f"- NEW total lines           : {total_new}")
    print(f"- NEW keys with modality    : {sum(1 for c in mod_key2cnt.values() if sum(c.values())>0)}")
    print(f"- NEW keys with anatomy     : {sum(1 for c in ana_key2cnt.values() if sum(c.values())>0)}")
    if mod_set:
        print(f"- Distinct modality labels  : {len(mod_set)}")
        for t, c in mod_set.most_common(15):
            print(f"  * {t}: {c}")
        if len(mod_set) > 15:
            print(f"  ... (+{len(mod_set)-15} more)")
    if ana_set:
        print(f"- Distinct anatomy labels   : {len(ana_set)}")
        for t, c in ana_set.most_common(15):
            print(f"  * {t}: {c}")
        if len(ana_set) > 15:
            print(f"  ... (+{len(ana_set)-15} more)")

    # Conflicts
    mod_conflicts = [(k, cnt) for k, cnt in mod_key2cnt.items() if len(cnt) > 1]
    ana_conflicts = [(k, cnt) for k, cnt in ana_key2cnt.items() if len(cnt) > 1]

    print("\n========== Conflict summary ==========")
    print(f"- modality conflict keys : {len(mod_conflicts)} (strategy={args.resolve_mod})")
    print(f"- anatomy  conflict keys : {len(ana_conflicts)} (strategy={args.resolve_ana})")

    # Optional dumps
    if args.conflict_dump_mod and mod_conflicts:
        outp = Path(args.conflict_dump_mod); outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for k, _ in mod_conflicts:
                for rec in mod_key2samples[k]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] Dumped modality-conflict records -> {args.conflict_dump_mod}")

    if args.conflict_dump_ana and ana_conflicts:
        outp = Path(args.conflict_dump_ana); outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for k, _ in ana_conflicts:
                for rec in ana_key2samples[k]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] Dumped anatomy-conflict records -> {args.conflict_dump_ana}")

    # Preview a few conflicts
    def _preview_conflicts(tag: str, conflicts, samples_map):
        if not conflicts:
            return
        print(f"\n-- {tag} conflicts preview --")
        for i, (k, cnt) in enumerate(conflicts[:args.show_conflicts], 1):
            qn, mems = k
            mems_str = ", ".join(mems) if mems else "<no-image-key>"
            print(f"[{i}] q='{qn[:100]}'")
            print(f"    imgs=[{mems_str}]")
            for t, c in cnt.most_common():
                print(f"      - {t}: {c} occurrence(s)")
            for s in samples_map[k][:2]:
                di = s.get("dataset_index", "NA")
                ln = s.get("_line_no", "NA")
                lab = (s.get(tag) or "").strip()
                print(f"        example: dataset_index={di}, line={ln}, {tag}='{lab}'")

    _preview_conflicts("modality", mod_conflicts, mod_key2samples)
    _preview_conflicts("anatomy",  ana_conflicts, ana_key2samples)

    # If conflicts exist and strategy == none -> abort
    if mod_conflicts and args.resolve_mod == "none":
        print("[ERROR] Modality conflicts present; rerun with --resolve_mod majority|first|last or dump for manual fix.")
        sys.exit(2)
    if ana_conflicts and args.resolve_ana == "none":
        print("[ERROR] Anatomy conflicts present; rerun with --resolve_ana majority|first|last or dump for manual fix.")
        sys.exit(2)

    # Build final key->label maps after resolution
    key2mod_final: Dict[Tuple[str, Tuple[str, ...]], str] = {}
    for k, cnt in mod_key2cnt.items():
        key2mod_final[k] = _choose_label_for_key(args.resolve_mod, cnt, mod_key2samples[k], "modality")

    key2ana_final: Dict[Tuple[str, Tuple[str, ...]], str] = {}
    for k, cnt in ana_key2cnt.items():
        key2ana_final[k] = _choose_label_for_key(args.resolve_ana, cnt, ana_key2samples[k], "anatomy")

    print(f"\n[INFO] Effective mapping sizes: modality={len(key2mod_final)}  anatomy={len(key2ana_final)}")

    # 2) Apply to OLD
    print(f"\n[INFO] Applying to OLD jsonl: {args.old_jsonl}")
    total_old = 0
    wrote = 0

    mod_applied = mod_skipped = 0
    ana_applied = ana_skipped = 0

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as fout:
        for rec in load_jsonl(args.old_jsonl):
            total_old += 1
            k = make_key_qimg(rec)

            # Modality update
            if k in key2mod_final:
                new_mod = key2mod_final[k]
                if args.force or _is_missing(rec.get("modality")):
                    rec["modality"] = new_mod
                    mod_applied += 1
                else:
                    mod_skipped += 1

            # Anatomy update
            if k in key2ana_final:
                new_ana = key2ana_final[k]
                if args.force or _is_missing(rec.get("anatomy")):
                    rec["anatomy"] = new_ana
                    ana_applied += 1
                else:
                    ana_skipped += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

    print("\n========== Merge Summary ==========")
    print(f"- OLD total lines   : {total_old}")
    print(f"- Output written    : {wrote}")
    print("")
    print(f"- Modality map size : {len(key2mod_final)}")
    print(f"  * applied         : {mod_applied}")
    print(f"  * skipped (kept)  : {mod_skipped}  (use --force to overwrite)")
    print("")
    print(f"- Anatomy map size  : {len(key2ana_final)}")
    print(f"  * applied         : {ana_applied}")
    print(f"  * skipped (kept)  : {ana_skipped}  (use --force to overwrite)")
    print(f"\n[OK] Wrote -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
