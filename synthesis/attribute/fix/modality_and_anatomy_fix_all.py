#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge back fixed modality/anatomy into the original JSONL by dataset_index.

- Reads the ORIGINAL jsonl (large file).
- Reads two small "fixed" jsonl files (one for modality, one for anatomy),
  produced by your earlier fix scripts.
- For each dataset_index present in the fixed files, writes the repaired fields
  back into the original record.
- By default, it only fills when the original value is missing/empty.
  Use --force to overwrite even non-empty values.

Only the following fields are updated (if present in the fixed files):
  Modality-side: ['modality', 'modality_raw_extracted', 'modality_source']
  Anatomy-side : ['anatomy',  'anatomy_raw_extracted',  'anatomy_source']

Everything else remains untouched.

Usage:
  python merge_back_fixes.py \
    --orig /path/to/original.jsonl \
    --mod-fix /path/to/missing_modality_fixed.jsonl \
    --ana-fix /path/to/missing_anatomy_fixed.jsonl \
    --out /path/to/merged.jsonl

Optional:
  --force   (overwrite even if original field is non-empty)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

MOD_FIELDS = ["modality", "modality_raw_extracted", "modality_source"]
ANA_FIELDS = ["anatomy",  "anatomy_raw_extracted",  "anatomy_source"]

def _is_missing(v: Any) -> bool:
    """Treat None / '' / 'null' / 'None' / 'NaN' (case-insensitive) as missing."""
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s in ("", "none", "null", "nan")
    return False

def _load_jsonl(path: str) -> Iterable[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # skip bad lines
                continue

def _to_key(di: Any) -> Optional[str]:
    """Normalize dataset_index to a string key; return None if invalid."""
    if di is None:
        return None
    try:
        # keep exact string form (robust to int/str in different files)
        return str(int(di)) if str(di).strip().isdigit() else str(di).strip()
    except Exception:
        return str(di).strip()

def _build_fix_map(path: Optional[str], fields: list) -> Dict[str, dict]:
    """
    Build a map: dataset_index(str) -> {field: value, ...} from a fixed jsonl.
    Only collects fields in `fields`.
    """
    m: Dict[str, dict] = {}
    if not path:
        return m
    for rec in _load_jsonl(path):
        key = _to_key(rec.get("dataset_index"))
        if not key:
            continue
        payload = {}
        for f in fields:
            if f in rec:
                payload[f] = rec[f]
        if payload:
            # last one wins if duplicates
            m[key] = payload
    return m

def main():
    ap = argparse.ArgumentParser(description="Merge back fixed modality/anatomy into original JSONL by dataset_index.")
    ap.add_argument("--orig", required=True, help="Original JSONL")
    ap.add_argument("--mod-fix", default=None, help="JSONL with fixed modality (from fix_missing_modality.py)")
    ap.add_argument("--ana-fix", default=None, help="JSONL with fixed anatomy  (from fix_missing_anatomy.py)")
    ap.add_argument("--out",  required=True, help="Output merged JSONL")
    ap.add_argument("--force", action="store_true", help="Overwrite even if original field is non-empty")
    args = ap.parse_args()

    mod_map = _build_fix_map(args.mod_fix, MOD_FIELDS)
    ana_map = _build_fix_map(args.ana_fix, ANA_FIELDS)

    # Stats
    total = 0
    wrote = 0
    mod_applied = 0
    mod_skipped = 0
    ana_applied = 0
    ana_skipped = 0

    outp = Path(args.out)
    with outp.open("w", encoding="utf-8") as fout:
        for rec in _load_jsonl(args.orig):
            total += 1
            key = _to_key(rec.get("dataset_index"))

            # Apply modality patch if available
            if key and key in mod_map:
                patch = mod_map[key]
                # modality field itself decides fill/overwrite; the sidecar fields follow
                if args.force or _is_missing(rec.get("modality")):
                    for f in MOD_FIELDS:
                        if f in patch:
                            rec[f] = patch[f]
                    mod_applied += 1
                else:
                    mod_skipped += 1

            # Apply anatomy patch if available
            if key and key in ana_map:
                patch = ana_map[key]
                if args.force or _is_missing(rec.get("anatomy")):
                    for f in ANA_FIELDS:
                        if f in patch:
                            rec[f] = patch[f]
                    ana_applied += 1
                else:
                    ana_skipped += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

    print("========== Merge Back Summary ==========")
    print(f"Original lines read : {total}")
    print(f"Output lines written: {wrote}")
    print("")
    print(f"Modality fixes present : {len(mod_map)}")
    print(f"  - applied            : {mod_applied}")
    print(f"  - skipped (kept orig): {mod_skipped} (use --force to overwrite)")
    print("")
    print(f"Anatomy fixes present  : {len(ana_map)}")
    print(f"  - applied            : {ana_applied}")
    print(f"  - skipped (kept orig): {ana_skipped} (use --force to overwrite)")
    print(f"\n[OK] Wrote merged JSONL -> {outp}")

if __name__ == "__main__":
    main()
