#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge filtered+rebalanced VQA JSONL into ONE parquet with the official eval schema:
images(list<struct<bytes:binary,path:string>>),
question(string),
options(string JSON),
answer_label(string),
answer(string),
dataset_name(string),
hash(string),
dataset_index(int64),
reasoning(null),
misc(string).

Usage:
python build_eval_parquet.py \
  --jsonl /path/biomedica_vqa_1_verified_balanced_5k.jsonl \
  --parquet_dir /home/efs/nwang60/datasets/biomedica_webdataset_VQA_parquet_10k \
  --out_parquet /home/efs/nwang60/datasets/biomedica_internvl3_verified_5k.parquet \
  --dataset_name biomedica_internvl3_verified_5K
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


# ---------- IO helpers ----------
def iter_jsonl(path_like: str):
    p = Path(path_like)
    files: List[Path] = []
    if p.is_file() and p.suffix.lower() == ".jsonl":
        files = [p]
    elif p.is_dir():
        files = sorted(p.rglob("*.jsonl"))
    else:
        files = [f for f in sorted(Path().glob(path_like))
                 if f.is_file() and f.suffix.lower() == ".jsonl"]
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    continue

# ---------- keys & extraction ----------
def composite_key(rec: Dict[str, Any]) -> Tuple:
    aid = rec.get("article_accession_id", None)
    img_ids = tuple(rec.get("image_id", []) or [])
    img_names = tuple(rec.get("image_file_name", []) or [])
    return (aid, img_ids, img_names)


def norm_generated_vqa(gv: Any) -> Optional[Dict[str, Any]]:
    """Expect dict with 'question', 'options'{A..E}, 'answer'/'answer_label'."""
    if not isinstance(gv, dict):
        return None
    q = gv.get("question")
    opts = gv.get("options") or {}
    ans_label = gv.get("answer")
    # 有的存的是 'answer'（字母），也可能是 'answer_label'
    if isinstance(gv.get("answer_label"), str):
        ans_label = gv.get("answer_label")
    if not (isinstance(q, str) and isinstance(opts, dict) and isinstance(ans_label, str)):
        return None
    # 确保 A..E 都在（允许空字符串，但必须有键）
    for k in ["A", "B", "C", "D"]:
        if k not in opts:
            return None
        if not isinstance(opts[k], str):
            # 强制成字符串，None 用空串兜底
            opts[k] = "" if opts[k] is None else str(opts[k])
    if ans_label not in ["A", "B", "C", "D"]:
        return None
    ans_text = opts.get(ans_label)
    return {
        "question": q,
        "options": opts,
        "answer_label": ans_label,
        "answer": ans_text,
    }

def load_jsonl_gv(jsonl_path: str):
    """构建两张索引表：by dataset_index & by composite key。"""
    by_ds_idx: Dict[int, Dict[str, Any]] = {}
    by_comp: Dict[Tuple, Dict[str, Any]] = {}
    total = ok = 0
    for rec in iter_jsonl(jsonl_path):
        total += 1
        gv = norm_generated_vqa(rec.get("generated_vqa"))
        if gv is None:
            continue
        # dataset_index
        ds_idx = rec.get("dataset_index", None)
        if isinstance(ds_idx, int):
            by_ds_idx[ds_idx] = {"gv": gv, "rec": rec}
            ok += 1
        # composite
        by_comp[composite_key(rec)] = {"gv": gv, "rec": rec}
        ok += 1
    print(f"[INFO] JSONL loaded: {total} lines, valid MCQ: ~{ok//2} (indexed by ds_idx & composite).")
    return by_ds_idx, by_comp

def index_parquet_dir(parquet_dir: str):
    """把原始 parquet 做索引。"""
    files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    by_ds_idx: Dict[int, Dict[str, Any]] = {}
    by_comp: Dict[Tuple, Dict[str, Any]] = {}
    scanned = 0

    for fp in files:
        tbl = pq.read_table(fp)
        n = tbl.num_rows
        scanned += n

        cols = {name: tbl[name] for name in tbl.column_names}
        has_ds = "dataset_index" in cols

        aid = cols.get("article_accession_id")
        img_ids = cols.get("image_id")
        img_names = cols.get("image_file_name")

        for i in range(n):
            entry = {
                "images": cols["images"][i].as_py() if "images" in cols else None,
                "hash": (cols["hash"][i].as_py() if "hash" in cols else "") or "",
                "dataset_index": cols["dataset_index"][i].as_py() if "dataset_index" in cols else None,
            }
            if has_ds:
                di = entry["dataset_index"]
                if isinstance(di, int):
                    by_ds_idx[di] = entry
            if aid is not None and img_ids is not None and img_names is not None:
                key = (
                    aid[i].as_py(),
                    tuple(img_ids[i].as_py() or []),
                    tuple(img_names[i].as_py() or []),
                )
                by_comp[key] = entry

    print(f"[INFO] Parquet indexed: {scanned} rows, keys: {len(by_ds_idx)} by ds_idx, {len(by_comp)} by composite.")
    return by_ds_idx, by_comp

# ---------- main transform ----------
def main():
    ap = argparse.ArgumentParser(description="Build eval-format parquet from filtered+rebalanced JSONL and original parquet dir.")
    ap.add_argument("--jsonl", required=True, help="Filtered & rebalanced JSONL (contains generated_vqa)")
    ap.add_argument("--parquet_dir", required=True, help="Original parquet folder")
    ap.add_argument("--out_parquet", required=True, help="Output parquet path")
    ap.add_argument("--dataset_name", default="biomedica_internvl3_verified_5K")
    args = ap.parse_args()

    j_by_ds, j_by_comp = load_jsonl_gv(args.jsonl)
    p_by_ds, p_by_comp = index_parquet_dir(args.parquet_dir)

    rows: List[Dict[str, Any]] = []
    taken = 0
    missed = 0

    # 逐条以 JSONL 为准，去原 parquet 抽取 images/hash/dataset_index
    for _, j in j_by_ds.items():
        gv = j["gv"]
        jr = j["rec"]
        # 1) by dataset_index
        entry = None
        ds = jr.get("dataset_index")
        if isinstance(ds, int):
            entry = p_by_ds.get(ds)
        # 2) fallback by composite
        if entry is None:
            entry = p_by_comp.get(composite_key(jr))

        if entry is None or entry.get("images") is None:
            missed += 1
            continue

        images_val = entry["images"]
        hash_val = entry.get("hash", "") or ""
        ds_idx_out = entry.get("dataset_index", ds if isinstance(ds, int) else None)
        if ds_idx_out is None:
            # 兜底，用累积序号
            ds_idx_out = taken

        options_str = json.dumps(gv["options"], ensure_ascii=False)
        rows.append({
            "images": images_val,
            "question": gv["question"],
            "options": options_str,                      # string
            "answer_label": gv["answer_label"],
            "answer": gv["answer"],
            "dataset_name": args.dataset_name,
            "hash": hash_val,
            "dataset_index": int(ds_idx_out),
            "reasoning": None,                           # Null 列
            "misc": "",                                  # string 列（空串）
        })
        taken += 1

    if taken == 0:
        raise RuntimeError("No matched rows. Check keys/dataset_index consistency.")

    print(f"[INFO] Matched/exported: {taken} rows; missed: {missed}")

    # ---- Build Arrow table with EXACT schema ----
    schema = pa.schema([
        pa.field("images", pa.list_(pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ]))),
        pa.field("question", pa.string()),
        pa.field("options", pa.string()),
        pa.field("answer_label", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("dataset_name", pa.string()),
        pa.field("hash", pa.string()),
        pa.field("dataset_index", pa.int64()),
        pa.field("reasoning", pa.null()),
        pa.field("misc", pa.string()),
    ])

    # 列装配
    images_arr     = pa.array([r["images"] for r in rows], type=schema.field("images").type)
    question_arr   = pa.array([r["question"] for r in rows], type=pa.string())
    options_arr    = pa.array([r["options"] for r in rows], type=pa.string())
    alabel_arr     = pa.array([r["answer_label"] for r in rows], type=pa.string())
    answer_arr     = pa.array([r["answer"] for r in rows], type=pa.string())
    dname_arr      = pa.array([r["dataset_name"] for r in rows], type=pa.string())
    hash_arr       = pa.array([r["hash"] for r in rows], type=pa.string())
    dsidx_arr      = pa.array([r["dataset_index"] for r in rows], type=pa.int64())
    reasoning_arr  = pa.nulls(len(rows))  # Null column
    misc_arr       = pa.array([r["misc"] for r in rows], type=pa.string())

    table = pa.Table.from_arrays(
        [images_arr, question_arr, options_arr, alabel_arr, answer_arr,
         dname_arr, hash_arr, dsidx_arr, reasoning_arr, misc_arr],
        schema=schema,
    )

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    print(f"[OK] Wrote -> {out_path}  (rows={table.num_rows})")


if __name__ == "__main__":
    main()