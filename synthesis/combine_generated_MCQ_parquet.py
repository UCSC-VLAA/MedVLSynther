#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional, List

from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


def iter_jsonl_files(path: str) -> Iterable[Path]:
    p = Path(path)
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.rglob("*.jsonl")):
            yield f
    else:
        # 允许通配
        for f in sorted(Path().glob(path)):
            if f.is_file() and f.suffix.lower() == ".jsonl":
                yield f


def make_composite_key(rec: Dict[str, Any]) -> Tuple:
    """组合键：article_accession_id + tuple(image_id) + tuple(image_file_name)"""
    aid = rec.get("article_accession_id", None)
    img_ids = tuple(rec.get("image_id", []) or [])
    img_names = tuple(rec.get("image_file_name", []) or [])
    return (aid, img_ids, img_names)


def normalize_generated_vqa(gv: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """清洗 generated_vqa -> {'question', 'options'{A..E}, 'answer'}；缺项补 None。"""
    if not isinstance(gv, dict):
        return None
    q = gv.get("question")
    ans = gv.get("answer")
    opts = gv.get("options") or {}
    options = {
        "A": opts.get("A"),
        "B": opts.get("B"),
        "C": opts.get("C"),
        "D": opts.get("D"),
        "E": opts.get("E"),
    }
    # question/answer 至少要是字符串（否则也写 None）
    q = str(q) if isinstance(q, str) else (None if q is None else str(q))
    ans = str(ans) if isinstance(ans, str) else (None if ans is None else str(ans))
    return {"question": q, "options": options, "answer": ans}


def pa_generated_vqa_array(items: List[Optional[Dict[str, Any]]]) -> pa.Array:
    """把 list[dict|None] 变成 Arrow struct 列（question/options/answer）。"""
    options_type = pa.struct([
        pa.field("A", pa.string()),
        pa.field("B", pa.string()),
        pa.field("C", pa.string()),
        pa.field("D", pa.string()),
        pa.field("E", pa.string()),
    ])
    gv_type = pa.struct([
        pa.field("question", pa.string()),
        pa.field("options", options_type),
        pa.field("answer", pa.string()),
    ])
    # Arrow 支持从 Python dict 直接构造 struct（缺失键视为 null）
    return pa.array(items, type=gv_type)


def load_generated_map(gen_path: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple, Dict[str, Any]]]:
    """
    读取 JSONL（可为目录/shards），建立两张表：
      - by_dataset_idx: dataset_index -> generated_vqa
      - by_composite:   (aid, image_id tuple, image_file_name tuple) -> generated_vqa
    """
    by_dataset_idx: Dict[int, Dict[str, Any]] = {}
    by_composite: Dict[Tuple, Dict[str, Any]] = {}

    total_lines = 0
    matched_lines = 0

    for jf in iter_jsonl_files(gen_path):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                gv = normalize_generated_vqa(rec.get("generated_vqa"))
                if gv is None:
                    # 没有或无法解析，跳过（保留空）
                    continue

                # 1) dataset_index
                ds_idx = rec.get("dataset_index", None)
                if isinstance(ds_idx, int):
                    by_dataset_idx[ds_idx] = gv
                    matched_lines += 1

                # 2) composite key
                key = make_composite_key(rec)
                by_composite[key] = gv
                matched_lines += 1

    print(f"[Info] Loaded generated_vqa: {len(by_dataset_idx)} by dataset_index, "
          f"{len(by_composite)} by composite key from {total_lines} jsonl lines.")
    return by_dataset_idx, by_composite


def merge_one_parquet(
    parq_path: Path,
    out_dir: Path,
    by_ds_idx: Dict[int, Dict[str, Any]],
    by_composite: Dict[Tuple, Dict[str, Any]],
) -> None:
    tbl = pq.read_table(parq_path)

    use_dataset_index = "dataset_index" in tbl.column_names
    n = tbl.num_rows

    # 为每一行找 generated_vqa
    merged: List[Optional[Dict[str, Any]]] = [None] * n

    if use_dataset_index and by_ds_idx:
        # 直接以 dataset_index 连接（最快）
        ds_col = tbl["dataset_index"].to_pylist()
        for i, ds_idx in enumerate(ds_col):
            gv = by_ds_idx.get(ds_idx)
            merged[i] = gv
    else:
        # 退化为组合键
        # 取需要的列
        aid_col = tbl["article_accession_id"] if "article_accession_id" in tbl.column_names else None
        img_id_col = tbl["image_id"] if "image_id" in tbl.column_names else None
        img_name_col = tbl["image_file_name"] if "image_file_name" in tbl.column_names else None

        if aid_col is None or img_id_col is None or img_name_col is None:
            raise ValueError(
                f"{parq_path} 缺少必要列用于匹配："
                f"{'article_accession_id' if aid_col is None else ''} "
                f"{'image_id' if img_id_col is None else ''} "
                f"{'image_file_name' if img_name_col is None else ''}"
            )

        aids = aid_col.to_pylist()
        img_ids = img_id_col.to_pylist()
        img_names = img_name_col.to_pylist()

        for i in range(n):
            key = (aids[i], tuple(img_ids[i] or []), tuple(img_names[i] or []))
            gv = by_composite.get(key)
            merged[i] = gv

    gv_array = pa_generated_vqa_array(merged)
    new_tbl = tbl.append_column("generated_vqa", gv_array)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / parq_path.name
    pq.write_table(new_tbl, out_path, compression="zstd")
    print(f"[OK] {parq_path.name} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Merge generated_vqa back into original parquet files.")
    ap.add_argument("--orig_parquet_dir", required=True, help="原始 parquet 文件夹（包含多个 .parquet）")
    ap.add_argument("--gen_jsonl_path", required=True,
                    help="推理输出：results.jsonl 或 shards 目录（会递归读取所有 .jsonl）")
    ap.add_argument("--out_dir", required=True, help="输出目录（写入附带 generated_vqa 的新 parquet）")
    args = ap.parse_args()

    by_ds_idx, by_comp = load_generated_map(args.gen_jsonl_path)

    parq_dir = Path(args.orig_parquet_dir)
    out_dir = Path(args.out_dir)
    files = sorted([p for p in parq_dir.glob("*.parquet")])

    if not files:
        raise FileNotFoundError(f"No parquet files found in {parq_dir}")

    for p in tqdm(files, desc="Merging"):
        merge_one_parquet(p, out_dir, by_ds_idx, by_comp)

    print(f"Done. Wrote {len(files)} parquet files -> {out_dir}")


if __name__ == "__main__":
    main()
