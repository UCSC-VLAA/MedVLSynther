#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from datasets import Dataset, Features, Image, Sequence, Value


def normalize_images_to_bytes_list(image_struct_list: Any) -> List[bytes]:
    """
    将“可能是 list[dict/bytes/path] 或 ndarray”的 images 统一为 list[bytes]。
    dict 形如 {"bytes": ..., "path": ...}；优先用 bytes，没有就从 path 读取。
    """
    if isinstance(image_struct_list, np.ndarray):
        image_struct_list = image_struct_list.tolist()
    if not isinstance(image_struct_list, list):
        return []

    out: List[bytes] = []
    for item in image_struct_list:
        try:
            if isinstance(item, dict):
                b = item.get("bytes")
                if b is not None:
                    if isinstance(b, memoryview):
                        b = b.tobytes()
                    elif isinstance(b, bytearray):
                        b = bytes(b)
                    if isinstance(b, bytes):
                        out.append(b)
                        continue
                p = item.get("path")
                if isinstance(p, str) and p:
                    try:
                        out.append(Path(p).read_bytes())
                        continue
                    except Exception as e:
                        print(f"[Warning] failed to read image from path '{p}': {e}")
                        continue

            if isinstance(item, (bytes, bytearray, memoryview)):
                out.append(bytes(item))
                continue

            if isinstance(item, str):
                p = Path(item)
                if p.exists():
                    try:
                        out.append(p.read_bytes())
                        continue
                    except Exception as e:
                        print(f"[Warning] failed to read image from path '{p}': {e}")
                        continue
        except Exception as e:
            print(f"[Warning] skip one image element due to error: {e}")
            continue

    return out


def convert_parquet(input_path: Path, output_path: Path, also_save_dir: Path | None = None):
    """
    将“格式一（images: List of {bytes,path}）”转为
    “格式二（images: List(Image(decode=True))）”，并写出 parquet。
    可选 also_save_dir 会额外保存一个带元数据的 HF dataset 目录（便于下游直接 load）。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {input_path}")

    print(f"Reading Parquet: {input_path}")
    table = pq.read_table(input_path)
    df = table.to_pandas()
    print(f"Loaded {len(df)} rows")

    # 1) 规范化 images -> list[bytes]
    print("Normalizing 'images' to List[bytes] ...")
    df["images"] = [normalize_images_to_bytes_list(v) for v in df["images"]]

    # 2) 规范化 reasoning：允许为空或字符串
    if "reasoning" in df.columns:
        def _fix_reasoning(x):
            # 将 NaN / 空串 -> None，其余统一本为 str
            if pd.isna(x):
                return None
            if isinstance(x, str):
                x2 = x.strip()
                return None if x2 == "" else x2
            return str(x)
        df["reasoning"] = df["reasoning"].map(_fix_reasoning)
    else:
        # 若没有该列，补一个全 None 的列
        df["reasoning"] = None

    # 3) 建立 HF Datasets，并声明正确的特征
    features = Features({
        "images": Sequence(feature=Image(decode=True)),   # 列表里的每个元素都是 Image
        "question": Value("string"),
        "options": Value("string"),
        "answer_label": Value("string"),
        "answer": Value("string"),
        "dataset_name": Value("string"),
        "hash": Value("string"),
        "dataset_index": Value("int64"),
        "reasoning": Value("string"),  # 允许 None（pyarrow string 支持 null）
        "misc": Value("string"),
    })

    ds = Dataset.from_pandas(df, features=features, preserve_index=False)

    # 简单校验：解码后应为 PIL.Image
    if len(ds) and len(ds[0]["images"]):
        print("features ->", ds.features)
        print("type(ds[0]['images'][0]) ->", type(ds[0]["images"][0]))

    # 写出 parquet（注意：parquet 不保留 HF 特征元数据）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Parquet to: {output_path}")
    ds.to_parquet(output_path)
    print("Parquet written.")

    # 可选：同时保存带元数据的版本，便于下游直接 load_from_disk
    if also_save_dir is not None:
        also_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Also saving a metadata-preserving copy to: {also_save_dir}")
        ds.save_to_disk(str(also_save_dir))
        print("Saved with metadata.")


def main():
    ap = argparse.ArgumentParser(
        description="Convert Parquet (images as List of {bytes,path}) to List(Image(decode=True))",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-i", "--input", type=Path, required=True, help="Input Parquet path (Format 1)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output Parquet path (Format 2 data)")
    ap.add_argument("--also-save-dir", type=Path, default=None,
                    help="Optional: directory for HF dataset with preserved metadata (save_to_disk)")
    args = ap.parse_args()

    convert_parquet(args.input, args.output, args.also_save_dir)


if __name__ == "__main__":
    main()
