#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import argparse, json, io, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Dataset
from datasets import Sequence as HFSequence
from datasets import Image as HFImage
from PIL import Image

def pil_from_item(it: Any) -> Optional[Image.Image]:
    # 支持 {'bytes':..., 'path':...} 或 直接 bytes
    if isinstance(it, dict):
        b = it.get("bytes", None)
        p = it.get("path", None)
        if b is not None:
            try:
                return Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                return None
        if p:
            # 有些是 tar::member 形式，通常打不开；这里尝试一次
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                return None
        return None
    elif isinstance(it, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(it)).convert("RGB")
        except Exception:
            return None
    return None

def map_row(ex: Dict[str, Any], *, strict_image: bool, keep_first_k: Optional[int]) -> Dict[str, Any]:
    # 1) images -> PIL 列表
    imgs = ex.get("images", None)
    pil_list: List[Image.Image] = []
    if isinstance(imgs, list):
        to_iter = imgs if keep_first_k is None else imgs[:keep_first_k]
        for it in to_iter:
            im = pil_from_item(it)
            if im is not None:
                pil_list.append(im)
    # 严格模式：一张都没有则丢样本（返回空 dict 让后续 filter 去掉）
    if strict_image and len(pil_list) == 0:
        return {}

    ex["images"] = pil_list

    # 2) options -> JSON 字符串
    opts = ex.get("options", None)
    if isinstance(opts, dict):
        ex["options"] = json.dumps(opts, ensure_ascii=False)
    elif isinstance(opts, str):
        # 假如已经是字符串，直接用
        ex["options"] = opts
    else:
        ex["options"] = json.dumps({}, ensure_ascii=False)

    # 3) 补齐 reasoning/hash/misc
    if ex.get("reasoning", None) is None:
        ex["reasoning"] = None
    if "hash" not in ex:
        ex["hash"] = None
    misc = ex.get("misc", None)
    if isinstance(misc, dict) or misc is None:
        ex["misc"] = json.dumps(misc or {}, ensure_ascii=False)
    elif not isinstance(misc, str):
        ex["misc"] = json.dumps({}, ensure_ascii=False)

    return ex

def collect_parquet_files(spec: str):
    """支持：
    - 绝对/相对通配: /a/b/*.parquet
    - 目录: /a/b   -> 递归找 *.parquet
    - 单文件: /a/b/c.parquet
    - 逗号分隔的多段: "/a/*.parquet,/b/foo.parquet,/c"
    """
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for part in parts:
        if any(ch in part for ch in "*?[]"):          # 有通配符 -> 用 glob
            out.extend(sorted(glob.glob(part)))
        else:
            p = Path(part)
            if p.is_dir():
                out.extend(str(x) for x in sorted(p.rglob("*.parquet")))
            elif p.is_file():
                out.append(str(p))
            else:
                # 既不是通配、也不是现存文件/目录：当作 glob 兜底
                out.extend(sorted(glob.glob(part)))
    # 去重并保持顺序
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

def main():
    ap = argparse.ArgumentParser(description="Parquet -> HF load_from_disk (images as bytes).")
    ap.add_argument("--parquet_glob", required=True,
                    help="parquet 路径或通配，如 '/path/*.parquet' 或 目录")
    ap.add_argument("--out_dir", required=True, help="保存为 HF 数据集目录（load_from_disk）")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--strict_image", action="store_true",
                    help="严格模式：没有成功恢复任意图像的样本将被丢弃")
    ap.add_argument("--keep_first_k_images", type=int, default=None,
                    help="每个样本最多保留前 K 张图（可控内存）")
    args = ap.parse_args()

    files = collect_parquet_files(args.parquet_glob)
    if not files:
        raise FileNotFoundError(f"No parquet matched: {args.parquet_glob}")
    print(f"[info] matched {len(files)} parquet files.")

    ds = load_dataset("parquet", data_files=files, split="train")
    ds = ds.map(
        lambda ex: map_row(ex, strict_image=False, keep_first_k=args.keep_first_k_images),
        num_proc=args.num_proc,
        desc="convert images+options",
    )

    if args.strict_image:
        ds = ds.filter(lambda ex: isinstance(ex.get("images", None), list) and len(ex["images"]) > 0,
                       num_proc=args.num_proc,
                       desc="filter empty-image rows")

    # 显式强制为 Sequence(Image)，确保保存为字节图像
    ds = ds.cast_column("images", HFSequence(HFImage()))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print(f"[OK] Saved HF dataset to: {args.out_dir}")

if __name__ == "__main__":
    main()