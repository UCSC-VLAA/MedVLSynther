#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, json, tarfile, argparse, math
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# -------------------- image I/O --------------------
def _open_member_bytes(tar_abs_path: str, member_path: str) -> bytes:
    # 为了简单和稳妥：每次打开-读取-关闭；对大规模可改成进程内 LRU 缓存
    with tarfile.open(tar_abs_path, "r") as tar:
        m = tar.getmember(member_path)
        return tar.extractfile(m).read()

def _load_image_from_wds(base_dir: str, rec: dict, idx: int) -> Tuple[bytes, str]:
    """
    返回 (bytes, path_repr). path_repr 统一成 "tar::member" 或文件系统路径。
    优先级:
      1) image_wds_tar[idx] + image_wds_member[idx]
      2) image_wds_path[idx] (形如 tar::member)
      3) image_path[idx]（若存在且可直接 open）
    """
    # 1) tar + member
    tar_rel = (rec.get("image_wds_tar") or [None])[idx]
    mem_rel = (rec.get("image_wds_member") or [None])[idx]
    if tar_rel and mem_rel:
        tar_abs = os.path.join(base_dir, tar_rel)
        b = _open_member_bytes(tar_abs, mem_rel)
        return b, f"{tar_rel}::{mem_rel}"

    # 2) tar::member
    wds_path = (rec.get("image_wds_path") or [None])[idx]
    if wds_path and "::" in wds_path:
        tar_rel2, mem_rel2 = wds_path.split("::", 1)
        tar_abs2 = os.path.join(base_dir, tar_rel2)
        b = _open_member_bytes(tar_abs2, mem_rel2)
        return b, wds_path

    # 3) fallback：直接文件系统路径（极少用）
    img_fs_path = (rec.get("image_path") or [None])[idx]
    if img_fs_path and os.path.exists(img_fs_path):
        with open(img_fs_path, "rb") as f:
            return f.read(), img_fs_path

    raise FileNotFoundError(f"Cannot locate image bytes for idx={idx}")

def _resize_long_side(img: Image.Image, target_long: int = 384) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side <= target_long:
        return img
    scale = target_long / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def _encode_bytes(img: Image.Image, fmt: str = "PNG", quality: int = 90) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "PNG":
        img.save(buf, format="PNG")
    elif fmt.upper() in ("JPG", "JPEG"):
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        raise ValueError("Unsupported format: " + fmt)
    return buf.getvalue()

# -------------------- per-record worker --------------------
def _process_one_record(task: Tuple[int, dict, str, int, str, int, bool, bool]) -> Tuple[int, Dict[str, Any], Dict[str, int]]:
    """
    进程内处理一条 JSON 记录：
    - 读取每张图
    - 缩放到指定长边（不放大）
    - 编码成 PNG/JPEG bytes
    - 产出 new_rec（保留原字段 + images + image_size_resized）
    - 返回 (原索引, new_rec 或 None（被丢弃）, 统计计数)
    """
    idx, rec, base_dir, long_side, fmt, quality, skip_missing, drop_empty = task

    ids = rec.get("image_id", []) or []
    n = len(ids)
    images_struct: List[Dict[str, Any]] = []
    image_size_resized: List[List[int]] = []
    kept_any = False

    total, resized, missing = 0, 0, 0

    for i in range(n):
        total += 1
        try:
            b, path_repr = _load_image_from_wds(base_dir, rec, i)
            im = Image.open(io.BytesIO(b))
            im2 = _resize_long_side(im, long_side)
            enc = _encode_bytes(im2, fmt=fmt, quality=quality)
            images_struct.append({"bytes": enc, "path": path_repr})
            image_size_resized.append(list(im2.size))
            resized += 1
            kept_any = True
        except Exception:
            missing += 1
            if skip_missing:
                # 丢弃这张图（不占位）
                continue
            else:
                # 占位（None/None）
                images_struct.append({"bytes": None, "path": None})
                image_size_resized.append([None, None])

    if drop_empty and not kept_any:
        # 丢弃整条记录
        return idx, None, {"total": total, "resized": resized, "missing": missing, "dropped": 1}

    new_rec = dict(rec)
    new_rec["images"] = images_struct
    new_rec["image_size_resized"] = image_size_resized
    return idx, new_rec, {"total": total, "resized": resized, "missing": missing, "dropped": 0}

# -------------------- build Arrow --------------------
def _to_arrow_table(records: List[Dict[str, Any]],
                    images_col_name: str = "images",
                    keep_cols: List[str] = None) -> pa.Table:
    """
    records 中必须已包含 images: list[{'bytes': binary, 'path': string}]
    其他列直接让 Arrow 推断（不要包含 PIL 对象）。
    """
    if keep_cols is None:
        keep_cols = sorted({k for rec in records for k in rec.keys()})

    images_type = pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    images_py = [rec.get(images_col_name, []) for rec in records]
    images_col = pa.array(images_py, type=images_type)

    cols = [images_col]
    names = [images_col_name]
    for name in keep_cols:
        if name == images_col_name:
            continue
        arr = pa.array([rec.get(name, None) for rec in records])
        cols.append(arr)
        names.append(name)
    return pa.table(cols, names=names)

# -------------------- main pipeline --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Parallel: JSON → load/resize images → pack images struct → write Parquet (optional shards)"
    )
    ap.add_argument("--base", required=True, help="WebDataset 根目录（含 commercial/noncommercial/other）")
    ap.add_argument("--input", required=True, help="抽样生成的 JSON 路径")
    ap.add_argument("--output", required=True, help="输出 Parquet 路径或前缀（若分片）")
    ap.add_argument("--long-side", type=int, default=384, help="长边目标（仅缩小，不放大）")
    ap.add_argument("--format", default="PNG", choices=["PNG", "JPG", "JPEG"], help="输出图像编码格式")
    ap.add_argument("--quality", type=int, default=90, help="JPEG 质量（仅当 format=JPG/JPEG 时有效）")
    ap.add_argument("--skip_missing", action="store_true", help="遇到缺图：跳过该图（否则保留为 None，并占位）")
    ap.add_argument("--drop_empty_record", action="store_true", help="若一条记录所有图均缺失，则丢弃该记录")
    ap.add_argument("--rows-per-shard", type=int, default=0, help=">0 则分片写出：prefix-00000-of-N.parquet")
    ap.add_argument("--compression", default="zstd", choices=["zstd","snappy","gzip","brotli","none"], help="Parquet 压缩")
    ap.add_argument("--row-group-size", type=int, default=4096, help="Parquet row group size")
    # 新增并行参数（其余保持不变）
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="并行进程数（默认 CPU 核心数）")
    ap.add_argument("--chunksize", type=int, default=8, help="map 投喂粒度（提高吞吐）")
    args = ap.parse_args()

    # 读取 JSON
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- 并行处理每条记录 ---
    tasks = [
        (i, rec, args.base, args.long_side, args.format, args.quality, args.skip_missing, args.drop_empty_record)
        for i, rec in enumerate(data)
    ]

    new_records_ordered: List[Dict[str, Any]] = [None] * len(tasks)
    total_imgs = resized_imgs = missing_imgs = dropped_records = 0

    # 使用进程池并行：保证稳定顺序（按索引回填）
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for idx, new_rec, cnt in tqdm(
            ex.map(_process_one_record, tasks, chunksize=args.chunksize),
            total=len(tasks), desc="Processing (parallel)"
        ):
            total_imgs   += cnt["total"]
            resized_imgs += cnt["resized"]
            missing_imgs += cnt["missing"]
            dropped_records += cnt["dropped"]
            if new_rec is not None:
                new_records_ordered[idx] = new_rec

    # 过滤掉被丢弃的
    out_records = [r for r in new_records_ordered if r is not None]

    if not out_records:
        raise RuntimeError("No records to write (all dropped?).")

    # Arrow 表（images 列 + 其他列，保持 images 为第一列）
    keep_cols = sorted({k for r in out_records for k in r.keys()})
    table = _to_arrow_table(out_records, images_col_name="images", keep_cols=keep_cols)

    # 写出
    compression = None if args.compression == "none" else args.compression
    writer_kwargs = dict(compression=compression, use_dictionary=True, write_statistics=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.rows_per_shard and args.rows_per_shard > 0:
        n = len(table)
        shards = math.ceil(n / args.rows_per_shard)
        for i in range(shards):
            lo = i * args.rows_per_shard
            hi = min((i + 1) * args.rows_per_shard, n)
            shard = table.slice(lo, hi - lo)
            out_path = f"{args.output}-{i:05d}-of-{shards:05d}.parquet"
            pq.write_table(shard, out_path, row_group_size=args.row_group_size, **writer_kwargs)
            print(f"[write] {out_path} rows={len(shard)}")
        print(f"Done. Shards: {shards}")
        schema_preview_path = f"{args.output}-00000-of-{shards:05d}.parquet"
        print("\nSchema preview:")
        print(pq.ParquetFile(schema_preview_path).schema)
    else:
        pq.write_table(table, args.output, row_group_size=args.row_group_size, **writer_kwargs)
        print(f"Done. Wrote: {args.output} rows={len(table)}")
        print("\nSchema preview:")
        print(pq.ParquetFile(args.output).schema)

    # 汇总
    print("\n=== Summary ===")
    print(f"Records in JSON     : {len(data)}")
    print(f"Records written     : {len(out_records)} (dropped {dropped_records})")
    print(f"Images total        : {total_imgs}")
    print(f"Images resized/kept : {resized_imgs}")
    print(f"Images missing      : {missing_imgs}")
    print(f"Format              : {args.format}  (quality={args.quality if args.format.upper() in ('JPG','JPEG') else '-'})")
    print(f"Workers             : {args.workers}  (chunksize={args.chunksize})")

if __name__ == "__main__":
    # macOS 下多进程建议保护 main；Linux 也没问题
    mp.freeze_support()
    main()
