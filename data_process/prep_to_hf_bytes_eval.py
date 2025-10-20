#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import glob
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset, Dataset
from datasets import Sequence as HFSequence
from datasets import Image as HFImage
from PIL import Image

# ---------------- utils ----------------
_PNG_OK_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "I", "F"}  # PNG 能直接保存的常见模式

def _pil_to_png_bytes_safe(im: Image.Image) -> Optional[bytes]:
    """尽量以 PNG 写出；若模式不兼容，先转换；失败再尝试 JPEG 作为兜底。"""
    img = im
    try:
        if img.mode not in _PNG_OK_MODES:
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        try:
            img = im
            if img.mode not in {"L", "RGB"}:
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            return buf.getvalue()
        except Exception:
            return None

def collect_parquet_files(spec: str) -> List[str]:
    """支持：绝对/相对通配、目录、单文件、逗号拼接"""
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    out: List[str] = []
    for part in parts:
        if any(ch in part for ch in "*?[]"):
            out.extend(sorted(glob.glob(part)))
        else:
            p = Path(part)
            if p.is_dir():
                out.extend(str(x) for x in sorted(p.rglob("*.parquet")))
            elif p.is_file():
                out.append(str(p))
            else:
                out.extend(sorted(glob.glob(part)))
    # 去重保序
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            uniq.append(f); seen.add(f)
    return uniq

# ---- bytes-like 统一落地为 bytes ----
def _coerce_to_bytes(obj) -> Optional[bytes]:
    """把各种 bytes-like（bytes/bytearray/pyarrow.Buffer/memoryview/np.ndarray 等）统一转成 bytes。"""
    if obj is None:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    # pyarrow.Buffer
    try:
        import pyarrow as pa  # 懒加载
        if isinstance(obj, pa.Buffer):
            return obj.to_pybytes()
    except Exception:
        pass
    # memoryview
    if isinstance(obj, memoryview):
        try:
            return obj.tobytes()
        except Exception:
            return None
    # numpy 或其他带 .tobytes() 的对象
    if hasattr(obj, "tobytes"):
        try:
            return obj.tobytes()
        except Exception:
            pass
    # list[int 0..255]
    if isinstance(obj, list) and obj and all(isinstance(x, int) and 0 <= x < 256 for x in obj):
        try:
            return bytes(obj)
        except Exception:
            pass
    return None

def _bytes_from_possible_fields(d: dict) -> Optional[bytes]:
    """从常见字段抓取字节：bytes / hex / base64 / bytes_list / path"""
    # 1) bytes-like
    b = _coerce_to_bytes(d.get("bytes", None))
    if b and len(b) > 0:
        return b

    # 2) 十六进制
    hx = d.get("hex") or d.get("bytes_hex")
    if isinstance(hx, str) and hx:
        try:
            s = hx.strip().lower()
            if s.startswith("0x"):
                s = s[2:]
            s = "".join(s.split())
            b2 = bytes.fromhex(s)
            if b2:
                return b2
        except Exception:
            pass

    # 3) base64
    b64 = d.get("b64") or d.get("bytes_b64") or d.get("base64") or d.get("bytes_base64")
    if isinstance(b64, str) and b64:
        try:
            b3 = base64.b64decode(b64, validate=False)
            if b3:
                return b3
        except Exception:
            pass

    # 4) list[int] / bytes_array
    arr = d.get("bytes_list") or d.get("array") or d.get("bytes_array")
    if isinstance(arr, list) and arr and all(isinstance(x, int) and 0 <= x < 256 for x in arr):
        try:
            b4 = bytes(arr)
            if b4:
                return b4
        except Exception:
            pass

    # 5) path
    p = d.get("path", None)
    if p:
        try:
            with open(p, "rb") as f:
                b5 = f.read()
                if b5:
                    return b5
        except Exception:
            try:
                im = Image.open(p)
                b6 = _pil_to_png_bytes_safe(im)
                if b6:
                    return b6
            except Exception:
                return None

    return None

def to_image_bytes_dict(item: Any) -> Optional[Dict[str, Optional[bytes]]]:
    """
    统一为 HF Image 接受的字典：{"bytes": <bytes>, "path": None}
    - 优先保留已有字节（原格式，不强制转 PNG）
    - 读不到再尝试 path / hex / base64
    - 最后兜底：如果是 PIL.Image 就转 PNG/JPEG
    """
    if isinstance(item, dict):
        by = _bytes_from_possible_fields(item)
        return {"bytes": by, "path": None} if by else None

    # 直接 bytes-like
    by2 = _coerce_to_bytes(item)
    if by2 and len(by2) > 0:
        return {"bytes": by2, "path": None}

    # PIL.Image
    if isinstance(item, Image.Image):
        by3 = _pil_to_png_bytes_safe(item)
        return {"bytes": by3, "path": None} if by3 else None

    return None

def map_row(ex: Dict[str, Any], *, strict_image: bool, keep_first_k: Optional[int]) -> Dict[str, Any]:
    # images -> list[{'bytes', 'path(None)'}]
    imgs = ex.get("images", None)
    out_imgs: List[Dict[str, Optional[bytes]]] = []
    if isinstance(imgs, list):
        iterable = imgs if keep_first_k is None else imgs[:keep_first_k]
        for it in iterable:
            d = to_image_bytes_dict(it)
            if d is not None and isinstance(d.get("bytes", None), (bytes, bytearray)):
                out_imgs.append(d)

    if strict_image and len(out_imgs) == 0:
        # 返回空字典，供后续 filter 剔除该样本
        return {}

    ex["images"] = out_imgs

    # options -> json string
    opts = ex.get("options", None)
    if isinstance(opts, dict):
        ex["options"] = json.dumps(opts, ensure_ascii=False)
    elif isinstance(opts, str):
        ex["options"] = opts
    else:
        ex["options"] = json.dumps({}, ensure_ascii=False)

    # 填充 reasoning/hash/misc
    ex["reasoning"] = ex.get("reasoning", None)
    if "hash" not in ex:
        ex["hash"] = None
    misc = ex.get("misc", None)
    if isinstance(misc, dict) or misc is None:
        ex["misc"] = json.dumps(misc or {}, ensure_ascii=False)
    elif not isinstance(misc, str):
        ex["misc"] = json.dumps({}, ensure_ascii=False)

    return ex

# ---------------- main ----------------
# 为了兼容多进程 map，这里用模块级配置变量 + 顶层 mapper 函数
_MAP_STRICT_IMAGE = False
_MAP_KEEP_FIRST_K: Optional[int] = None

def _mapper(ex: Dict[str, Any]) -> Dict[str, Any]:
    return map_row(ex, strict_image=_MAP_STRICT_IMAGE, keep_first_k=_MAP_KEEP_FIRST_K)

def main():
    ap = argparse.ArgumentParser(description="Parquet -> HF (images saved as bytes).")
    ap.add_argument("--parquet_glob", required=True,
                    help="支持绝对/相对通配、目录、单文件、逗号拼接")
    ap.add_argument("--out_dir", required=True, help="保存为 datasets.save_to_disk 目录")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--strict_image", action="store_true",
                    help="严格模式：无任何有效图像的样本将被删除")
    ap.add_argument("--keep_first_k_images", type=int, default=None,
                    help="每样本最多保留前 K 张图，控制体积")
    args = ap.parse_args()

    files = collect_parquet_files(args.parquet_glob)
    if not files:
        raise FileNotFoundError(f"No parquet matched: {args.parquet_glob}")
    print(f"[info] matched {len(files)} parquet file(s).")

    # 读 parquet
    ds: Dataset = load_dataset("parquet", data_files=files, split="train")

    # ✅ 立刻把 images 关成原始字节视图（decode=False），避免自动解码成 PIL
    ds = ds.cast_column("images", HFSequence(HFImage(decode=False)))

    # 设置 mapper 配置（模块级变量，便于多进程序列化）
    global _MAP_STRICT_IMAGE, _MAP_KEEP_FIRST_K
    _MAP_STRICT_IMAGE = False  # 先不过滤
    _MAP_KEEP_FIRST_K = args.keep_first_k_images

    # 统一：images->bytes、options->json 等
    ds = ds.map(
        _mapper,
        num_proc=args.num_proc,
        desc="normalize images->bytes + options->json",
    )

    if args.strict_image:
        def _has_image_bytes(ex):
            imgs = ex.get("images", None)
            if not isinstance(imgs, list) or len(imgs) == 0:
                return False
            # 至少一张有“可判断的字节”
            for d in imgs:
                if not isinstance(d, dict):
                    continue
                b = d.get("bytes", None)
                if b is None:
                    continue
                # 直接 bytes-like
                if isinstance(b, (bytes, bytearray, memoryview)) and len(b) > 0:
                    return True
                # pyarrow.Buffer / 带 tobytes 的对象
                try:
                    import pyarrow as pa
                    if isinstance(b, pa.Buffer) and b.size > 0:
                        return True
                except Exception:
                    pass
                if hasattr(b, "tobytes"):
                    try:
                        if len(b.tobytes()) > 0:
                            return True
                    except Exception:
                        pass
            return False

        ds = ds.filter(_has_image_bytes, num_proc=args.num_proc,
                       desc="filter empty-image rows")

    # 再声明一遍：images 是 Sequence(Image(decode=False))，以 bytes 形式存储
    ds = ds.cast_column("images", HFSequence(HFImage(decode=False)))


    # 实体化索引，避免 save_to_disk() 在 _estimate_nbytes 中对 0 除
    ds = ds.flatten_indices()

    # 额外安全检查
    n = len(ds)
    if n == 0:
        raise RuntimeError("After processing, dataset has 0 rows — nothing to save.")
    print(f"[info] final dataset size: {n}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print(f"[OK] Saved HF dataset to: {args.out_dir}")
    print("Tip: 读取后若要直接访问 bytes，请先 cast：")
    print("  ds = load_from_disk(...).cast_column('images', datasets.Sequence(datasets.Image(decode=False)))")

if __name__ == "__main__":
    main()