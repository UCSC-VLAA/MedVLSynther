# analysis/metadata_viewer.py
# -*- coding: utf-8 -*-
import dotenv
dotenv.load_dotenv(override=True)

import os
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd

import datasets
import streamlit as st
from PIL import Image


# ============================ Page Config ============================
st.set_page_config(page_title="Med-VLM Metadata Viewer", layout="wide")


# ============================ Helpers ============================

def _normalize_list_str(x) -> List[str]:
    """ensure list[str]"""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]

def _safe_get_list(row: Dict[str, Any], key: str) -> List[Any]:
    v = row.get(key, [])
    return v if isinstance(v, list) else []

def _safe_index(lst: List[Any], idx: int, default=None):
    try:
        return lst[idx]
    except Exception:
        return default

def make_composite_key(rec: Dict[str, Any]) -> Tuple:
    """
    composite key: article_accession_id + tuple(image_id) + tuple(image_file_name)
    both (Parquet/JSONL) follow this rule
    """
    aid = rec.get("article_accession_id", None)
    img_ids = tuple(_normalize_list_str(rec.get("image_id", [])))
    img_names = tuple(_normalize_list_str(rec.get("image_file_name", [])))
    return (aid, img_ids, img_names)


def to_pil_from_parquet_image_entry(entry: Any) -> Optional[Image.Image]:
    try:
        if isinstance(entry, dict):
            b = entry.get("bytes", None)
            if b is not None:
                if isinstance(b, memoryview):
                    b = b.tobytes()
                elif not isinstance(b, (bytes, bytearray)):
                    b = bytes(b)
                return Image.open(io.BytesIO(b)).convert("RGB")
            p = entry.get("path", None)
            if isinstance(p, str) and os.path.exists(p):
                return Image.open(p).convert("RGB")
        if isinstance(entry, str) and os.path.exists(entry):
            return Image.open(entry).convert("RGB")
    except Exception:
        return None
    return None


def prettify_json(obj: Any, drop_keys: Optional[List[str]] = None) -> str:
    if drop_keys and isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k not in set(drop_keys)}
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def jsonl_signature(path: str, num_lines_hint: int) -> Tuple[int, int, int]:
    p = Path(path)
    try:
        st_ = p.stat()
        return (st_.st_size, int(st_.st_mtime), int(num_lines_hint))
    except Exception:
        return (0, 0, int(num_lines_hint))


# ============================ Loaders (cached) ============================

@st.cache_data(show_spinner=False)
def load_parquet_split(parquet_path_or_glob: str) -> datasets.Dataset:
    p = Path(parquet_path_or_glob)
    files: List[str] = []
    if p.exists() and p.is_dir():
        files = sorted(str(pp) for pp in p.glob("*.parquet"))
    else:
        from glob import glob
        files = sorted(glob(parquet_path_or_glob))

    if not files:
        raise FileNotFoundError(f"No parquet files matched: {parquet_path_or_glob}")

    ds_dict = datasets.load_dataset("parquet", data_files={"data": files})
    return ds_dict["data"]


@st.cache_data(show_spinner=False)
def load_jsonl_records(jsonl_path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


@st.cache_data(show_spinner=False)
def build_indices(
    _parquet_ds: datasets.Dataset,      
    _jsonl_recs: List[Dict[str, Any]],  
    parquet_fp: str,                    
    jsonl_sig: Tuple[int, int, int], 
):
    """
    index: 
      - parquet_key2idx: composite key -> parquet line number
      - jsonl_key2rec:   composite key -> JSONL record
      - jsonl_idx2rec:   dataset_index -> JSONL record
      - matched_keys:    intersection
    """
    parquet_ds = _parquet_ds
    jsonl_recs = _jsonl_recs

    parquet_key2idx: Dict[Tuple, int] = {}
    for i in range(len(parquet_ds)):
        row = parquet_ds[i]
        key = make_composite_key(row)
        if key not in parquet_key2idx: 
            parquet_key2idx[key] = i

    jsonl_key2rec: Dict[Tuple, Dict[str, Any]] = {}
    jsonl_idx2rec: Dict[int, Dict[str, Any]] = {}
    for rec in jsonl_recs:
        key = make_composite_key(rec)
        if key not in jsonl_key2rec:
            jsonl_key2rec[key] = rec
        ds_idx = rec.get("dataset_index", None)
        if isinstance(ds_idx, int) and ds_idx not in jsonl_idx2rec:
            jsonl_idx2rec[ds_idx] = rec

    matched_keys = sorted(set(parquet_key2idx.keys()) & set(jsonl_key2rec.keys()),
                          key=lambda k: parquet_key2idx[k])

    return parquet_key2idx, jsonl_key2rec, jsonl_idx2rec, matched_keys


# ============================ Sidebar: Inputs ============================

st.sidebar.header("⚙️ 数据源参数")

parquet_input = st.sidebar.text_input(
    "Parquet 路径/通配符/目录",
    "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_VQA_parquet_25k/*.parquet",
)

jsonl_input = st.sidebar.text_input(
    "JSONL（rubric/验证元数据）路径",
    "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_score_filtered.jsonl",
)

# jsonl_input = st.sidebar.text_input(
#     "JSONL（rubric/验证元数据）路径",
#     "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_filtered_mean9670_13k_rebalanced_subset10k.jsonl",
# )


# 控件：按 article_accession_id 过滤、分页、跳转 dataset_index
st.sidebar.header("🔍 过滤与跳转")
filter_aid = st.sidebar.text_input("按 article_accession_id 精确过滤（可留空）", value="")
rows_per_page = st.sidebar.slider("每页条数", min_value=1, max_value=40, value=8, step=1)
goto_ds_index = st.sidebar.text_input("跳转 dataset_index（来自 JSONL）", value="")
go_button = st.sidebar.button("🚀 加载 / 刷新")

# —— 默认状态，避免 KeyError —— #
for k, v in [
    ("parquet_ds", None),
    ("jsonl_recs", []),
    ("parquet_key2idx", {}),
    ("jsonl_key2rec", {}),
    ("jsonl_idx2rec", {}),
    ("matched_keys", []),
    ("filtered_keys", []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# 首次自动加载（当路径存在时）
auto_boot = ("_boot" not in st.session_state) and Path(parquet_input).parent.exists()
if auto_boot:
    st.session_state["_boot"] = True
    go_button = True

if go_button:
    with st.spinner("Loading parquet & jsonl ..."):
        try:
            st.session_state.parquet_ds = load_parquet_split(parquet_input)
            st.session_state.jsonl_recs = load_jsonl_records(jsonl_input)

            # 可哈希指纹
            parquet_fp = getattr(st.session_state.parquet_ds, "_fingerprint", "") or ""
            jsonl_sig = jsonl_signature(jsonl_input, num_lines_hint=len(st.session_state.jsonl_recs))

            (st.session_state.parquet_key2idx,
             st.session_state.jsonl_key2rec,
             st.session_state.jsonl_idx2rec,
             st.session_state.matched_keys) = build_indices(
                 st.session_state.parquet_ds,
                 st.session_state.jsonl_recs,
                 parquet_fp,
                 jsonl_sig,
             )
            # 过滤
            if filter_aid:
                st.session_state.filtered_keys = [
                    k for k in st.session_state.matched_keys if (k[0] == filter_aid)
                ]
            else:
                st.session_state.filtered_keys = list(st.session_state.matched_keys)

        except Exception as e:
            st.error(f"加载失败：{e}")
            st.session_state.matched_keys = []
            st.session_state.filtered_keys = []

# Early exit
if st.session_state.parquet_ds is None:
    st.info("⬅️ 在左侧填入 Parquet 与 JSONL 路径后点击『加载 / 刷新』。")
    st.stop()


# ====== Status metrics ======
with st.sidebar.expander("状态 / 统计", expanded=True):
    try:
        st.markdown(f"- Parquet 条数：`{len(st.session_state.parquet_ds)}`")
    except Exception:
        st.markdown(f"- Parquet 条数：`?`")
    st.markdown(f"- JSONL 条数：`{len(st.session_state.jsonl_recs)}`")
    st.markdown(f"- 复合键配对条数：`{len(st.session_state.matched_keys)}`")
    if filter_aid:
        st.markdown(f"- 过滤后条数：`{len(st.session_state.filtered_keys)}`")


# ============================ Rendering ============================

def render_pair_by_key(key: Tuple):
    """按复合键渲染一条配对记录（逐图展示 caption/context 对齐）。"""
    parquet_idx = st.session_state.parquet_key2idx.get(key, None)
    jsonl_rec = st.session_state.jsonl_key2rec.get(key, None)
    if parquet_idx is None or jsonl_rec is None:
        st.warning("未找到完整配对记录。")
        return

    row = st.session_state.parquet_ds[parquet_idx]

    # 顶部概览
    st.markdown(
        f"#### Pair • Parquet idx: `{parquet_idx}` • "
        f"article_accession_id: `{key[0]}`"
    )
    st.markdown(f"**Article title:** {row.get('article_title', '')}")

    # 逐图对齐展示
    imgs = _safe_get_list(row, "images")
    caps = _safe_get_list(row, "caption")                  # list[str]
    ctxs = _safe_get_list(row, "context")                  # list[list[str]]
    img_ids = _normalize_list_str(row.get("image_id", []))
    img_names = _normalize_list_str(row.get("image_file_name", []))

    if imgs:
        st.markdown(f"**Images:** {len(imgs)} 张")
        for i, ent in enumerate(imgs):
            with st.container():
                ic, tc = st.columns([1, 2])

                with ic:
                    pil = to_pil_from_parquet_image_entry(ent)
                    if pil is not None:
                        st.image(pil, use_container_width=True)
                    else:
                        st.caption("（无法渲染该图片）")

                with tc:
                    this_img_id = _safe_index(img_ids, i, "<none>")
                    this_img_name = _safe_index(img_names, i, "<none>")
                    this_cap = _safe_index(caps, i, "<none>")
                    raw_ctx = _safe_index(ctxs, i, [])
                    this_ctx_list = raw_ctx if isinstance(raw_ctx, list) else [str(raw_ctx)] if raw_ctx is not None else []

                    st.markdown(f"**Image #{i+1}** — image_id: `{this_img_id}` • file: `{this_img_name}`")
                    st.markdown(f"**Caption:** {this_cap if this_cap is not None else '<none>'}")

                    if this_ctx_list:
                        with st.expander("Context（展开查看）", expanded=False):
                            for j, para in enumerate(this_ctx_list, 1):
                                st.markdown(f"{j}. {para}")
                    else:
                        st.markdown("**Context:** <none>")

            st.divider()
    else:
        st.caption("无图片")

    # JSONL 的 MCQ（若有）
    gen_vqa = jsonl_rec.get("generated_vqa") or row.get("generated_vqa") or {}
    if gen_vqa:
        st.markdown("### Generated VQA（from JSONL if available）")
        q = gen_vqa.get("question", "")
        st.markdown(f"- **Q:** {q}")
        opts = gen_vqa.get("options", {})
        if isinstance(opts, dict) and opts:
            st.markdown("- **Options:**")
            for k in ["A", "B", "C", "D", "E"]:
                if k in opts:
                    st.markdown(f"  - **{k}.** {opts[k]}")
        ans = gen_vqa.get("answer", None)
        if ans is not None:
            st.markdown(f"- **Answer:** **{ans}**")

    # JSONL 扩展元数据（除 raw_model_output 外全部展示）
    st.markdown("---")
    st.markdown("**JSONL 扩展元数据**（已隐藏 raw_model_output）")

    ds_idx = jsonl_rec.get("dataset_index", None)
    st.markdown(f"- **dataset_index**: `{ds_idx}`")

    # verify_pass 绿色/红色高亮
    vp = jsonl_rec.get("verify_pass", None)
    vs = jsonl_rec.get("verify_score", None)
    try:
        # 新版 Streamlit 支持 :green[] / :red[] 语法
        if isinstance(vp, bool):
            st.markdown(f"- **verify_pass:** {':green[True]' if vp else ':red[False]'}")
        elif vp is not None:
            st.markdown(f"- **verify_pass:** `{vp}`")
    except Exception:
        # 旧版回退
        if isinstance(vp, bool):
            (st.success if vp else st.error)(f"verify_pass: {vp}")
        elif vp is not None:
            st.markdown(f"- **verify_pass:** `{vp}`")

    if vs is not None:
        st.markdown(f"- **verify_score:** `{vs}`")

    # Rubric 表格（DataFrame 渲染）
    rub = (jsonl_rec.get("rubric_verification") or {}).get("rubric", None)
    if isinstance(rub, list) and rub:
        st.markdown("**Rubric 明细**")
        rub_rows = []
        for r in rub:
            rub_rows.append({
                "idx": r.get("idx", ""),
                "category": r.get("category", ""),
                "weight": r.get("weight", ""),
                "score": r.get("score", ""),
                "title": r.get("title", ""),
                "notes": r.get("notes", ""),
            })
        rub_df = pd.DataFrame(rub_rows, columns=["idx","category","weight","score","title","notes"])
        # Streamlit 老版本没有 hide_index 参数，这里统一 reset_index 以免出现双索引
        rub_df = rub_df.reset_index(drop=True)
        st.dataframe(rub_df, use_container_width=True)

    # 其余 JSONL 字段（隐藏 raw_model_output）
    with st.expander("查看 JSONL 原始记录（已去 raw_model_output）", expanded=False):
        st.code(prettify_json(jsonl_rec, drop_keys=["raw_model_output"]), language="json")


# ============================ Jump by dataset_index ============================

if goto_ds_index.strip():
    try:
        target_idx = int(goto_ds_index.strip())
        rec = st.session_state.jsonl_idx2rec.get(target_idx, None)
        if rec is None:
            st.error(f"未在 JSONL 中找到 dataset_index={target_idx} 的记录。")
        else:
            key = make_composite_key(rec)
            if key not in st.session_state.parquet_key2idx:
                st.warning("已找到 JSONL 记录，但在 Parquet 中未找到同键记录（复合键不匹配）。")
            render_pair_by_key(key)
        st.stop()
    except ValueError:
        st.error("dataset_index 请输入整数。")
        st.stop()


# ============================ Paged Browse ============================

keys = st.session_state.filtered_keys or st.session_state.matched_keys or []
total = len(keys)
if total == 0:
    st.warning("当前筛选条件下无配对记录。请尝试清空过滤条件或检查路径&键是否一致。")
    st.stop()

total_pages = max(1, math.ceil(total / rows_per_page))
page_num = st.sidebar.number_input("页码", min_value=1, max_value=total_pages, value=1, step=1)

start = (page_num - 1) * rows_per_page
end = min(start + rows_per_page, total)
show_keys = keys[start:end]

st.markdown(f"### 显示配对记录 {start}–{end-1} / 共 {total} 条（第 {page_num}/{total_pages} 页）")
if filter_aid:
    st.caption(f"过滤条件：article_accession_id = `{filter_aid}`")

for k in show_keys:
    render_pair_by_key(k)

st.caption("Med-VLM metadata viewer • Streamlit")
