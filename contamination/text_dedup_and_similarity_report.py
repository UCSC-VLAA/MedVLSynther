#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip install faiss-cpu sentence-transformers pandas pyarrow datasets tqdm ujson rapidfuzz datasketch pillow
import argparse, os, glob, json, math, re, sys, ujson, io
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from rapidfuzz.distance.Levenshtein import normalized_similarity as lev_sim
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss

# ------------------------- Config -------------------------

EXPECTED_TEXT_FIELDS_DEFAULT = "question,options"

# ------------------------- Text utils -------------------------

def try_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            return ujson.loads(s)
        except Exception:
            return None

def stringify_options(opt_raw) -> str:
    """options 既可能是 JSON 字符串，也可能本身就是 dict。统一输出 'A. ...\\nB. ...' 形式。"""
    if isinstance(opt_raw, dict):
        parts = []
        for k in sorted(opt_raw.keys()):
            v = opt_raw.get(k)
            if v is None:
                continue
            parts.append(f"{k}. {str(v)}")
        return "\n".join(parts)
    if isinstance(opt_raw, str):
        obj = try_json_load(opt_raw)
        if isinstance(obj, dict):
            parts = []
            for k in sorted(obj.keys()):
                v = obj.get(k)
                if v is None:
                    continue
                parts.append(f"{k}. {str(v)}")
            if parts:
                return "\n".join(parts)
        return opt_raw.strip()
    return ""

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\.\,\-\+\(\)\/#:%;']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # 数字归一化（如不想要可注释）
    s = re.sub(r"\b\d+(\.\d+)?\b", "<NUM>", s)
    return s

def build_record_text(row: Dict[str, Any], fields: List[str]) -> str:
    parts = []
    for f in fields:
        val = row.get(f, "")
        if f == "options":
            val = stringify_options(val)
        if isinstance(val, str):
            parts.append(val)
    return normalize_text(" \n ".join([p for p in parts if p]))

def char_ngrams(s: str, n: int = 13) -> List[str]:
    s = f"^{s}$"
    if not s:
        return []
    L = len(s)
    return [s[i:i+n] for i in range(max(0, L-n+1))]

# ------------------------- IO: text -------------------------

def load_parquet_to_df(paths: List[str]) -> pd.DataFrame:
    """用 pandas.read_parquet 逐文件读取，仅选择文本列，避免 HF features cast 冲突。"""
    want_cols = [
        "question", "options", "answer_label", "answer",
        "dataset_name", "hash", "dataset_index", "reasoning", "misc"
    ]
    files = []
    for p in paths:
        files.extend(glob.glob(p) if any(ch in p for ch in "*?[]") else [p])
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {paths}")

    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_parquet(f, columns=want_cols, engine="pyarrow")
        except Exception:
            df = pd.read_parquet(f, engine="pyarrow")
            cols_have = [c for c in want_cols if c in df.columns]
            df = df[cols_have]

        for col in ["question", "options", "answer", "dataset_name", "hash", "misc"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")
        if "dataset_index" in df.columns:
            df["dataset_index"] = pd.to_numeric(df["dataset_index"], errors="coerce").fillna(-1).astype(int)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=want_cols)

    out = pd.concat(dfs, ignore_index=True)
    for c in want_cols:
        if c not in out.columns:
            out[c] = "" if c != "dataset_index" else -1
    return out

# ------------------------- IO: images (from HF parquet only) -------------------------

def _uid_from_row(ds_name: Any, idx: Any) -> str:
    return f"{ds_name}#{int(idx)}"

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-#\.]+", "_", str(s))

def _iter_images_from_hf(ds, take_first_only: bool = True):
    """仅从 parquet 中取图：List(Image(decode=True)) 或 List({'bytes','path'}) 的 bytes。"""
    for ex in ds:
        ds_name = ex.get("dataset_name", "")
        ds_idx  = ex.get("dataset_index", -1)
        uid = _uid_from_row(ds_name, ds_idx)
        imgs = ex.get("images", None)
        if not imgs or not isinstance(imgs, list):
            continue

        def _decode_one(el):
            if isinstance(el, Image.Image):
                return el.convert("RGB")
            if isinstance(el, dict):
                b = el.get("bytes", None)
                if isinstance(b, (bytes, bytearray)):
                    try:
                        return Image.open(io.BytesIO(b)).convert("RGB")
                    except Exception:
                        return None
                return None
            return None

        if take_first_only:
            im = _decode_one(imgs[0])
            if im is not None:
                yield uid, im
        else:
            for el in imgs:
                im = _decode_one(el)
                if im is not None:
                    yield uid, im

def _cache_img_path(cache_dir: str, uid: str) -> str:
    return os.path.join(cache_dir, _safe(uid) + ".jpg")

def _maybe_cache_image(cache_dir: str, uid: str, img: Image.Image):
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_img_path(cache_dir, uid)
    if not os.path.exists(p):
        try:
            img.save(p, format="JPEG", quality=90)
        except Exception:
            img.save(p.replace(".jpg", ".png"), format="PNG")

def _load_cached_image(cache_dir: str, uid: str) -> Optional[Image.Image]:
    p = _cache_img_path(cache_dir, uid)
    if os.path.exists(p):
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None
    p2 = p.replace(".jpg", ".png")
    if os.path.exists(p2):
        try:
            return Image.open(p2).convert("RGB")
        except Exception:
            return None
    return None

def build_image_cache_from_parquet(paths: List[str], cache_dir: str, take_first_only: bool = True) -> int:
    """仅用于缓存首张图到本地；返回成功缓存的图片数。"""
    if not paths:
        return 0
    ds = load_dataset("parquet", data_files=paths, split="train")
    n = 0
    for uid, im in tqdm(list(_iter_images_from_hf(ds, take_first_only)), desc="[ImageCache]"):
        _maybe_cache_image(cache_dir, uid, im)
        n += 1
    return n

# ------------------------- Helpers: examples dumping -------------------------

def _row_match(df: pd.DataFrame, ds_name: str, idx: int) -> Optional[pd.Series]:
    m = df[(df.get("dataset_name","") == ds_name) & (df.get("dataset_index",-1) == int(idx))]
    if len(m) == 0:
        return None
    return m.iloc[0]

def _dump_pair_example(row_pair: Dict[str, Any],
                       df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       fields: List[str],
                       base_dir: str,
                       label: str,
                       sim_col: str,
                       rank: int,
                       cache_dir: Optional[str] = None):
    """为单对 pair 落地 meta.json / query.txt / train.txt / （可选）query.jpg / train.jpg"""
    qds = row_pair.get("query_dataset", "")
    qidx = int(row_pair.get("query_index", -1))
    tds = row_pair.get("train_dataset", "")
    tidx = int(row_pair.get("train_index", -1))
    qid  = row_pair.get("query_id", f"{qds}#{qidx}")
    tid  = row_pair.get("train_id", f"{tds}#{tidx}")
    sim  = row_pair.get(sim_col, None)

    qrow = _row_match(df_test, qds, qidx)
    trow = _row_match(df_train, tds, tidx)

    sub = f"{rank:04d}__q_{_safe(qid)}__t_{_safe(tid)}"
    if sim is not None:
        sub += f"__{sim_col}_{float(sim):.3f}"
    out_dir = os.path.join(base_dir, sub)
    os.makedirs(out_dir, exist_ok=True)

    def _row_to_dict(r: Optional[pd.Series]) -> Dict[str, Any]:
        if r is None:
            return {}
        d = {k: (None if pd.isna(v) else v) for k,v in r.to_dict().items()}
        d["__normalized_text__"] = build_record_text(d, fields)
        return d

    qdict = _row_to_dict(qrow)
    tdict = _row_to_dict(trow)

    # 若有图片缓存，尝试取出尺寸信息
    qimg_info, timg_info = None, None
    if cache_dir:
        qimg = _load_cached_image(cache_dir, qid)
        timg = _load_cached_image(cache_dir, tid)
        if qimg is not None:
            qimg.save(os.path.join(out_dir, "query.jpg"), quality=90)
            qimg_info = {"mode": qimg.mode, "size": {"w": qimg.size[0], "h": qimg.size[1]}}
        if timg is not None:
            timg.save(os.path.join(out_dir, "train.jpg"), quality=90)
            timg_info = {"mode": timg.mode, "size": {"w": timg.size[0], "h": timg.size[1]}}

    meta = {
        "label": label,
        "similarity_col": sim_col,
        "similarity": float(sim) if sim is not None else None,
        "query_id": qid, "train_id": tid,
        "query_dataset": qds, "query_index": qidx,
        "train_dataset": tds, "train_index": tidx,
        "fields": fields,
        "query_record": qdict,
        "train_record": tdict,
        "query_image": qimg_info,
        "train_image": timg_info
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 便于肉眼对比的纯文本
    def _write_txt(path: str, r: Optional[pd.Series], d: Dict[str, Any], tag: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{tag}: {d.get('dataset_name','')}#{d.get('dataset_index','')}\n")
            if r is None:
                f.write("(record not found)\n"); return
            for k in ["question", "options", "answer", "answer_label", "misc"]:
                if k in r:
                    f.write(f"\n## {k}\n{r[k]}\n")
            f.write("\n## __normalized_text__\n")
            f.write(d.get("__normalized_text__", "") + "\n")

    _write_txt(os.path.join(out_dir, "query.txt"), qrow, qdict, "QUERY")
    _write_txt(os.path.join(out_dir, "train.txt"), trow, tdict, "TRAIN")

def dump_top_examples(pairs_df: pd.DataFrame,
                      df_train: pd.DataFrame,
                      df_test: pd.DataFrame,
                      fields: List[str],
                      out_dir: str,
                      label: str,
                      sim_col: str,
                      top_n: int = 10,
                      cache_dir: Optional[str] = None) -> Optional[str]:
    """按相似度降序取前 top_n 对，落地子文件夹；返回 examples 根路径"""
    if pairs_df is None or pairs_df.empty:
        return None
    ex_dir = os.path.join(out_dir, f"{label.lower().replace(' ','_')}_examples")
    os.makedirs(ex_dir, exist_ok=True)
    top = pairs_df.sort_values([sim_col, "rank"] if "rank" in pairs_df.columns else [sim_col],
                               ascending=[False, True]).head(top_n)
    for i, row in enumerate(top.to_dict(orient="records"), start=1):
        _dump_pair_example(row, df_train, df_test, fields, ex_dir, label, sim_col, i, cache_dir=cache_dir)
    return ex_dir

# ------------------------- MinHash + Levenshtein -------------------------

def run_minhash_lsh(
    df_train: pd.DataFrame,
    df_query: pd.DataFrame,
    text_fields: List[str],
    num_perm: int = 128,
    jaccard_threshold: float = 0.80,
    char_ngram_n: int = 13,
    levenshtein_threshold: float = 0.90,
    limit_per_query: int = 50,
) -> pd.DataFrame:

    def prep(df):
        texts, ids, metas = [], [], []
        for _, row in df.iterrows():
            uid = f"{row.get('dataset_name','')}#{row.get('dataset_index','')}"
            txt = build_record_text(row, text_fields)
            texts.append(txt); ids.append(uid)
            metas.append((row.get("dataset_name",""), int(row.get("dataset_index", -1))))
        return ids, texts, metas

    train_ids, train_txts, train_meta = prep(df_train)
    query_ids, query_txts, query_meta = prep(df_query)

    def to_minhash(s: str) -> MinHash:
        mh = MinHash(num_perm=num_perm)
        for ng in char_ngrams(s, n=char_ngram_n):
            mh.update(ng.encode("utf-8", errors="ignore"))
        return mh

    print(f"[MinHash] building train signatures (num_perm={num_perm}) ...")
    train_mh = [to_minhash(t) for t in tqdm(train_txts)]
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    for uid, mh in tqdm(list(zip(train_ids, train_mh)), desc="[MinHash] indexing"):
        lsh.insert(uid, mh)

    print(f"[MinHash] querying + Levenshtein filter >= {levenshtein_threshold} ...")
    rows = []
    no_cand = 0
    cand_total = 0
    kept = 0

    for (qid, qtxt, (qds, qidx)) in tqdm(list(zip(query_ids, query_txts, query_meta))):
        qmh = to_minhash(qtxt)
        cand = lsh.query(qmh)
        if not cand:
            no_cand += 1
            continue
        cand = cand[:limit_per_query]
        cand_total += len(cand)
        cand_pos = {tid: pos for pos, tid in enumerate(train_ids)}
        for tid in cand:
            pos = cand_pos.get(tid)
            if pos is None:
                continue
            sim = lev_sim(qtxt, train_txts[pos])
            if sim >= levenshtein_threshold:
                kept += 1
                tds, tidx = train_meta[pos]
                rows.append({
                    "query_id": qid, "query_dataset": qds, "query_index": int(qidx),
                    "train_id": tid, "train_dataset": tds, "train_index": int(tidx),
                    "levenshtein_sim": float(sim)
                })

    cols = ["query_id","query_dataset","query_index","train_id","train_dataset","train_index","levenshtein_sim"]
    out = pd.DataFrame(rows, columns=cols)
    print(f"[MinHash] queries with NO candidates: {no_cand}/{len(query_ids)}")
    print(f"[MinHash] total candidates (pre-Lev): {cand_total}, kept by Lev: {kept}")
    if out.empty:
        return out
    return out.sort_values("levenshtein_sim", ascending=False)

# ------------------------- Embedding + FAISS -------------------------

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def run_embedding_faiss(
    df_train: pd.DataFrame,
    df_query: pd.DataFrame,
    text_fields: List[str],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 64,
    topk: int = 5,
    cosine_threshold: float = 0.88,
    device: Optional[str] = None,
    tau_list: Optional[List[float]] = None,
    exclude_same_id: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    def prep(df):
        texts, ids, metas = [], [], []
        for _, row in df.iterrows():
            uid = f"{row.get('dataset_name','')}#{row.get('dataset_index','')}"
            txt = build_record_text(row, text_fields)
            texts.append(txt); ids.append(uid)
            metas.append((row.get("dataset_name",""), int(row.get("dataset_index", -1))))
        return ids, texts, metas

    train_ids, train_txts, train_meta = prep(df_train)
    query_ids, query_txts, query_meta = prep(df_query)

    print(f"[Embedding] loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    def encode_texts(txts: List[str]) -> np.ndarray:
        embs = model.encode(txts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)
        return np.asarray(embs, dtype=np.float32)

    print("[Embedding] encoding train...")
    train_vecs = l2_normalize(encode_texts(train_txts))
    print("[Embedding] encoding query...")
    query_vecs = l2_normalize(encode_texts(query_txts))

    d = train_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train_vecs)

    print(f"[FAISS] searching topk={topk} ...")
    sims, nbrs = index.search(query_vecs, topk)

    if tau_list is None or len(tau_list) == 0:
        tau_list = [0.85, 0.88, 0.90]

    maxsims = []
    for qi, qid in enumerate(query_ids):
        chosen = None
        for rk in range(topk):
            nb = int(nbrs[qi, rk]); sim = float(sims[qi, rk])
            tid = train_ids[nb]
            if exclude_same_id and (qid == tid):
                continue
            chosen = sim; break
        maxsims.append(np.nan if chosen is None else chosen)

    maxsims = np.asarray(maxsims, dtype=np.float32)
    valid = ~np.isnan(maxsims)
    maxsim_mean   = float(np.nanmean(maxsims)) if valid.any() else float("nan")
    maxsim_median = float(np.nanmedian(maxsims)) if valid.any() else float("nan")
    maxsim_p95    = float(np.nanpercentile(maxsims[valid], 95)) if valid.any() else float("nan")

    overlap_stats = []
    valid_cnt = int(np.sum(valid))
    total_q = len(maxsims)
    for tau in tau_list:
        cnt = int(np.sum(maxsims[valid] >= tau))
        rate = cnt / valid_cnt if valid_cnt > 0 else 0.0
        overlap_stats.append({"tau": float(tau), "count": cnt, "rate": float(rate)})

    rows = []
    for qi, qid in enumerate(query_ids):
        qds, qidx = query_meta[qi]
        for rk in range(topk):
            sim = float(sims[qi, rk])
            if sim < cosine_threshold:
                continue
            nb = int(nbrs[qi, rk])
            tds, tidx = train_meta[nb]
            tid = train_ids[nb]
            if exclude_same_id and (qid == tid):
                continue
            rows.append({
                "query_id": qid, "query_dataset": qds, "query_index": int(qidx),
                "train_id": tid, "train_dataset": tds, "train_index": int(tidx),
                "cosine_sim": sim, "rank": rk
            })
    pairs = pd.DataFrame(rows, columns=[
        "query_id","query_dataset","query_index",
        "train_id","train_dataset","train_index",
        "cosine_sim","rank"
    ])
    if not pairs.empty:
        pairs = pairs.sort_values(["cosine_sim","rank"], ascending=[False,True])

    aux = {
        "maxsim": {"mean": maxsim_mean, "median": maxsim_median, "p95": maxsim_p95},
        "overlap_at": overlap_stats,
        "maxsims_count": int(total_q),
        "maxsims_valid": int(valid_cnt),
    }
    return pairs, aux

# ------------------------- Reporting -------------------------

def hist_counts(series: pd.Series, bins: List[float]) -> List[Tuple[str,int]]:
    if series.empty:
        return []
    counts, edges = np.histogram(series, bins=bins)
    res = []
    for i, c in enumerate(counts):
        res.append((f"[{edges[i]:.2f},{edges[i+1]:.2f})", int(c)))
    return res

def print_section(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def report_pairs(
    pairs: pd.DataFrame,
    n_queries: int,
    sim_col: str,
    buckets: List[float],
    label: str,
    out_dir: str,
) -> Dict[str, Any]:
    print_section(f"{label} — Summary")
    if pairs.empty:
        print("No pairs found above threshold.")
        return {"pairs": 0, "queries": n_queries, "hit_queries": 0, "hit_rate": 0.0}

    hit_queries = pairs["query_id"].nunique()
    hit_rate = hit_queries / max(1, n_queries)

    top1 = pairs.sort_values([ "query_id", sim_col ], ascending=[True, False])\
                .drop_duplicates(["query_id"])\
                .reset_index(drop=True)
    top1_mean = float(top1[sim_col].mean())
    top1_median = float(top1[sim_col].median())
    top1_p95 = float(top1[sim_col].quantile(0.95))

    hist = hist_counts(pairs[sim_col], buckets)

    per_q = pairs.groupby("query_id").size().values
    per_q_mean = float(np.mean(per_q))
    per_q_max  = int(np.max(per_q)) if len(per_q)>0 else 0

    top_train = (pairs.groupby(["train_id","train_dataset"])["query_id"]
                      .nunique().reset_index(name="n_queries")
                      .sort_values("n_queries", ascending=False).head(10))

    top_query = (pairs.groupby(["query_id","query_dataset"])["train_id"]
                      .nunique().reset_index(name="n_trains")
                      .sort_values("n_trains", ascending=False).head(10))

    print(f"Queries scanned         : {n_queries}")
    print(f"Pairs >= threshold      : {len(pairs)}")
    print(f"Queries with any hit    : {hit_queries}  (hit-rate={hit_rate:.3f})")
    print(f"Per-query hits: mean={per_q_mean:.2f}, max={per_q_max}")
    print(f"Top-1 similarity: mean={top1_mean:.3f}, median={top1_median:.3f}, p95={top1_p95:.3f}")
    print("Similarity histogram:")
    for b, c in hist:
        print(f"  {b}: {c}")

    print("\nTop train templates (by #distinct queries matched):")
    if not top_train.empty:
        for _, r in top_train.iterrows():
            print(f"  {r['train_dataset']}#{r['train_id'].split('#')[-1]}  -> {int(r['n_queries'])} queries")
    else:
        print("  (none)")

    print("\nTop query duplicates (by #distinct trains matched):")
    if not top_query.empty:
        for _, r in top_query.iterrows():
            print(f"  {r['query_dataset']}#{r['query_id'].split('#')[-1]} -> {int(r['n_trains'])} trains")
    else:
        print("  (none)")

    top_train_path = os.path.join(out_dir, f"{label.lower().replace(' ','_')}_top_train_templates.csv")
    top_train.to_csv(top_train_path, index=False)
    top_query_path = os.path.join(out_dir, f"{label.lower().replace(' ','_')}_top_query_duplicates.csv")
    top_query.to_csv(top_query_path, index=False)

    return {
        "pairs": int(len(pairs)),
        "queries": int(n_queries),
        "hit_queries": int(hit_queries),
        "hit_rate": float(hit_rate),
        "per_query_hits_mean": float(per_q_mean),
        "per_query_hits_max": int(per_q_max),
        "top1_mean": top1_mean,
        "top1_median": top1_median,
        "top1_p95": top1_p95,
        "hist": hist,
        "top_train_csv": top_train_path,
        "top_query_csv": top_query_path,
    }

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Text audit with terminal report + saved CSV/JSON + examples (+images from HF parquet).")
    ap.add_argument("--train", nargs="+", required=True, help="Train parquet path(s) or globs (text fields)")
    ap.add_argument("--test",  nargs="+", required=True, help="Test parquet path(s) or globs (text fields)")
    ap.add_argument("--fields", default=EXPECTED_TEXT_FIELDS_DEFAULT, help="Comma-separated text fields")
    ap.add_argument("--out_dir", default="text_audit_out", help="Output directory")
    ap.add_argument("--examples_topk", type=int, default=10, help="Top-N suspicious pairs to dump as examples")

    # images (optional, HF parquet only; used to dump query.jpg/train.jpg for text pairs)
    ap.add_argument("--train_img", nargs="+", help="HF parquet path(s) to load TRAIN images from parquet (for examples)")
    ap.add_argument("--test_img",  nargs="+", help="HF parquet path(s) to load TEST images from parquet (for examples)")
    ap.add_argument("--img_take_first_only", action="store_true", default=True,
                    help="Only cache the first image per record (default True)")

    # A: MinHash
    ap.add_argument("--run_minhash", action="store_true")
    ap.add_argument("--num_perm", type=int, default=128)
    ap.add_argument("--jaccard_thr", type=float, default=0.80)
    ap.add_argument("--ngram_n", type=int, default=13)
    ap.add_argument("--lev_thr", type=float, default=0.92)
    ap.add_argument("--lsh_limit_per_query", type=int, default=50)

    # B: Embedding
    ap.add_argument("--run_embed", action="store_true")
    ap.add_argument("--model", default="BAAI/bge-m3", help="e.g., BAAI/bge-m3 or intfloat/e5-large-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cos_thr", type=float, default=0.88)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tau_list", default="0.85,0.88,0.90",
                    help="Comma-separated cosine thresholds for Overlap@tau, e.g. '0.85,0.88,0.90'")
    ap.add_argument("--exclude_same_id", action="store_true",
                    help="When train==test, exclude identical (dataset#index) self-matches in MaxSim and pairs.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    print("[IO] Loading train parquet (text) ...")
    df_train = load_parquet_to_df(args.train)
    print(f"[IO] train shape: {df_train.shape}")

    print("[IO] Loading test parquet (text) ...")
    df_test = load_parquet_to_df(args.test)
    print(f"[IO] test shape : {df_test.shape}")

    # optional: build image cache for examples
    cache_dir = os.path.join(args.out_dir, "cache_images")
    if (args.train_img and len(args.train_img) > 0) or (args.test_img and len(args.test_img) > 0):
        if args.train_img:
            print("[Images] Caching TRAIN images from parquet ...")
            _ = build_image_cache_from_parquet(args.train_img, cache_dir, take_first_only=args.img_take_first_only)
        if args.test_img:
            print("[Images] Caching TEST images from parquet ...")
            _ = build_image_cache_from_parquet(args.test_img, cache_dir, take_first_only=args.img_take_first_only)
        print(f"[Images] Cache dir: {cache_dir}")

    n_queries = len(df_test)
    summary = {
        "meta": {
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "fields": fields
        }
    }

    # ---------- Pipeline A ----------
    if args.run_minhash:
        print("\n=== Pipeline A: MinHash + Levenshtein ===")
        pairsA = run_minhash_lsh(
            df_train=df_train,
            df_query=df_test,
            text_fields=fields,
            num_perm=args.num_perm,
            jaccard_threshold=args.jaccard_thr,
            char_ngram_n=args.ngram_n,
            levenshtein_threshold=args.lev_thr,
            limit_per_query=args.lsh_limit_per_query,
        )
        outA = os.path.join(args.out_dir, "minhash_lev_pairs.csv")
        pairsA.to_csv(outA, index=False)
        print(f"[SAVE] {outA}  (n={len(pairsA)})")

        bucketsA = [0.80, 0.85, 0.90, 0.95, 1.01]
        repA = report_pairs(
            pairs=pairsA,
            n_queries=n_queries,
            sim_col="levenshtein_sim",
            buckets=bucketsA,
            label="MinHash_Levenshtein",
            out_dir=args.out_dir,
        )
        # examples (with images if cached)
        exA_dir = dump_top_examples(
            pairsA, df_train, df_test, fields, args.out_dir,
            label="MinHash_Levenshtein", sim_col="levenshtein_sim",
            top_n=args.examples_topk, cache_dir=cache_dir if os.path.exists(cache_dir) else None
        )
        if exA_dir:
            repA["examples_dir"] = exA_dir
            print(f"[EXAMPLES] MinHash_Levenshtein examples -> {exA_dir}")
        summary["minhash_lev"] = repA

    # ---------- Pipeline B ----------
    if args.run_embed:
        print("\n=== Pipeline B: Embedding (BGE/E5) + FAISS ===")
        tau_vals = [float(x.strip()) for x in args.tau_list.split(",") if x.strip()]
        pairsB, auxB = run_embedding_faiss(
            df_train=df_train,
            df_query=df_test,
            text_fields=fields,
            model_name=args.model,
            batch_size=args.batch_size,
            topk=args.topk,
            cosine_threshold=args.cos_thr,
            device=args.device,
            tau_list=tau_vals,
            exclude_same_id=args.exclude_same_id,
        )
        outB = os.path.join(args.out_dir, "embed_faiss_pairs.csv")
        pairsB.to_csv(outB, index=False)
        print(f"[SAVE] {outB}  (n={len(pairsB)})")

        print("\n===================")
        print("MaxSim (all queries)")
        print("===================")
        print(f"mean={auxB['maxsim']['mean']:.3f}  median={auxB['maxsim']['median']:.3f}  p95={auxB['maxsim']['p95']:.3f}")
        print("\n=====================")
        print("Overlap@tau (MaxSim≥τ)")
        print("=====================")
        for item in auxB["overlap_at"]:
            print(f"tau={item['tau']:.2f}: {item['count']}/{auxB['maxsims_count']}  (overlap={item['rate']:.3f})")

        bucketsB = [0.80, 0.85, 0.90, 0.95, 1.01]
        repB = report_pairs(
            pairs=pairsB,
            n_queries=n_queries,
            sim_col="cosine_sim",
            buckets=bucketsB,
            label="Embedding_FAISS",
            out_dir=args.out_dir,
        )
        repB.update({
            "maxsim": auxB["maxsim"],
            "overlap_at": auxB["overlap_at"],
            "maxsims_count": auxB["maxsims_count"],
            "maxsims_valid": auxB["maxsims_valid"],
            "cos_thr": float(args.cos_thr),
            "topk": int(args.topk),
            "model": args.model,
            "exclude_same_id": bool(args.exclude_same_id),
        })
        # examples (with images if cached)
        exB_dir = dump_top_examples(
            pairsB, df_train, df_test, fields, args.out_dir,
            label="Embedding_FAISS", sim_col="cosine_sim",
            top_n=args.examples_topk, cache_dir=cache_dir if os.path.exists(cache_dir) else None
        )
        if exB_dir:
            repB["examples_dir"] = exB_dir
            print(f"[EXAMPLES] Embedding_FAISS examples -> {exB_dir}")

        summary["embed_faiss"] = repB

    # 保存 JSON 总结
    sum_path = os.path.join(args.out_dir, "summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[SUMMARY SAVED] {sum_path}")

    if (not args.run_minhash) and (not args.run_embed):
        print("\n(No pipeline selected) Use --run_minhash and/or --run_embed.", file=sys.stderr)

if __name__ == "__main__":
    main()
