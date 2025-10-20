#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, io, json, ujson, sys, warnings, hashlib, shutil
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import faiss

from datasets import load_dataset, Dataset

# =========================
# HF parquet -> PIL loader
# =========================

def _try_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            return ujson.loads(s)
        except Exception:
            return None

def _uid(ex: Dict[str, Any]) -> str:
    return f"{ex.get('dataset_name','')}#{ex.get('dataset_index',-1)}"

def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_#." else "_" for ch in str(s))

FIELDS_KEEP = ["question","options","answer_label","answer",
               "dataset_name","dataset_index","hash","reasoning","misc"]

def _to_py(v):
    # 尽量转成可 JSON 的 Python 基本类型
    if isinstance(v, (str, int, float)) or v is None:
        return v
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)

def _extract_meta(ex: Dict[str, Any], im: Optional[Image.Image]) -> Dict[str, Any]:
    meta = {"id": _uid(ex)}
    for k in FIELDS_KEEP:
        if k in ex:
            meta[k] = _to_py(ex[k])
    imgs = ex.get("images", None)
    n_imgs = len(imgs) if isinstance(imgs, list) else (0 if imgs is None else 1)
    meta["n_images"] = int(n_imgs)
    if im is not None:
        meta["first_image_mode"] = im.mode
        meta["first_image_size"] = {"width": im.size[0], "height": im.size[1]}
    return meta

def _iter_images_and_meta(ds: Dataset, take_first_only: bool = True):
    """
    仅从 parquet 中取图（不访问 path），并同时产出元信息（剔除 images 字段）。
    产出: (uid, PIL.Image, meta_dict)
    """
    for ex in ds:
        uid = _uid(ex)
        imgs = ex.get("images", None)
        if not imgs or not isinstance(imgs, list):
            continue

        def _decode_one(el) -> Optional[Image.Image]:
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
                yield uid, im, _extract_meta(ex, im)
        else:
            for el in imgs:
                im = _decode_one(el)
                if im is not None:
                    yield uid, im, _extract_meta(ex, im)

def load_hf_parquet(paths: List[str]) -> Dataset:
    return load_dataset("parquet", data_files=paths, split="train")

# =========================
# Small image cache helpers
# =========================

def _cache_path(cache_dir: str, uid: str) -> str:
    return os.path.join(cache_dir, _safe(uid) + ".jpg")

def maybe_cache_image(cache_dir: str, uid: str, img: Image.Image):
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, uid)
    if not os.path.exists(path):
        try:
            img.save(path, format="JPEG", quality=90)
        except Exception:
            img.save(path.replace(".jpg", ".png"), format="PNG")

def load_cached_image(cache_dir: str, uid: str) -> Optional[Image.Image]:
    p = _cache_path(cache_dir, uid)
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

# =========================
# Hash utilities
# =========================

def md5_of_pixels(img: Image.Image) -> str:
    arr = np.asarray(img, dtype=np.uint8)
    return hashlib.md5(arr.tobytes()).hexdigest()

def compute_phash(img: Image.Image, hash_size: int = 8) -> imagehash.ImageHash:
    return imagehash.phash(img, hash_size=hash_size)

def phash_to_bytes(h: imagehash.ImageHash) -> bytes:
    bits = np.packbits(h.hash.flatten().astype(np.uint8))
    return bits.tobytes()  # 64 bits -> 8 bytes

# =========================
# Embedding model (open-clip / medclip)
# =========================

def load_vision_model(model_name: str = "biomedclip", device: str = "cuda"):
    """
    优先 open-clip：biomedclip -> ViT-B-16；备选 medclip。
    返回 encode_fn(images: List[PIL]) -> np.ndarray[L2-normalized]
    """
    try:
        import torch
        import open_clip

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "biomedclip", pretrained="biomedclip"
            )
            model = model.to(device); model.eval()
            @torch.no_grad()
            def encode_fn(imgs: List[Image.Image]) -> np.ndarray:
                batch = torch.stack([preprocess(im).to(device) for im in imgs], dim=0)
                feats = model.encode_image(batch)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                return feats.detach().cpu().numpy().astype(np.float32)
            return encode_fn
        except Exception:
            pass

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
        model = model.to(device); model.eval()
        @torch.no_grad()
        def encode_fn(imgs: List[Image.Image]) -> np.ndarray:
            batch = torch.stack([preprocess(im).to(device) for im in imgs], dim=0)
            feats = model.encode_image(batch)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            return feats.detach().cpu().numpy().astype(np.float32)
        return encode_fn

    except Exception as e:
        warnings.warn(f"open-clip unavailable ({e}); trying medclip…")
        try:
            import torch
            from medclip import MedCLIPModel, MedCLIPProcessor
            device_t = torch.device(device if torch.cuda.is_available() else "cpu")
            model = MedCLIPModel("medclip-vit-base-patch16").to(device_t); model.eval()
            processor = MedCLIPProcessor()
            @torch.no_grad()
            def encode_fn(imgs: List[Image.Image]) -> np.ndarray:
                inputs = processor(images=imgs, return_tensors="pt").to(device_t)
                feats = model.get_image_features(**inputs)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                return feats.detach().cpu().numpy().astype(np.float32)
            return encode_fn
        except Exception as e2:
            raise RuntimeError("No vision model available. Please install 'open-clip-torch' or 'medclip'.") from e2

# =========================
# MD5 + pHash pipeline (export pairs & examples with full meta)
# =========================

def run_md5_phash_hf(
    train_ds: Dataset,
    test_ds: Dataset,
    phash_hash_size: int = 8,
    phash_topk: int = 3,
    phash_maxdist_list: List[int] = [4, 8, 16],
    phash_pair_cutoff: int = 8,
    take_first_only: bool = True,
    out_dir: Optional[str] = None,
    examples_topk: int = 10,
):
    cache_dir = os.path.join(out_dir, "cache_images") if out_dir else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # build train hashes + meta
    train_ids, train_md5, train_phash = [], [], []
    md5_to_train = {}
    fail_train = 0
    train_meta_map: Dict[str, Dict[str, Any]] = {}
    for uid, im, meta in tqdm(list(_iter_images_and_meta(train_ds, take_first_only)), desc="[Hash] train"):
        try:
            if cache_dir: maybe_cache_image(cache_dir, uid, im)
            phb = phash_to_bytes(compute_phash(im, phash_hash_size))
            m = md5_of_pixels(im)
            train_ids.append(uid)
            train_md5.append(m)
            train_phash.append(phb)
            md5_to_train.setdefault(m, []).append(uid)
            train_meta_map[uid] = meta
        except Exception:
            fail_train += 1
    md5_set = set(train_md5)

    # pHash index
    valid_idx = [i for i, b in enumerate(train_phash) if isinstance(b, (bytes, bytearray)) and len(b) == 8]
    xb = np.frombuffer(b"".join(train_phash[i] for i in valid_idx), dtype=np.uint8).reshape(len(valid_idx), 8)
    index_bin = faiss.IndexBinaryFlat(64)
    if len(valid_idx) > 0:
        index_bin.add(xb)

    # probe test
    total_test = 0
    md5_hits = 0
    phash_rows = []
    min_hams = []
    fail_test = 0
    md5_pairs = []    # (query_id, train_id)
    phash_pairs = []  # (query_id, train_id, hamming, rank)
    test_meta_map: Dict[str, Dict[str, Any]] = {}

    for uid, im, meta in tqdm(list(_iter_images_and_meta(test_ds, take_first_only)), desc="[Hash] test"):
        total_test += 1
        if cache_dir: maybe_cache_image(cache_dir, uid, im)
        test_meta_map[uid] = meta

        # MD5
        try:
            qm = md5_of_pixels(im)
            if qm in md5_set:
                md5_hits += 1
                for tid in md5_to_train.get(qm, [])[:phash_topk]:
                    md5_pairs.append((uid, tid))
        except Exception:
            pass

        # pHash
        try:
            q = phash_to_bytes(compute_phash(im, phash_hash_size))
            if len(valid_idx) > 0:
                xq = np.frombuffer(q, dtype=np.uint8)[None, :]
                D, I = index_bin.search(xq, phash_topk)
                dists = D[0].astype(int)
                best = int(dists[0]) if len(dists) > 0 else np.inf
                min_hams.append(best if best != np.inf else np.nan)
                for rk, (dist, nb) in enumerate(zip(dists, I[0])):
                    if dist <= phash_pair_cutoff:
                        train_uid = train_ids[ valid_idx[nb] ]
                        phash_pairs.append((uid, train_uid, int(dist), int(rk)))
                for d in phash_maxdist_list:
                    phash_rows.append((d, int(np.any(dists <= d))))
            else:
                min_hams.append(np.nan)
        except Exception:
            fail_test += 1
            min_hams.append(np.nan)

    # aggregate
    phash_stats = []
    for d in phash_maxdist_list:
        vals = [hit for dd, hit in phash_rows if dd == d]
        cnt = int(np.sum(vals)) if len(vals) > 0 else 0
        rate = cnt / max(1, total_test)
        phash_stats.append({"d": int(d), "count": cnt, "rate": float(rate)})

    min_hams = np.asarray(min_hams, dtype=float)
    valid = ~np.isnan(min_hams)
    minH_mean   = float(np.nanmean(min_hams)) if valid.any() else float("nan")
    minH_median = float(np.nanmedian(min_hams)) if valid.any() else float("nan")
    minH_p95    = float(np.nanpercentile(min_hams[valid], 95)) if valid.any() else float("nan")

    # report
    print("\n===== Image hashes — MD5 + pHash (HF parquet) =====")
    print(f"Train imgs (ok/fail): {(len(train_ids))}/{fail_train}")
    print(f"Test  imgs (count)  : {total_test} (fail~{fail_test})")
    print(f"MD5 duplicates      : {md5_hits}/{total_test}  (rate={md5_hits/max(1,total_test):.4f})")
    print("pHash Overlap@d (Hamming ≤ d):")
    for s in phash_stats:
        print(f"  d={s['d']:>2}: {s['count']}/{total_test}  (rate={s['rate']:.4f})")
    print("Min Hamming best: mean={:.2f}  median={:.2f}  p95={:.2f}".format(minH_mean, minH_median, minH_p95))

    # save CSV
    md5_df = pd.DataFrame(md5_pairs, columns=["query_id","train_id"])
    phash_df = pd.DataFrame(phash_pairs, columns=["query_id","train_id","hamming","rank"])
    if out_dir:
        if not md5_df.empty:
            md5_csv = os.path.join(out_dir, "hash_md5_pairs.csv")
            md5_df.to_csv(md5_csv, index=False)
            print(f"[SAVE] {md5_csv} (n={len(md5_df)})")
        if not phash_df.empty:
            phash_csv = os.path.join(out_dir, "hash_phash_pairs.csv")
            phash_df.sort_values(["hamming","rank"], ascending=[True,True]).to_csv(phash_csv, index=False)
            print(f"[SAVE] {phash_csv} (n={len(phash_df)})")

    # dump examples (Top-N) + full meta
    examples = {}
    def _write_pair_meta(e_dir, label, qid, tid, metric_name, metric_value, **kw):
        meta = {
            "label": label,
            "metric": {"name": metric_name, "value": metric_value},
            "query": test_meta_map.get(qid, {"id": qid}),
            "train": train_meta_map.get(tid, {"id": tid})
        }
        meta.update(kw or {})
        with open(os.path.join(e_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    if out_dir:
        if not md5_df.empty:
            md5_top = md5_df.head(examples_topk)
            md5_dir = os.path.join(out_dir, "Hash_MD5_examples")
            os.makedirs(md5_dir, exist_ok=True)
            for i, r in enumerate(md5_top.itertuples(index=False), start=1):
                sub = f"{i:04d}__q_{_safe(r.query_id)}__t_{_safe(r.train_id)}__md5"
                e_dir = os.path.join(md5_dir, sub); os.makedirs(e_dir, exist_ok=True)
                _write_pair_meta(e_dir, "Hash_MD5", r.query_id, r.train_id, "md5_hamming", 0)
                qimg = load_cached_image(cache_dir, r.query_id)
                timg = load_cached_image(cache_dir, r.train_id)
                if qimg: qimg.save(os.path.join(e_dir, "query.jpg"), quality=90)
                if timg: timg.save(os.path.join(e_dir, "train.jpg"), quality=90)
            examples["md5_examples_dir"] = md5_dir
            print(f"[EXAMPLES] MD5 examples -> {md5_dir}")

        if not phash_df.empty:
            ph_top = phash_df.sort_values(["hamming","rank"], ascending=[True,True]).head(examples_topk)
            ph_dir = os.path.join(out_dir, "Hash_pHash_examples")
            os.makedirs(ph_dir, exist_ok=True)
            for i, r in enumerate(ph_top.itertuples(index=False), start=1):
                sub = f"{i:04d}__q_{_safe(r.query_id)}__t_{_safe(r.train_id)}__hamming_{int(r.hamming):02d}__rank_{int(r.rank)}"
                e_dir = os.path.join(ph_dir, sub); os.makedirs(e_dir, exist_ok=True)
                _write_pair_meta(e_dir, "Hash_pHash", r.query_id, r.train_id, "hamming", int(r.hamming), rank=int(r.rank))
                qimg = load_cached_image(cache_dir, r.query_id)
                timg = load_cached_image(cache_dir, r.train_id)
                if qimg: qimg.save(os.path.join(e_dir, "query.jpg"), quality=90)
                if timg: timg.save(os.path.join(e_dir, "train.jpg"), quality=90)
            examples["phash_examples_dir"] = ph_dir
            print(f"[EXAMPLES] pHash examples -> {ph_dir}")

    return {
        "train_rows": int(len(train_ids)),
        "test_rows": int(total_test),
        "md5_hits": int(md5_hits),
        "md5_rate": float(md5_hits / max(1, total_test)),
        "phash_overlap": phash_stats,
        "min_hamming": {"mean": minH_mean, "median": minH_median, "p95": minH_p95},
        "failures": {"train": int(fail_train), "test": int(fail_test)},
        "pairs": {
            "md5_csv": os.path.join(out_dir, "hash_md5_pairs.csv") if out_dir and not md5_df.empty else None,
            "phash_csv": os.path.join(out_dir, "hash_phash_pairs.csv") if out_dir and not phash_df.empty else None,
        },
        "examples": examples,
        "cache_dir": cache_dir
    }

# =========================
# Embedding + FAISS (export pairs & examples with full meta)
# =========================

def run_embed_faiss_hf(
    train_ds: Dataset,
    test_ds: Dataset,
    device: str = "cuda",
    batch_size: int = 128,
    topk: int = 5,
    cos_thr: float = 0.88,
    tau_list: List[float] = [0.85, 0.88, 0.90],
    take_first_only: bool = True,
    exclude_same_id: bool = False,
    out_dir: Optional[str] = None,
    examples_topk: int = 10
):
    encode_fn = load_vision_model(device=device)
    cache_dir = os.path.join(out_dir, "cache_images") if out_dir else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # encode train (+ meta)
    train_ids, train_vecs, buf = [], [], []
    train_meta_map: Dict[str, Dict[str, Any]] = {}
    for uid, im, meta in tqdm(list(_iter_images_and_meta(train_ds, take_first_only)), desc="[Embed] train"):
        train_ids.append(uid); buf.append(im); train_meta_map[uid] = meta
        if cache_dir: maybe_cache_image(cache_dir, uid, im)
        if len(buf) >= batch_size:
            train_vecs.append(encode_fn(buf)); buf = []
    if buf: train_vecs.append(encode_fn(buf))
    if len(train_vecs) == 0:
        raise RuntimeError("No train embeddings.")
    train_vecs = np.vstack(train_vecs).astype(np.float32)

    # encode test (+ meta)
    test_ids, test_vecs, buf = [], [], []
    test_meta_map: Dict[str, Dict[str, Any]] = {}
    for uid, im, meta in tqdm(list(_iter_images_and_meta(test_ds, take_first_only)), desc="[Embed] test"):
        test_ids.append(uid); buf.append(im); test_meta_map[uid] = meta
        if cache_dir: maybe_cache_image(cache_dir, uid, im)
        if len(buf) >= batch_size:
            test_vecs.append(encode_fn(buf)); buf = []
    if buf: test_vecs.append(encode_fn(buf))
    if len(test_vecs) == 0:
        raise RuntimeError("No test embeddings.")
    test_vecs = np.vstack(test_vecs).astype(np.float32)

    # FAISS
    d = train_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train_vecs)
    sims, nbrs = index.search(test_vecs, topk)

    # MaxSim & Overlap@τ
    maxsims = []
    for qi, qid in enumerate(test_ids):
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
    max_mean   = float(np.nanmean(maxsims)) if valid.any() else float("nan")
    max_median = float(np.nanmedian(maxsims)) if valid.any() else float("nan")
    max_p95    = float(np.nanpercentile(maxsims[valid], 95)) if valid.any() else float("nan")

    overlap = []
    tot = int(np.sum(valid))
    for tau in tau_list:
        cnt = int(np.sum(maxsims[valid] >= tau))
        overlap.append({"tau": float(tau), "count": cnt, "rate": float(cnt / max(1, tot))})

    # collect pairs ≥ cos_thr
    rows = []
    hist_bins = [0.80, 0.85, 0.90, 0.95, 1.01]
    hist_counts = [0, 0, 0, 0]
    for qi, qid in enumerate(test_ids):
        for rk in range(topk):
            nb = int(nbrs[qi, rk]); sim = float(sims[qi, rk])
            if sim < cos_thr:
                continue
            tid = train_ids[nb]
            if exclude_same_id and (qid == tid):
                continue
            rows.append((qid, tid, sim, rk))
            if hist_bins[1] <= sim < hist_bins[2]: hist_counts[1] += 1
            elif hist_bins[2] <= sim < hist_bins[3]: hist_counts[2] += 1
            elif hist_bins[3] <= sim < hist_bins[4]: hist_counts[3] += 1
            elif hist_bins[0] <= sim < hist_bins[1]: hist_counts[0] += 1

    pairs = pd.DataFrame(rows, columns=["query_id","train_id","cosine_sim","rank"])
    pairs = pairs.sort_values(["cosine_sim","rank"], ascending=[False, True])

    # report
    print("\n===== Image embeddings — FAISS (HF parquet) =====")
    print(f"Train embeddings : {len(train_ids)}")
    print(f"Test  embeddings : {len(test_ids)}")
    print(f"Pairs ≥ {cos_thr:.2f} : {len(pairs)}")
    hit_q = pairs["query_id"].nunique() if not pairs.empty else 0
    print(f"Queries with any hit: {hit_q}/{len(test_ids)} (rate={hit_q/max(1,len(test_ids)):.4f})")
    if not pairs.empty:
        top1 = (pairs.sort_values(["query_id","cosine_sim"], ascending=[True,False])
                      .drop_duplicates(["query_id"]))
        print("Top-1 cosine: mean={:.3f}  median={:.3f}  p95={:.3f}".format(
            float(top1["cosine_sim"].mean()),
            float(top1["cosine_sim"].median()),
            float(top1["cosine_sim"].quantile(0.95)),
        ))
        print("Similarity histogram (pairs):")
        print(f"  [0.80,0.85): {hist_counts[0]}")
        print(f"  [0.85,0.90): {hist_counts[1]}")
        print(f"  [0.90,0.95): {hist_counts[2]}")
        print(f"  [0.95,1.01): {hist_counts[3]}")
    print("MaxSim: mean={:.3f}  median={:.3f}  p95={:.3f}".format(max_mean, max_median, max_p95))
    print("Overlap@τ:")
    for o in overlap:
        print(f"  τ={o['tau']:.2f}: {o['count']}/{len(test_ids)}  (rate={o['rate']:.4f})")

    # save CSV & dump examples with full meta
    pairs_path = None
    if out_dir:
        pairs_path = os.path.join(out_dir, "embed_pairs.csv")
        pairs.to_csv(pairs_path, index=False)
        print(f"[SAVE] {pairs_path} (n={len(pairs)})")

        if not pairs.empty:
            ex_dir = os.path.join(out_dir, "Embedding_examples")
            os.makedirs(ex_dir, exist_ok=True)
            head = pairs.head(examples_topk)

            def _write_pair_meta(e_dir, qid, tid, cos, rank):
                meta = {
                    "label": "Embedding_FAISS",
                    "metric": {"name": "cosine", "value": float(cos), "rank": int(rank)},
                    "query": test_meta_map.get(qid, {"id": qid}),
                    "train": train_meta_map.get(tid, {"id": tid})
                }
                with open(os.path.join(e_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

            for i, r in enumerate(head.itertuples(index=False), start=1):
                sub = f"{i:04d}__q_{_safe(r.query_id)}__t_{_safe(r.train_id)}__cosine_{float(r.cosine_sim):.3f}__rank_{int(r.rank)}"
                e_dir = os.path.join(ex_dir, sub); os.makedirs(e_dir, exist_ok=True)
                _write_pair_meta(e_dir, r.query_id, r.train_id, r.cosine_sim, r.rank)
                qimg = load_cached_image(cache_dir, r.query_id)
                timg = load_cached_image(cache_dir, r.train_id)
                if qimg: qimg.save(os.path.join(e_dir, "query.jpg"), quality=90)
                if timg: timg.save(os.path.join(e_dir, "train.jpg"), quality=90)
            print(f"[EXAMPLES] Embedding examples -> {ex_dir}")

    return {
        "pairs_df": pairs,
        "summary": {
            "pairs": int(len(pairs)),
            "queries": int(len(test_ids)),
            "hit_queries": int(hit_q),
            "hit_rate": float(hit_q/max(1,len(test_ids))),
            "hist_pairs": {
                "[0.80,0.85)": int(hist_counts[0]),
                "[0.85,0.90)": int(hist_counts[1]),
                "[0.90,0.95)": int(hist_counts[2]),
                "[0.95,1.01)": int(hist_counts[3]),
            },
            "maxsim": {"mean": max_mean, "median": max_median, "p95": max_p95},
            "overlap_at": overlap,
            "pairs_csv": pairs_path
        },
        "cache_dir": cache_dir
    }

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="Image-level contamination (HF parquet only): MD5+pHash; Embedding+FAISS (+examples+full-meta)")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test",  nargs="+", required=True)
    ap.add_argument("--out_dir", default="image_audit_hf")
    ap.add_argument("--examples_topk", type=int, default=10, help="Top-N example pairs to dump for each method")

    # hash
    ap.add_argument("--run_hash", action="store_true")
    ap.add_argument("--phash_hash_size", type=int, default=8)
    ap.add_argument("--phash_topk", type=int, default=3)
    ap.add_argument("--phash_maxdist", default="4,8,16",
                    help="Comma-separated d for Overlap@d reporting")
    ap.add_argument("--phash_pair_cutoff", type=int, default=8,
                    help="Collect pHash pairs with Hamming ≤ cutoff for CSV/examples")

    # embed
    ap.add_argument("--run_embed", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cos_thr", type=float, default=0.88)
    ap.add_argument("--tau_list", default="0.85,0.88,0.90")
    ap.add_argument("--exclude_same_id", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[IO] Loading train parquet (HF)…")
    ds_train = load_hf_parquet(args.train)
    print(ds_train.features)

    print("[IO] Loading test parquet (HF)…")
    ds_test  = load_hf_parquet(args.test)
    print(ds_test.features)

    summary = {"meta": {"train_rows": ds_train.num_rows, "test_rows": ds_test.num_rows}}

    if args.run_hash:
        maxd = [int(x.strip()) for x in args.phash_maxdist.split(",") if x.strip()]
        res_hash = run_md5_phash_hf(
            train_ds=ds_train,
            test_ds=ds_test,
            phash_hash_size=args.phash_hash_size,
            phash_topk=args.phash_topk,
            phash_maxdist_list=maxd,
            phash_pair_cutoff=args.phash_pair_cutoff,
            out_dir=args.out_dir,
            examples_topk=args.examples_topk,
        )
        summary["hash"] = {
            "train_rows": res_hash["train_rows"],
            "test_rows":  res_hash["test_rows"],
            "md5_hits":   res_hash["md5_hits"],
            "md5_rate":   res_hash["md5_rate"],
            "phash_overlap": res_hash["phash_overlap"],
            "min_hamming": res_hash["min_hamming"],
            "failures": res_hash["failures"],
            "pairs_csv": res_hash["pairs"],
            "examples_dir": res_hash["examples"],
        }

    if args.run_embed:
        tau_vals = [float(x.strip()) for x in args.tau_list.split(",") if x.strip()]
        res_emb = run_embed_faiss_hf(
            train_ds=ds_train,
            test_ds=ds_test,
            device=args.device,
            batch_size=args.batch_size,
            topk=args.topk,
            cos_thr=args.cos_thr,
            tau_list=tau_vals,
            exclude_same_id=args.exclude_same_id,
            out_dir=args.out_dir,
            examples_topk=args.examples_topk,
        )
        summary["embed"] = res_emb["summary"]
        summary["embed"].update({
            "cos_thr": float(args.cos_thr),
            "topk": int(args.topk),
            "exclude_same_id": bool(args.exclude_same_id),
        })

    sum_path = os.path.join(args.out_dir, "summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SUMMARY SAVED] {sum_path}")

    if (not args.run_hash) and (not args.run_embed):
        print("\n(No pipeline selected) Use --run_hash and/or --run_embed.", file=sys.stderr)

if __name__ == "__main__":
    main()
