# Contamination Audit (train–test overlap)

**Purpose:** Minimal tools to audit potential contamination between a training pool and a held-out test set, at both **text** and **image** levels, using conservative (hash/lexical) and semantic (embedding) checks.

## Contents

- **`text_dedup_and_similarity_report.py`**
  - **What it checks**
    - **MinHash + Levenshtein**: finds near-exact lexical duplicates over normalized text (e.g., `question`, `options`).
    - **Embedding + FAISS**: sentence-encoder similarity to flag semantically close train–test pairs; also reports per-query **MaxSim** and **Overlap@τ**.
  - **What it produces**
    - CSVs of matched pairs; a compact `summary.json` (pairs, hit rate, histograms, MaxSim/Overlap); optional example folders with `meta.json` (+ cached images if available).

- **`image_contamination_report.py`**
  - **What it checks**
    - **MD5 (pixels)**: exact image duplicates.
    - **pHash (64-bit)**: near-duplicate images via Hamming distance (binary FAISS retrieval).
    - **Embedding + FAISS**: visual semantic similarity (OpenCLIP/BiomedCLIP or MedCLIP); per-image **MaxSim** and **Overlap@τ**.
  - **What it produces**
    - CSVs for MD5/pHash/embedding pairs; `summary.json` (MD5 rate, pHash Overlap@d, similarity histograms, MaxSim/Overlap); example folders with side-by-side images and full metadata.


**Notes**
- All similarity thresholds (Levenshtein, cosine, pHash Hamming) are configurable; self-matches can be excluded.
- Normalization for text includes lowercasing, whitespace cleanup, optional digit masking, and option formatting.

**Example script**

`test_embedding_verbose_ours_testset.sh`: 

```bash
python text_dedup_and_similarity_report.py \
  --train "/path/to/your/trainset" \
  --test  "/path/to/your/testset" \
  --run_minhash --run_embed \
  --examples_topk 10 \
  --train_img "/path/to/your/trainset" \
  --test_img  "/path/to/your/testset" \
  --out_dir text_verbose \
  --ngram_n 3 --jaccard_thr 0.50 --num_perm 256 --lsh_limit_per_query 200 --lev_thr 0.90 \
  --model "BAAI/bge-m3" --cos_thr 0.88 --topk 5
```

`image_hush_verbose_ours_testset.sh`:

```bash
python image_contamination_report.py \
  --train "path/to/the/trainset/*.parquet" \
  --test  "path/to/the/testset/*.parquet" \
  --out_dir image_embedding_hush_verbose \
  --run_hash --phash_pair_cutoff 8 --phash_topk 3 --phash_maxdist "4,8,16" \
  --run_embed --device cuda --topk 5 --cos_thr 0.88 \
  --examples_topk 10
```