python text_dedup_and_similarity_report.py \
  --train "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-Eval/data/*.parquet" \
  --test  "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-Eval/data/*.parquet" \
  --fields "question,options" \
  --run_minhash \
  --out_dir out_text_audit_string_debug \
  --ngram_n 5 \
  --jaccard_thr 0.60 \
  --lev_thr 0.85 \
  --lsh_limit_per_query 100