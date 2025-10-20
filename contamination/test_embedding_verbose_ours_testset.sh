python text_dedup_and_similarity_report.py \
  --train "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k/*.parquet" \
  --test  "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-Eval/data/*.parquet" \
  --run_minhash --run_embed \
  --examples_topk 10 \
  --train_img "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k/*.parquet" \
  --test_img  "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-Eval/data/*.parquet" \
  --out_dir text_verbose_ours_testset_v2 \
  --ngram_n 3 --jaccard_thr 0.50 --num_perm 256 --lsh_limit_per_query 200 --lev_thr 0.90 \
  --model "BAAI/bge-m3" --cos_thr 0.88 --topk 5 