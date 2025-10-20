python image_contamination_report.py \
  --train "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k/*.parquet" \
  --test  "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-Eval/data/*.parquet" \
  --out_dir image_hush_verbose_ours_testset_v2 \
  --run_hash --phash_pair_cutoff 8 --phash_topk 3 --phash_maxdist "4,8,16" \
  --run_embed --device cuda --topk 5 --cos_thr 0.88 \
  --examples_topk 10