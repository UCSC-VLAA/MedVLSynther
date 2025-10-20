python image_contamination_report.py \
  --train "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k/*.parquet" \
  --test  "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-pmc_vqa/data/*.parquet" \
  --run_hash \
  --phash_maxdist "4,8,16" \
  --out_dir image_hush_ours_pmc
