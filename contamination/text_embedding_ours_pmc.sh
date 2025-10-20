python text_dedup_and_similarity_report.py \
  --train "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k/*.parquet" \
  --test  "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-pmc_vqa/data/*.parquet" \
  --fields "question,options" \
  --run_embed \
  --model "BAAI/bge-m3" \
  --device cuda \
  --out_dir text_embedding_ours_pmc