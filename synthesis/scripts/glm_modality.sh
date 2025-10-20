python glm_modality.py \
  --model /opt/dlami/nvme/nwang60/GLM-4.5V \
  --data_files /opt/dlami/nvme/nwang60/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k \
  --output_dir outputs/glm_modality \
  --tp_size 4 --dp_size 2 \
  --batch_size 16 --max_tokens 2048 