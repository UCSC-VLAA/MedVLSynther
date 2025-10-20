python vllm_glm_type.py \
  --model /home/efs/nwang60/models/GLM-4.5V \
  --data_files "/home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k/*.parquet" \
  --output_dir outputs_verify/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k_type \
  --tp_size 4 --dp_size 2 \
  --batch_size 16 --max_tokens 64 \
  --merge_after --keep_raw_output