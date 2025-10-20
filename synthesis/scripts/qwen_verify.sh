python synthesis/verify/qwen_verify.py \
  --data_files "/home/efs/nwang60/datasets/biomedica_webdataset_glm_VQA_parquet_25k/*.parquet" \
  --model /home/efs/nwang60/models/Qwen2.5-VL-72B-Instruct \
  --dp_size 1 --tp_size 8 \
  --batch_size 32 --max_tokens 4096 \
  --max_images_per_prompt 8 \
  --output_dir outputs_verify/biomedica_vqa_glm_generated_qwen_verified_25k \
  --merge_after --keep_raw_output