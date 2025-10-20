python synthesis/cot/glm_cot.py \
  --model /home/efs/nwang60/models/GLM-4.5V \
  --dataset_name /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k \
  --output_dir outputs_eval/glm_biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset10k_cot \
  --tp_size 4 --dp_size 2 \
  --batch_size 16 --max_tokens 2048