python synthesis/generate/glm_generate_pmc.py \
  --data_files "/home/efs/nwang60/datasets/biomedica_webdataset_parquet_25k/*.parquet" \
  --model /home/efs/nwang60/models/GLM-4.5V \
  --dp_size 2 --tp_size 4 \
  --batch_size 16 --max_tokens 4096 \
  --glm_precision bf16 \
  --output_dir outputs/biomedica_vqa_25k_glm_pmc \
  --small_image_policy skip \
  --merge_after --keep_raw_output