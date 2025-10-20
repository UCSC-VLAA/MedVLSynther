python synthesis/verify/glm_verify.py \
  --model /home/efs/nwang60/models/GLM-4.5V \
  --data_files "/home/efs/nwang60/datasets/biomedica_webdataset_internvl_VQA_parquet_25k/*.parquet" \
  --output_dir outputs_verify/med_internvl_generation_25k_vqa_verify_glm \
  --tp_size 4 --dp_size 2 \
  --batch_size 16 --max_tokens 2048 \
  --max_images_per_prompt 8 \
  --min_image_side 28 --small_image_policy skip --merge_after --keep_raw_output