export CUDA_VISIBLE_DEVICES=0,1,2,3
python synthesis/generate/internvl_generate.py \
  --data_files "/home/efs/nwang60/datasets/biomedica_webdataset_parquet_25k/*.parquet" \
  --model /home/efs/nwang60/models/InternVL3_5-38B \
  --dp_size 1 --tp_size 4 \
  --batch_size 32 --max_tokens 4096 \
  --max_images_per_prompt 8 \
  --output_dir outputs/biomedica_internvl_generation_vqa_3_5 \
  --merge_after --keep_raw_output