python synthesis/generate/qwen_generate.py \
  --data_files "/home/efs/nwang60/datasets/biomedica_webdataset_parquet_25k/*.parquet" \
  --model /home/efs/nwang60/models/Qwen2.5-VL-72B-Instruct \
  --dp_size 1 --tp_size 8 \
  --batch_size 16 --max_tokens 4096 \
  --trust_remote_code \
  --placeholder_style qwen25 \
  --output_dir outputs/biomedica_vqa_25k_qwen72b \
  --merge_after --keep_raw_output