python eval/run_offline_inference.py \
    --model outputs/converted/qwen2_5_vl_7b_biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset1k \
    --dp_size 8 \
    --tp_size 1 \
    --temperature 0.0 \
    --batch_size 32 \
    --output_dir outputs/eval/qwen2_5_vl_7b_biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset1k_eval \
    --dataset_name /home/efs/nwang60/datasets/MedVLThinker-Eval \
    --max_tokens 4096