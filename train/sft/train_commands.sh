bash train/sft/sft_vlm_local.sh \
    --train_dataset_name /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_mean9670_13k_rebalanced_subset_cot_rebalanced_7k_subset5k_trl \
    --model_name "/home/efs/nwang60/models/Qwen2.5-VL-3B-Instruct" \
    --epochs 5 \
    --gpu_count 8 \
    --output_dir outputs/sft-ours \
    --exp_name 3b \
    --gradient_checkpointing False \
    --use_flash_attention_2 False

bash train/sft/sft_vlm_local.sh \
    --train_dataset_name /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_mean9670_13k_rebalanced_subset_cot_rebalanced_7k_subset5k_trl \
    --model_name "/home/efs/nwang60/models/Qwen2.5-VL-7B-Instruct" \
    --epochs 5 \
    --gpu_count 8 \
    --output_dir outputs/sft-ours \
    --exp_name 7b \
    --gradient_checkpointing False \
    --use_flash_attention_2 False