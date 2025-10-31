# data_process

Tools for make datasets for verl or trl.

- `prep_to_hf_bytes.py`: convert trainset into huggingface pyarrow file.
- `prep_to_hf_bytes_eval.py`: convert testset into huggingface pyarrow file. Download testset here: [MedVLThinker-Eval](https://huggingface.co/datasets/UCSC-VLAA/MedVLThinker-Eval)
- `convert_verl_format.py`: convert trainset pyarrow file into verl format.
- `convert_trl_for_vlm.py`: comvert trainset pyarrow file into trl format.

## trainset data format

```py
{
    "images": [PIL.Image],           # List of images                           
    "question": str,                 # Question text
    "options": Dict[str, str],       # Multiple choice options
    "answer_label": str,             # Correct answer label (A, B, C, D, E)
    "answer": str,                   # Full answer text
    "reasoning": str,                # Chain-of-thought reasoning (optional)
    "dataset_name": str,             # Source dataset name
    "dataset_index": int             # Unique sample identifier
}
```

## verl example

```bash
python data_process/prep_to_hf_bytes.py \
    --parquet_glob "/home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k/*.parquet" \
    --out_dir /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k_hf \
    --num_proc 32 --strict_image --keep_first_k_images 6

python data_process/convert_verl_format.py \
    --local_data_dir /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k_hf \
    --data_source biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_subset5k_13k \
    --ability medical_mcqa \
    --split train \
    --output_dir  /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k_verl \
    --num_proc 32
```

## trl example

```bash
python convert_trl_for_vlm.py \
  --is_local true \
  -d /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_mean9670_13k_rebalanced_subset_cot_rebalanced_7k_subset5k_hf \
  --tokenizer_name Qwen/Qwen2.5-VL-7B-Instruct \
  --out_parquet /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_mean9670_13k_rebalanced_subset_cot_rebalanced_7k_subset5k_trl/train.parquet \
  -n 16 \
  --keep_in_memory True
```
