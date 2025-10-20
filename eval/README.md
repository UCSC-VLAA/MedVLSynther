# Eval

Minimal utilities to run model evaluation, regrade outputs, and compute final metrics for medical MCQA.

## Scripts

- `run_offline_inference.py`: Run a trained **Qwen** model on a test set and dump raw predictions.
- `regrade_eval_results.py`: Robustly parse responses and **re-judge** outputs to fix earlier misparses.
- `easy_metric.py`: Scan a path for `regraded_eval_results.jsonl` files and **aggregate** final metrics.
- `qwen_medical_benchmark.py`: Benchmark **Qwen2.5-VL 72B** on the test set.
- `internvl_medical_benchmark.py`: Benchmark **InternVL-3.5 38B** on the test set.
- `glm_medical_benchmark.py`: Benchmark **GLM-4.5 108B** on the test set.

## Example

```bash
# run offline inference
bash eval/scripts/qwen2_5_vl_7b_biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k_eval.sh

# regrade result
python eval/regrade_eval_results.py -i outputs/eval/qwen2_5_vl_7b_biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k_eval/eval_results.jsonl --dataset_name /home/efs/nwang60/datasets/MedVLThinker-Eval

# compute final metrics
python eval/easy_metric.py -p outputs/eval/qwen2_5_vl_7b_biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k_eval
```