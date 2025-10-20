# Analysis

- `result_viewer.py`: evaluate result viewer.

- `rubric_viewer.py`: MCQ verify result viewer.

Update line 190-198 with your local path:

```py
parquet_input = st.sidebar.text_input(
    "Parquet 路径/通配符/目录",
    "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_VQA_parquet_25k/*.parquet",
)

jsonl_input = st.sidebar.text_input(
    "JSONL（rubric/验证元数据）路径",
    "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_score_filtered.jsonl",
)
```
- `bonus_item.py`: checks all non-essential (“bonus”) rubric items and summarizes coverage/violations.

- `type_distribution.py`: computes question counts by **type** and outputs compact summaries.

- `type_modality_anatomy_distribution.py`: compute question number of different types, modalities and anatomies.

- `vqa_distribution.py`: generated_vqa + image_secondary_label distribution. Including word frequency, token length, image_secondary_label distribution.