# Synthesis

This folder provides the code for the full **synthesis pipeline**.

**Prerequisite dataset: **
Download Biomedica: [https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M](https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M)

## Tool introduction

### root

* **`curate_data_json_mp_wholeset.py`**: Curate a target number of samples from the full webdataset.

  * Selection:

    * Keep items with `image_primary_label = "Clinical Imaging"` and `image_secondary_label ∈ { 'x-ray radiography', 'optical coherence tomography', 'endoscopy', 'intraoral imaging', 'angiography', 'procedural image', 'skull', 'patient photo', 'functional magnetic resonance', 'magnetic resonance', 'eye', 'mammography', 'electrocardiography', 'clinical imaging', 'skin lesion', 'ultrasound', 'specimen', 'computerized tomography', 'laryngoscopy', 'teeth', 'intraoperative image', 'surgical procedure', 'brain' }`.
    * Also keep items with `image_primary_label = "Microscopy"` **and** `image_secondary_label = "light microscopy"`.
    * Each sample is assigned to **one** secondary-label bucket. Buckets are **balanced** to take equal proportions of the total.
    * If a PMC article contains multiple images sharing the **same caption**, they are grouped for **multi-image MCQ** construction. The Image:Caption count distribution is enforced as **1,2,3,4,5,6 → 90%,2%,2%,2%,2%,2%**.
  * **How to use**:

    ```bash
    python curate_data_json_mp_wholeset.py \
        --base /path/to/biomedica_webdataset_24M \
        --out  /path/to/output/json/output.json \
        --total 25000 \
        --workers 16
    ```

* **`convert_json_parquet_mp.py`**: Merge image bytes and text fields into **Parquet** for downstream MCQ generation.

  * **How to use**:

    ```bash
    python convert_json_parquet_mp.py \
        --base /path/to/biomedica_surgery_subset_webdataset \
        --input /path/to/output/json/output.json \
        --out /path/to/output/parquet \
        --rows-per-shard 10000 \
        --workers 8
    ```

* **`combine_generated_MCQ_parquet.py`**: Combine curated Parquet with **generated MCQs** (verification-ready).

  * **How to use**:

    ```bash
    python combine_generated_MCQ_parquet.py \
        --orig_parquet_dir /path/to/output/parquet \
        --gen_jsonl_path /generated/MCQs/jsonl \
        --out_dir /generated/MCQs/parquet
    ```

* **`compute_rubric.py`**: Compute rubric verification scores of generated MCQs (no files written; for probing your verify rubric quality).

* **`anotate_rubric_score.py`**: Join the verifier outputs with generated MCQs; produces a JSONL containing question/options/answer and rubric outputs.

  * **How to use**:

    ```bash
    python anotate_rubric_score.py \
        --input /path/to/verify/output/results.jsonl \
        --output /path/to/output/anotated/with/verifed/score/results.jsonl \
        --score_type normalized
    ```

* **`filter_by_score.py`**: Filter verified MCQs by a threshold. Use `--require_pass` to keep only items that fully pass the **essential** criteria. (Recommend inspecting score distributions with `compute_rubric.py` first.)

  * **How to use**:

    ```bash
    python filter_by_score.py \
        --input /path/to/output/anotated/with/verifed/score/results.jsonl \
        --output /path/to/filtered/MCQs/result.jsonl \
        --min_score 0.90 \
        --require_pass
    ```

* **`rebalance_vqa.py`**: Rebalance answer options to reduce label bias.

  * **How to use**:

    ```bash
    python rebalance_vqa.py \
        --input /path/to/filtered/MCQs/result.jsonl \
        --output /path/to/filtered/MCQs/balanced/result.jsonl
    ```

* **`rebalance_vqa_pmc.py`**: Rebalance answer options for **PMC-VQA style** (ABCD).

  * **How to use**:

    ```bash
    python rebalance_vqa_pmc.py \
        --input /path/to/PMCVQAstyle/MCQs/result.jsonl \
        --output /path/to/PMCVQAstyle/MCQs/balanced/result.jsonl
    ```

* **`extract_vqa_subset_jsonl.py`**: Extract balanced subsets (e.g., 1k/2k/5k/10k) from the whole set; balances by answer option **A–E**.

  * **How to use**:

    ```bash
    python extract_vqa_subset_jsonl.py \
        --input /path/to/filtered/MCQs/balanced/result.jsonl \
        --output /path/to/filtered/MCQs/balanced/subset/result.jsonl \
        -k 5000
    ```

* **`extract_vqa_subset_jsonl_pmc.py`**: Extract balanced subsets for **PMC-VQA style** (answers **A–D** only).

  * **How to use**:

    ```bash
    python extract_vqa_subset_jsonl_pmc.py \
        --input /path/to/PMCVQAstyle/MCQs/balanced/result.jsonl \
        --output /path/to/PMCVQAstyle/MCQs/balanced/subset/result.jsonl \
        -k 5000
    ```

* **`construct_trainset.py`**: Combine filtered & rebalanced (optionally subset) MCQs **with images** and write final **Parquet**.

  * **How to use**:

    ```bash
    python construct_trainset.py \
      --jsonl /path/to/filtered/MCQs/balanced/subset/result.jsonl \
      --parquet_dir /path/to/output/parquet \
      --out_parquet /path/to/final/MCQs/parquet \
      --dataset_name dataset_name
    ```

* **`construct_trainset_pmc.py`**: Same as above for **PMC-VQA style**.

  * **How to use**:

    ```bash
    python construct_trainset_pmc.py \
      --jsonl /path/to/PMCVQAstyle/MCQs/balanced/subset/result.jsonl \
      --parquet_dir /path/to/output/parquet \
      --out_parquet /path/to/final/MCQs/parquet \
      --dataset_name dataset_name
    ```

* **`construct_trainset_cot.py`**: Construct final Parquet **with reasoning** (SFT-focused variant).

  * **How to use**:

    ```bash
    python construct_trainset_cot.py \
      --jsonl /path/to/filtered/MCQs/balanced/subset/result.jsonl \
      --parquet_dir /path/to/output/parquet \
      --out_parquet /path/to/final/MCQs/parquet \
      --dataset_name dataset_name
    ```

### generate

Code for MCQ generation (image + caption + context) with our **generation rubric** prompts:

* `glm_generate.py`: GLM-4.5V 108B
* `internvl_generate.py`: InternVL-3.5 38B
* `qwen_generate.py`: Qwen-2.5-VL 72B
* `glm_generate_pmc.py`: GLM-4.5V 108B with **PMC-VQA style** prompt

### verify

MCQ verification with our **verification rubric** prompts:

* `glm_verify.py`: GLM-4.5V 108B
* `qwen_verify.py`: Qwen-2.5-VL 72B

### cot

SFT data curation:

* `glm_cot.py`: Generate reasoning **CoT** with GLM-4.5V 108B.
* `add_cot.py`: Keep correctly-answered items with parsable reasoning and append **CoT** to the MCQ JSONL.

  * **How to use**:

    ```bash
    python add_cot.py \
      --eval_jsonl /eval/result/with/cot/jsonl \
      --meta_jsonl /original/MCQ/jsonl \
      --out_jsonl /output/MCQ/with/cot/jsonl \
      --use_answer_to_break_ties
    ```

### attribute

* `glm_type.py`: Assign **question type**.
* `glm_modality.py`: Assign **modality** and **anatomy**.
* `fix/`: Tools to locate and manually fix unparsable or problematic records.

### scripts

Shell entrypoints to run the steps end-to-end.

## Examples for the pipeline

### GLM generate, Qwen verify, subset5k:

```bash
python curate_data_json_mp_wholeset.py \
    --base /home/efs/nwang60/datasets/biomedica_webdataset_24M \
    --out  /home/efs/nwang60/datasets/biomedica_25k/output.json \
    --total 25000 \
    --workers 16

python convert_json_parquet_mp.py \
    --base /home/efs/nwang60/datasets/biomedica_webdataset_24M \
    --input /home/efs/nwang60/datasets/biomedica_25k/output.json \
    --out /home/efs/nwang60/datasets/biomedica_webdataset_parquet_25k \
    --rows-per-shard 10000 \
    --workers 8

bash synthesis/scripts/glm_generate.sh

python combine_generated_MCQ_parquet.py \
    --orig_parquet_dir /home/efs/nwang60/datasets/biomedica_webdataset_parquet_25k \
    --gen_jsonl_path outputs/biomedica_vqa_25k_glm/results.jsonl \
    --out_dir /home/efs/nwang60/datasets/biomedica_webdataset_glm_VQA_parquet_25k

bash synthesis/scripts/qwen_verify.sh

python anotate_rubric_score.py \
    --input outputs_verify/biomedica_vqa_glm_generated_qwen_verified_25k/results.jsonl\
    --output /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_glm_verified_VQA_json_25k/biomedica_webdataset_glm_generated_glm_verified_VQA_json_25k_score.jsonl \
    --score_type normalized

python filter_by_score.py \
    --input /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_score.jsonl \
    --output /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_filtered_mean9670.jsonl\
    --min_score 0.9670 \
    --require_pass

python extract_vqa_subset_jsonl.py \
    --input /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_filtered_mean9670_13k_rebalanced.jsonl \
    --output /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_filtered_mean9670_13k_rebalanced_subset5k.jsonl \
    -k 5000

python construct_trainset.py \
    --jsonl /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k/biomedica_webdataset_glm_generated_qwen_verified_VQA_json_25k_filtered_mean9670_13k_rebalanced_subset5k.jsonl \
    --parquet_dir /home/efs/nwang60/datasets/biomedica_webdataset_glm_VQA_parquet_25k \
    --out_parquet /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k.parquet \
    --dataset_name biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset5k
```