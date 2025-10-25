# MedVLSynther: Synthesizing High-Quality Visual Question Answering from Medical Documents with Generator-Verifier LMMs

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-orange)](https://ucsc-vlaa.github.io/MedVLSynther/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-blue)](https://huggingface.co/ns-wang)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**MedVLSynther** is a rubric-guided generator-verifier framework that synthesizes high-quality multiple-choice VQA items directly from open biomedical literature by conditioning on figures, captions, and in-text references. Applying this pipeline to PubMed Central yields MedVLSynther-13K: 13,087 audited questions over 14,803 images spanning 13 imaging modalities and 28 anatomical regions. Training open-weight LMMs with reinforcement learning using verifiable rewards **improves accuracy across six medical VQA benchmarks**, achieving averages of **55.85** (3B) and **57.56** (7B), with up to **77.21** on VQA-RAD and 66.36 on PathVQA, outperforming strong medical LMMs.

## 🔥 Highlights

- **Fully open stack** — End-to-end release of code, data curation scripts, checkpoints, and evaluation to enable full reproduction and auditing.

- **Automatic, open-sourced pipeline** — A rubric-guided generator–verifier workflow turns figures + captions into exam-quality MCQs with minimal manual effort, and is designed for easy extension.

- **Contamination analysis assurance** — We audit potential train/test overlap at both text and image levels; under our protocol, we find **no** leakage between our training data and evaluation suites.

- **Effective in practice** — Training open-weight LMMs on our verified synthetic data yields consistent gains across standard medical VQA benchmarks.

## 📋 Table of Contents

- [Installation](#-Installation)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Models and Results](#-models-and-results)
- [Citation](#-citation)

## 🚀 Installation

### Prerequisites

- Python 3.10
- CUDA 12.1 or later

```
git clone 
conda create -n medvlsynther python==3.10
conda activate medvlsynther
```

### Training Environment

We use **verl** for GRPO and **trl** for SFT.

GRPO:

```bash
conda activate medvlsynther
git clone https://github.com/volcengine/verl.git
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

trl:

```bash
conda create -n medvlsynther_sft python==3.10
conda activate medvlsynther_sft
# Install torch according to your own cuda version
pip install trl transformers
```

### Synthesis Environment

Because GLM‑4.5V requires recent vLLM and transformers, we recommend using the SFT (TRL) environment for the entire synthesis pipeline.

## 🚀 Installation

### Demo

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load the model
model_name="ns-wang/MedVLSynther-7B-RL_13K"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# Example usage
messages_1 = [
    {
        "role": "system",
        "content": "You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/7bMMMU.png",
            },
            {"type": "text", "text": "This line of of myelinated axons in layer IV of visual cortex represents the axons of cells in the Choices: (A) Superior colliculus. (B) Lateral geniculate.(C) Retina. (D) Medial geniculate."},
        ],
    }
]

messages_2 = [
    {
        "role": "system",
        "content": "You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/7bslake.png",
            },
            {"type": "text", "text": "Does the picture contain kidney? Choices: (A) Yes (B) No"},
        ],
    }
]

# Preparation for inference
messages = messages_2

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.6, top_p=0.95, do_sample=True)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## 📊 Datasets

### Available Datasets

We release **MedVLSynther-13K** and the subsets used in our paper. Each set targets medical vision–language QA and supports RLVR/SFT training.

| Dataset | Generator | Verifier | Modality | Description | Download |
|---|---|---|---|---|---|
| **MedVLSynther-13K** | GLM-4.5V 108B | Qwen2.5-VL 72B | Image–Text | Full training set for medical VQA (used for RLVR). | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-13K) |
| **MedVLSynther-10K** | GLM-4.5V 108B | Qwen2.5-VL 72B | Image–Text | 10K-sample training subset for RLVR. | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-10K) |
| **MedVLSynther-5K**  | GLM-4.5V 108B | Qwen2.5-VL 72B | Image–Text | 5K-sample training subset for RLVR. | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-5K) |
| **MedVLSynther-2K**  | GLM-4.5V 108B | Qwen2.5-VL 72B | Image–Text | 2K-sample training subset for RLVR. | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-2K) |
| **MedVLSynther-1K**  | GLM-4.5V 108B | Qwen2.5-VL 72B | Image–Text | 1K-sample training subset for RLVR. | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-1K) |
| **MedVLSynther-5K-qwen-glm** | Qwen2.5-VL 72B | GLM-4.5V 108B | Image–Text | 5K subset for **generator and verifier choice** ablation (GLM→Qwen generator, Qwen→GLM verifier). | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-5K-qwen-glm) |
| **MedVLSynther-5K-internvl-glm** | InternVL-3.5 38B | GLM-4.5V 108B | Image–Text | 5K subset for **generator choice** ablation (InternVL→GLM verifier). | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-5K-internvl-glm) |
| **MedVLSynther-5K-glm-glm** | GLM-4.5V 108B | GLM-4.5V 108B | Image–Text | 5K subset for **verifier choice** ablation (Qwen→GLM verifier). | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-5K-glm-glm) |
| **MedVLSynther-5K-no-verify** | GLM-4.5V 108B | N/A | Image–Text | 5K subset for **verifier necessity** ablation (no verification step). | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-5K-no-verify) |
| **MedVLSynther-5K-PMC-style** | GLM-4.5V 108B | N/A | Image–Text | 5K subset generated with **PMC-VQA–style** prompts. | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-5K-PMC-style) |
| **MedVLSynther-5K-SFT** | GLM-4.5V 108B | N/A | Image–Text | 5K subset generated for SFT training. | [🤗 HF](https://huggingface.co/datasets/ns-wang/MedVLSynther-5K-SFT) |

### Dataset Usage

```python
from datasets import load_dataset

# Load evaluation dataset
eval_dataset = load_dataset("UCSC-VLAA/MedVLThinker-Eval")

# Load training dataset
TBD
```

For dataset details and utilizing the synthesis pipeline, please refer to [synthesis/README.md](synthesis/README.md) and [data_process/README.md](data_process/README.md).


<details><summary>Dataset details and preparation of your own</summary>

### Data Format

All train datasets follow a unified format, just the same as MedVLThinker:

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

### Prepare Evaluation Data

Please download [MedVLThinker-Eval](https://huggingface.co/datasets/UCSC-VLAA/MedVLThinker-Eval).

### Prepare Training Data

Please download the dataset you want to use above, e.g., MedVLSynther-13K:

```bash
huggingface download ...
```

Prepare it for verl format:

```bash
python data_process/prep_to_hf_bytes.py \
    --parquet_glob "data/MedVLSynther-13K/*.parquet" \
    --out_dir data/MedVLSynther-13K_hf \
    --num_proc 32 --strict_image --keep_first_k_images 6

python data_process/convert_verl_format.py \
    --local_data_dir data/MedVLSynther-13K_hf \
    --data_source MedVLSynther-13K \
    --ability medical_mcqa \
    --split train \
    --output_dir data/MedVLSynther-13K_verl \
    --num_proc 32
```

</details>

## 🏋️ Training

### Reinforcement Learning (GRPO)

Please refer to [train](train).

After training, you need to convert verl checkpoints for inference:

```bash
python -m verl.model_merger merge --backend fsdp --local_dir /path/to/checkpoints/global_step_xxx/actor --target_dir /path/to/converted/checkpoints
```

### Supervised Fine-tuning (SFT)

```bash
bash train/sft/train_commands.sh
```

## Evaluation

### Evaluation for trained model

Please refer to [eval/README.md](eval/README.md).

## 📈 Models and Results 

### Available Models

| Model | Size | RL/SFT | Training Data | Download |
|-------|------|----------------|---------------|----------|
| **RL Models** |
| MedVLSynther-3B-RL_1K | 3B | RL | MedVLSynther-1K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_1K) |
| MedVLSynther-3B-RL_2K | 3B | RL | MedVLSynther-2K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_2K) |
| MedVLSynther-3B-RL_5K | 3B | RL | MedVLSynther-5K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_5K) |
| MedVLSynther-3B-RL_10K | 3B | RL | MedVLSynther-10K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_10K) |
| MedVLSynther-3B-RL_13K | 3B | RL | MedVLSynther-13K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_13K) |
| MedVLSynther-3B-RL_5K_qwen-glm | 3B | RL | MedVLSynther-5K-qwen-glm | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_5K_qwen-glm) |
| MedVLSynther-3B-RL_5K_internvl-glm | 3B | RL | MedVLSynther-5K-internvl-glm | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_5K_internvl-glm) |
| MedVLSynther-3B-RL_5K_glm-glm | 3B | RL | MedVLSynther-5K-glm-glm | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_5K_glm-glm) |
| MedVLSynther-3B-RL_5K_no-verify | 3B | RL | MedVLSynther-5K-no-verify | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_5K_no-verify) |
| MedVLSynther-3B-RL_5K_PMC-style | 3B | RL | MedVLSynther-5K-PMC-style | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-3B-RL_5K_PMC-style) |
| MedVLSynther-7B-RL_1K | 7B | RL | MedVLSynther-1K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_1K) |
| MedVLSynther-7B-RL_2K | 7B | RL | MedVLSynther-2K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_2K) |
| MedVLSynther-7B-RL_5K | 7B | RL | MedVLSynther-5K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_5K) |
| MedVLSynther-7B-RL_10K | 7B | RL | MedVLSynther-10K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_10K) |
| MedVLSynther-7B-RL_13K | 7B | RL | MedVLSynther-13K | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_13K) |
| MedVLSynther-7B-RL_5K_qwen-glm | 7B | RL | MedVLSynther-5K-qwen-glm | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_5K_qwen-glm) |
| MedVLSynther-7B-RL_5K_internvl-glm | 7B | RL | MedVLSynther-5K-internvl-glm | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_5K_internvl-glm) |
| MedVLSynther-7B-RL_5K_glm-glm | 7B | RL | MedVLSynther-5K-glm-glm | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_5K_glm-glm) |
| MedVLSynther-7B-RL_5K_no-verify | 7B | RL | MedVLSynther-5K-no-verify | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_5K_no-verify) |
| MedVLSynther-7B-RL_5K_PMC-style | 7B | RL | MedVLSynther-5K-PMC-style | [🤗 HF](https://huggingface.co/ns-wang/MedVLSynther-7B-RL_5K_PMC-style) |
| **SFT Models** |
| MedVLThinker-3B-SFT_5K | 3B | SFT | MedVLSynther-5K-SFT | [🤗 HF](https://huggingface.co/ns-wang/MedVLThinker-3B-SFT_5K) |
| MedVLThinker-7B-SFT_5K | 7B | SFT | MedVLSynther-5K-SFT | [🤗 HF](https://huggingface.co/ns-wang/MedVLThinker-7B-SFT_5K) |

### Benchmark Results

Comparison with other methods.

| Model                      | PMC    | MMMU   | MedX-M | PathVQA | SLAKE  | VQA-Rad |  Avg.   |
| :------------------------- | :----: | :----: | :----: | :-----: | :----: | :-----: |  :----: |
| General LMM                |        |        |        |         |        |         |         |
| Gemme 3 4B                 | 44\.42 | 46\.67 | 21\.89 | 59\.24  | 66\.59 | 56\.86  |  49\.28 |
| Qwen2\.5-VL-3B-Instruct    | 44\.77 | 44\.12 | 20\.69 | 61\.96  | 61\.30 | 62\.01  |  49\.14 |
| Qwen2\.5-VL-7B-Instruct    | 49\.30 | 52\.94 | 18\.89 | 65\.39  | 65\.71 | 68\.75  |  53\.50 |
| Medical LMM                |        |        |        |         |        |         |         |
| MedGemma 4B                | 42\.73 | 32\.55 | 8\.17  | 59\.64  | 83\.49 | 78\.55  |  50\.86 |
| MedGemma 27B               | 36\.75 | 35\.88 | 12\.13 | 62\.09  | 77\.40 | 72\.67  |  49\.49 |
| Llava Med v1\.5 Mistral 7B | 34\.28 | 31\.37 | 22\.56 | 56\.52  | 62\.82 | 56\.74  |  44\.05 |
| HuatuoGPT-Vision-7B        | 53\.39 | 50\.59 | 22\.00 | 63\.53  | 75\.00 | 63\.60  |  54\.69 |
| MedVLThinker-3B            | 47\.32 | 52\.16 | 22\.90 | 62\.28  | 63\.38 | 71\.08  |  53\.19 |
| MedVLThinker-7B            | 50\.67 | 56\.86 | 24\.43 | 66\.83  | 65\.79 | 64\.71  |  54\.88 |
| MedVLSynther-3B            | 50\.23 | 52\.35 | 21\.40 | 62\.82  | 74\.76 | 73\.53  |  55\.85 |
| MedVLSynther-7B            | 53\.78 | 57\.06 | 23\.15 | 66\.36  | 67\.79 | 77\.21  |  57\.36 |

## 📁 Project Structure

```
MedVLSynther/
├── analysis/          # Result analysis
├── assets/            # Assets for this project
├── contamination/     # Contamination analysis
├── data_process/      # Data preprocessing and preparation
├── eval/              # Evaluation scripts and benchmarks
├── synthesis/         # Data synthesis
├── train/             # Training scripts and configurations
└── README.md          # This file
```

## 📚 Citation



