import dotenv

dotenv.load_dotenv(override=True)


import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import json

import transformers
import trl
from datasets import DatasetDict, concatenate_datasets, load_dataset
from qwen_vl_utils import process_vision_info


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    block_size: int = field(default=32768)
    train_file_path: Optional[str] = field(
        default="med-vlrm/med-vlm-pmc_vqa-gpt_4o_reasoning-tokenized"
    )
    dagger: bool = field(default=False)
    use_flash_attention_2: bool = field(default=False)


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if config.use_flash_attention_2:
        logging.info("Use flash_attention_2")
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        logging.info("Disable flash_attention_2")

    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs.update(
            {
                "device_map": "auto",
                "torch_dtype": "auto",
                "use_cache": False,
            }
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name, **kwargs
        )
    else:
        # NOTE xk: In s1, flash-attn is not used.
        # kwargs = {"torch_dtype": "auto", "attn_implementation": "flash_attention_2", "use_cache": False}
        kwargs = {}
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model_name, **kwargs
            )
        except Exception as e:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_name, **kwargs
            )

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     config.model_name, use_fast=True
    # )
    processor = transformers.AutoProcessor.from_pretrained(config.model_name)

    # if "Llama" in config.model_name:
    #     instruction_template = "<|start_header_id|>user<|end_header_id|>"
    #     response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    #     # Use a token that is never used
    #     tokenizer.pad_token = "<|reserved_special_token_5|>"
    # elif "Qwen" in config.model_name:
    #     instruction_template = "<|im_start|>user"
    #     response_template = "<|im_start|>assistant\n"
    #     # Use a token that is never used
    #     tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    # collator = trl.DataCollatorForCompletionOnlyLM(
    #     instruction_template=instruction_template,
    #     response_template=response_template,
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )
    def collator(examples):
        text_list = []
        image_inputs_list = []
        for example in examples:
            messages = build_message(example)

            text = processor.apply_chat_template(messages, tokenize=False)
            text_list.append(text)

            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs)

        batch = processor(
            text=text_list, images=image_inputs_list, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = (
            -100
        )  # Mask padding tokens in labels

        # mark the tokens before assistant response as -100
        response_start = "<|im_start|>assistant\n"

        response_start_token_ids = processor.tokenizer(
            response_start, add_special_tokens=False
        )["input_ids"]

        # find the start index of the assistant response
        batch_start_index = []
        for input_ids in batch["input_ids"]:
            # find the start idx of the response_start_token_ids in input_ids
            # response_start_token_ids is a list, all the tokens in it should be found in input_ids
            input_ids = input_ids.tolist()  # Convert tensor to list
            found = False
            for i in range(len(input_ids)):
                if (
                    input_ids[i : i + len(response_start_token_ids)]
                    == response_start_token_ids
                ):
                    batch_start_index.append(i + len(response_start_token_ids))
                    found = True
                    break
            if not found:
                raise ValueError(
                    "Could not find the start of the assistant response in input_ids."
                )

        # Set the tokens before the assistant response to -100 in labels
        for i, start_idx in enumerate(batch_start_index):
            labels[i, :start_idx] = -100

        batch["labels"] = labels  # Add labels to the batch
        return batch

    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size
    # https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py
    args.remove_unused_columns = False
    args.dataset_kwargs = {"skip_prepare_dataset": True}
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
        args=args,
        data_collator=collator,
    )

    trainer.train()
    # final_ckpt_path = os.path.join(args.output_dir, "checkpoint-final")
    # trainer.save_model(output_dir=final_ckpt_path)
    # tokenizer.save_pretrained(final_ckpt_path)
    trainer.accelerator.wait_for_everyone()


INSTRUCTION_PROMPT = r"You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags."


def build_message(sample):
    question = sample["question"]
    raw_options = sample["options"]
    options = json.loads(raw_options)

    prompt = f"Question: {question}\n\nOptions:"
    for letter, option in options.items():
        prompt += f"\n\n{letter}. {option}"
    prompt = INSTRUCTION_PROMPT + "\n\n" + prompt

    answer_label = sample["answer_label"]
    answer = sample["answer"]
    reasoning = sample["reasoning"]

    response = f"<think> {reasoning.strip()} </think>\n<answer> {answer_label.strip()} </answer>"

    images = sample.get("images", [])
    # https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl#2-load-dataset-
    images_prompt = [{"type": "image", "image": img} for img in images]

    message = [
        {"role": "user", "content": images_prompt + [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]
    return message


if __name__ == "__main__":
    train()