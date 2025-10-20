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
import transformers
import trl
from datasets import DatasetDict, concatenate_datasets, load_dataset


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    train_file_path: Optional[str] = field(default="mmqm/m194k_r1_tokenized-250224")
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
        logging.info(f"Use flash_attention_2")
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        logging.info(f"Disable flash_attention_2")

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
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True
    )
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size
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


if __name__ == "__main__":
    train()