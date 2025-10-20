import dotenv

dotenv.load_dotenv(override=True)
import json
import os
import string
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence as SequenceType

import click
import datasets
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from datasets import Image as ImageData
from PIL import Image

INSTRUCTION_PROMPT = r"You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags."


@click.command()
@click.option(
    "--local_data_dir",
    type=Path,
    default="data/local/med-vlm-m23k-qwen2_5_vl_3b-easy_to_hard",
)
@click.option("--is_local", type=bool, default=True, help="Use local data source.")
# remote
@click.option(
    "--dataset_path",
    type=str,
    default="med-vlrm/med-vlm-eval-qwen2_5_vl_size",
    help="Remote dataset path if not using local data.",
)
@click.option(
    "--dataset_subset", type=str, default=None, help="Subset of the dataset to use."
)
@click.option(
    "--dataset_split", type=str, default="test", help="Split of the dataset to use."
)
# shuffle and select dataset size
@click.option(
    "--shuffle",
    is_flag=True,
    default=False,
    help="Shuffle the dataset before processing",
)
@click.option("--seed", type=int, default=42, help="Seed for shuffling the dataset")
@click.option(
    "--dataset_size", type=int, default=None, help="Number of samples to process"
)
# data source, ability, split
@click.option(
    "--data_source", type=str, default="med-vlm-m23k-qwen2_5_vl_3b-easy_to_hard"
)
@click.option(
    "--ability", type=str, default="medical_mcqa", help="Ability to use for the dataset"
)
@click.option(
    "--split",
    type=str,
    default="train",
    help="Dataset split to process (e.g., 'train', 'test', 'validation')",
)
# number of processes to use for mapping
@click.option(
    "--num_proc", type=int, default=16, help="Number of processes to use for mapping"
)
# output directory to save the processed dataset
@click.option(
    "--output_dir",
    type=Path,
    default="data/verl/med-vlm-m23k-qwen2_5_vl_3b-easy_to_hard",
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    print(f"Arguments: {args}")

    is_local = args.is_local

    if is_local:
        local_data_dir = args.local_data_dir
        dataset = datasets.load_from_disk(local_data_dir)
        print(f"Loaded local dataset from {local_data_dir}")
    else:
        dataset_path = args.dataset_path
        dataset_subset = args.dataset_subset
        dataset_split = args.dataset_split
        dataset = datasets.load_dataset(
            dataset_path,
            subset=dataset_subset,
            split=dataset_split,
        )
        print(f"Loaded remote dataset: {dataset_path} ({len(dataset)} samples)")

    shuffle = args.shuffle
    seed = args.seed
    dataset_size = args.dataset_size
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
        print(f"Shuffled dataset with seed {seed}.")
    if dataset_size is not None:
        dataset = dataset.select(range(dataset_size))
        print(f"Selected {dataset_size} samples from the dataset.")

    data_source = args.data_source
    ability = args.ability
    split = args.split
    num_proc = args.num_proc

    dataset = dataset.map(
        make_map_fn(split, data_source, ability),
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        with_indices=True,
        keep_in_memory=True,
        desc=f"Processing {split} split",
    )

    # save to shards: https://github.com/huggingface/datasets/issues/7047#issuecomment-2233163406
    save_parquet_path = args.output_dir / f"{split}.parquet"
    dataset.to_parquet(save_parquet_path)
    print(f"Saved processed dataset to {save_parquet_path}")


def make_map_fn(split, data_source, ability):
    def process_fn(example, idx):
        # How interleave multi-modal inputs work in verl: https://github.com/volcengine/verl/blob/2aed8d0a4570166ff107b7af2c035c3a46a78101/verl/utils/dataset/rl_dataset.py#L169-L175
        """
        problem = example.pop("problem")
        prompt = instruction_following + problem
        answer = example.pop("answer")
        # only a simple image
        image = example.pop("image")
        images = [image]
        """

        question = example.pop("question")
        raw_options = example.pop("options")
        options = json.loads(raw_options)

        # build prompt.
        # we follow the format in `eval/run_offline_inference.py`
        prompt = f"Question: {question}\n\nOptions:"
        for letter, option in options.items():
            prompt += f"\n\n{letter}. {option}"
        prompt = INSTRUCTION_PROMPT + "\n\n" + prompt

        images = example.pop("images")
        if images is not None:
            if not isinstance(images, SequenceType):
                raise ValueError(
                    f"Expected 'images' to be a sequence, got {type(images)}: {images}"
                )

            prompt = "".join(["<image>"] * len(images)) + prompt
            image_data = {
                "images": images,
            }
        else:
            image_data = {}

        answer = example.pop("answer")
        answer_label = example.pop("answer_label")

        dataset_name = example.pop("dataset_name")
        dataset_index = example.pop("dataset_index")

        data = {
            # indexing reward function: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # image data
            **image_data,
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                # dataset original
                "question": question,
                "options": raw_options,
                "answer_label": answer_label,
                "answer": answer,
                "dataset_name": dataset_name,
                "dataset_index": dataset_index,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    main()