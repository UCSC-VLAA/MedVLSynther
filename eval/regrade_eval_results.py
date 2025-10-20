import dotenv

dotenv.load_dotenv(override=True)

import collections
import json
import re
from pathlib import Path
from types import SimpleNamespace

import click
import datasets
import numpy as np
import pandas as pd
import tqdm


def _extract_answer(text: str) -> str:
    """Extract the model’s final outputs."""
    m = list(re.finditer(r"<answer>(.*?)</answer>", text, re.S))
    if m:
        text = m[-1].group(1).strip()
        first_line = re.search(r"\s*([^\n\r]+)", text)
        if first_line:
            return first_line.group(1).strip()
        else:
            return text

    m = list(re.finditer(r"answer:\s*(.*)\s*", text, re.I))
    if m:
        return m[-1].group(1).strip()

    m = list(re.finditer(r"answer is:\s*(.*)\s*", text, re.I))
    if m:
        return m[-1].group(1).strip()

    m = list(re.finditer(r"answer is\s*(.*)\s*", text, re.I))
    if m:
        return m[-1].group(1).strip()

    # m = list(re.finditer(r"answer\s*(.*)\s*", text, re.I))
    # if m:
    #     return m[-1].group(1).strip()

    m = list(re.finditer(r"is:\s*(.*)\s*", text, re.I))
    if m:
        return m[-1].group(1).strip()

    return text.strip()


def extract_answer(text):
    text = _extract_answer(text)
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    text = text.replace("<answer>", "")
    text = text.replace("</answer>", "")
    text = text.strip()
    return text


def grade_answer(prediction, answer, answer_label=None, llava_med_rule=False):
    if llava_med_rule:
        # NOTE(xk): llava med cannot follow the instruction about output format, thus we use such a loose rule
        if answer.lower() in prediction.lower():
            return True
    if answer_label is not None:
        if prediction.strip().lower() == f"{answer_label}. {answer}".strip().lower():
            return True
        elif prediction.strip().lower() == answer_label.strip().lower():
            return True

    if prediction.strip().lower() == answer.strip().lower():
        return True

    return False


FAILED_TO_CONVERT = []


def regrade_data(data, options_dataset, llava_med_rule=False):
    prompts = data["prompts"]
    answer = data["answer"]
    answer_label = data["answer_label"]
    dataset_index = data["dataset_index"]
    num_rollouts = data["num_rollouts"]
    # to be updated
    # num_correct = data["num_correct"]
    options = options_dataset[dataset_index]
    options = json.loads(options)

    num_correct = 0
    pred_letter_list = []
    for parsed_output in data["parsed_outputs"]:
        output_text = parsed_output["output_text"]
        # to be updated
        # pred_letter = parsed_output["pred_letter"]
        # to be updated
        # is_correct = parsed_output["is_correct"]

        # re-extract answer label with new regex rules
        pred_letter = extract_answer(output_text)
        converted_pred_letter = pred_letter
        is_convert = False
        for option_answer_label, option_answer in options.items():
            if grade_answer(
                pred_letter, option_answer, option_answer_label, llava_med_rule
            ):
                converted_pred_letter = option_answer_label
                is_convert = True
                break
        if not is_convert:
            FAILED_TO_CONVERT.append(
                {
                    "pred_letter": pred_letter,
                    "answer": answer,
                    "answer_label": answer_label,
                    "dataset_index": dataset_index,
                    "output_text": output_text,
                    "prompts": prompts,
                }
            )

        parsed_output["pred_letter"] = converted_pred_letter
        if grade_answer(pred_letter, answer, answer_label, llava_med_rule):
            parsed_output["is_correct"] = True
            num_correct += 1
        else:
            parsed_output["is_correct"] = False

        pred_letter_list.append(converted_pred_letter)

    data["prev_num_correct"] = data.get("num_correct", -1)
    data["num_correct"] = num_correct
    majority_vote_pred_letter = collections.Counter(pred_letter_list).most_common(1)[0][
        0
    ]
    data["majority_vote_pred_letter"] = majority_vote_pred_letter
    data["average_num_correct"] = num_correct / num_rollouts
    data["pass_at_num_rollouts"] = num_correct > 0
    data["majority_at_num_rollouts"] = majority_vote_pred_letter == answer_label
    data["options"] = options

    return data


@click.command()
@click.option(
    "--input_results_jsonl",
    "-i",
    default="outputs/temp_0.5-n_5/v0/Qwen2.5-VL-7B-Instruct/eval_results.jsonl",
    help="Path to the results JSONL file.",
)
@click.option(
    "--dataset_name",
    default="med-vlrm/med-vlm-eval-v2",
    help="Name of the dataset to load.",
)
@click.option("--llava_med_rule", is_flag=True, help="Use Llava Med rules for grading.")
@click.option("--split", default="test", help="Dataset split to use.")
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    dataset_name = args.dataset_name
    split = args.split
    dataset = datasets.load_dataset(dataset_name, split=split)

    options_dataset = dataset["options"]

    results_jsonl = args.input_results_jsonl
    results_jsonl = Path(results_jsonl)
    output_results_jsonl = results_jsonl.parent / f"regraded_{results_jsonl.name}"
    llava_med_rule = args.llava_med_rule

    with open(results_jsonl, "r") as f, open(output_results_jsonl, "w") as out_f:
        for line in tqdm.tqdm(f):
            data = json.loads(line)
            regrade_data(data, options_dataset, llava_med_rule)
            out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"Regraded results saved to {output_results_jsonl}")


if __name__ == "__main__":
    main()