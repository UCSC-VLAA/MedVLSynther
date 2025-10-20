"""
FreedomIntelligence/HuatuoGPT-Vision-34B-hf
FreedomIntelligence/HuatuoGPT-Vision-7B-hf
Eren-Senoglu/llava-med-v1.5-mistral-7b-hf  # manually set `config.json`: "head_dim": 128,
google/medgemma-4b-it  # add model_prompt_type="gemma3" to the command line args
google/medgemma-27b-it  # add model_prompt_type="gemma3" to the command line args


Debug:

wget https://www.nibib.nih.gov/sites/default/files/inline-images/CT-abdomen-400x267.jpg -O misc/ct.jpg

image = Image.open("misc/ct.jpg")
messages = [{
        "prompt": "<start_of_image>\nWhat is this?",
        "multi_modal_data": {"image": [image]},
}]
output = llm.generate(messages)

# gemma 3 prompt: https://github.com/vllm-project/vllm/blob/9f414a12adb991d04d2adf0b80f1f115d6281fad/examples/offline_inference/vision_language.py#L208-L210
image = Image.open("misc/ct.jpg")
question = "What is it?"
messages = [{
        "prompt": "<bos><start_of_turn>user\n"
            f"<start_of_image><start_of_image>{question}<end_of_turn>\n"
            "<start_of_turn>model\n",
        "multi_modal_data": {"image": [image, image]},
}]
output = llm.generate(messages)
"""

import dotenv

dotenv.load_dotenv(override=True)

import json
import multiprocessing as mp
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # 关键：vLLM 内部 worker 用 spawn
import io
import pyarrow as pa
import re
import shutil
import traceback
import types
from math import ceil
from pathlib import Path

import click
import pandas as pd
from datasets import load_dataset
from merge_results import compute_results_acc, merge_output
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import trange
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


def main(**kwargs):
    try:
        _main(**kwargs)
    except Exception as e:
        global_dp_rank = -1
        if "global_dp_rank" in kwargs:
            global_dp_rank = kwargs["global_dp_rank"]

        print(f"Rank [{global_dp_rank}]:  Exception occurred: {e}")
        traceback.print_exc()

        exit(1)


def _main(
    *,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    tp_size,
    args,
    barrier,
):

    # NOTE(xk): vllm does not support DP well, so we do not use it.
    # Guess: The last batch with different number of samples causes the halt.

    # os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    # os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    # os.environ["VLLM_DP_SIZE"] = str(dp_size)
    # os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    # os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the

    # gpu_ids = range(local_dp_rank * tp_size, (local_dp_rank + 1) * tp_size)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    # print(f"Rank [{global_dp_rank}]: Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    # 计算本进程要用的本地 GPU 列表（尊重外部 CUDA_VISIBLE_DEVICES）
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        all_ids = [int(x) for x in visible.split(",") if x.strip() != ""]
    else:
        # 没设置就用物理 0..N-1
        import torch
        all_ids = list(range(torch.cuda.device_count()))

    start = local_dp_rank * tp_size
    end = (local_dp_rank + 1) * tp_size
    sel = all_ids[start:end]
    assert len(sel) == tp_size, f"Not enough GPUs: need {tp_size}, got {sel} from {all_ids}"

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, sel))
    print(f"Rank [{global_dp_rank}]: Using GPUs (physical ids): {os.environ['CUDA_VISIBLE_DEVICES']}")
    # engine processes.

    # Sample prompts.
    # ---------- Load dataset ----------
    dataset_name = args.dataset_name
    subset = args.subset
    split = args.split
    dataset_size = args.dataset_size
    num_proc = args.num_proc

    ds = load_dataset(dataset_name, subset)[split]
    if dataset_size:
        ds = ds.select(range(dataset_size))

    # test dataloading
    model = args.model
    processor = None
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model)
    except Exception:
        # InternVL 系没有 HF Processor，vLLM 自带处理器，直接跳过即可
        pass


    build_prompt(ds[0], tokenizer, args)

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    promts_per_rank = ceil(len(ds) / dp_size)
    start = global_dp_rank * promts_per_rank
    end = min(start + promts_per_rank, len(ds))
    ds = ds.select(range(start, end))

    output_dir = Path(args.output_dir)
    output_dir = output_dir / "shards"
    out_file = output_dir / f"dp_{global_dp_rank}.jsonl"
    if out_file.exists() and not args.overwrite:
        dataset_index_set = set()
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    dataset_index_set.add(result["dataset_index"])
        original_len_ds = len(ds)
        # filtered already processed dataset
        ds = ds.filter(
            lambda row: row["dataset_index"] not in dataset_index_set,
            num_proc=num_proc,
            keep_in_memory=True,
        )
        new_len_ds = len(ds)
        print(
            f"Rank [{global_dp_rank}]: Filtered dataset from {original_len_ds} to {new_len_ds} records."
        )

    if len(ds) == 0:
        print(f"Rank [{global_dp_rank}]: have no data; exiting.")
        barrier.wait()
        return

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens
    n = args.n
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Create an LLM.
    model = args.model
    enforce_eager = args.enforce_eager
    trust_remote_code = args.trust_remote_code
    gpu_memory_utilization = args.gpu_memory_utilization
    max_model_len = args.max_model_len
    dtype = args.dtype
    seed = args.seed
    
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enforce_eager=enforce_eager,
        # enable_expert_parallel=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        seed=seed,
        limit_mm_per_prompt={"image": args.max_images},
    )

    # Print the outputs.
    batch_size = args.batch_size
    for start_idx in trange(
        0,
        len(ds),
        batch_size,
        unit_scale=batch_size,
        desc=f"[Global DP Rank {global_dp_rank}] Processing dataset",
    ):
        end_idx = min(len(ds), start_idx + batch_size)

        ds_chunk = ds.select(range(start_idx, end_idx))
        prompts = [build_prompt(row, tokenizer, args) for row in ds_chunk]
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        results = []

        for idx, (row, output) in enumerate(zip(ds_chunk, outputs)):
            # In each output, it consists of multiple rollouts,
            # by default it is 1.

            # metadata
            dp_index = start_idx + idx
            row_prompt = prompts[idx]["prompt"]
            dataset_name = row["dataset_name"]
            dataset_index = row["dataset_index"]

            # answer
            answer_label = row["answer_label"]
            answer = row["answer"]

            # predictions
            parsed_outputs = []

            for rollout_output in output.outputs:
                output_text = rollout_output.text.strip()

                pred_letter = extract_answer(output_text)
                is_correct = grade_answer(pred_letter, answer, answer_label)

                parsed_outputs.append(
                    {
                        "output_text": output_text,
                        "pred_letter": pred_letter,
                        "is_correct": is_correct,
                    }
                )

            # stats
            num_rollouts = len(parsed_outputs)
            num_correct = sum(1 for o in parsed_outputs if o["is_correct"])

            results.append(
                {
                    # metadata
                    "dp_index": dp_index,
                    "prompts": row_prompt,
                    "dataset_name": dataset_name,
                    "dataset_index": dataset_index,
                    # answer
                    "answer_label": answer_label,
                    "answer": answer,
                    # predictions
                    "parsed_outputs": parsed_outputs,
                    # stats
                    "num_rollouts": num_rollouts,
                    "num_correct": num_correct,
                }
            )

        output_dir = Path(args.output_dir)
        output_dir = output_dir / "shards"
        output_dir.mkdir(parents=True, exist_ok=True)

        out_file = output_dir / f"dp_{global_dp_rank}.jsonl"
        print(f"Saving results to '{out_file}'...")
        with open(out_file, "a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} records to '{out_file}'.")

    output_dir = Path(args.output_dir)
    output_dir = output_dir / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"dp_{global_dp_rank}.jsonl"
    if not out_file.exists():
        if barrier is not None:
            barrier.wait()
        print(f"Rank [{global_dp_rank}]: No output file found. Exiting.")
        return

    out_acc_file = out_file.parent / f"acc-{out_file.stem}.json"
    result_acc = compute_results_acc(out_file)
    print(f"Accuracy: {result_acc}")
    with open(out_acc_file, "w", encoding="utf-8") as f:
        json.dump(result_acc, f, indent=2, ensure_ascii=False)
    print(f"Saved accuracy to '{out_acc_file}'.")

    # NOTE(xk) Wait for all processes to finish before exiting.
    # Otherwise, the main process (using pytorch dist) may exit before all processes finish writing.
    if barrier is not None:
        barrier.wait()


def extract_answer(text: str) -> str:
    """Extract the model’s final outputs."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return m.group(1).strip() if m else text.strip()


def grade_answer(prediction, answer, answer_label=None):
    if answer_label is not None:
        if prediction.strip().lower() == f"{answer_label}. {answer}".strip().lower():
            return True
        elif prediction.strip().lower() == answer_label.strip().lower():
            return True

    if prediction.strip().lower() == answer.strip().lower():
        return True

    return False


def build_prompt(row, tokenizer, args):
    item = build_messages(row, tokenizer, args)
    if getattr(args, "debug", False):
        preview = item["prompt"]
        nimg = len(item.get("multi_modal_data", {}).get("image", []))
        print(f"Prompt preview (first 400 chars):\n{preview[:400]}...\n#images={nimg}\n")
    return item


INSTRUCTION_PROMPT = r"You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags."

def _ensure_pil_images(images_field, base_dir=None):
    """把 parquet 里的 images 字段转成 List[PIL.Image]。支持 dict{bytes,path}/path/Arrow 包装/PIL。"""
    if images_field is None:
        return []
    out = []
    for it in images_field:
        if isinstance(it, pa.lib.StructValue):
            it = it.as_py()
        if isinstance(it, Image.Image):
            out.append(it.convert("RGB")); continue
        if isinstance(it, dict):
            b = it.get("bytes"); p = it.get("path")
            if b is not None:
                if isinstance(b, (memoryview, pa.lib.Buffer)):
                    b = bytes(b) if isinstance(b, memoryview) else b.to_pybytes()
                out.append(Image.open(io.BytesIO(b)).convert("RGB")); continue
            if p:
                if not os.path.isabs(p) and base_dir:
                    p = os.path.join(base_dir, p)
                out.append(Image.open(p).convert("RGB")); continue
        if isinstance(it, (str, os.PathLike)) and os.path.exists(it):
            out.append(Image.open(it).convert("RGB")); continue
        raise ValueError(f"Unrecognized image entry: {type(it)} -> {it}")
    return out

# def build_messages(row, args):
#     question = row["question"]
#     raw_options = row["options"]
#     options = json.loads(raw_options)
#     system_block = f"System:\n{R1_SYSTEM_PROMPT}\n"

#     prompt = f"Question: {question}\n\nOptions:"
#     for letter, option in options.items():
#         prompt += f"\n\n{letter}. {option}"
#     instrution_prompt = getattr(args, "instruction_prompt", None)
#     if instrution_prompt is None:
#         instrution_prompt = INSTRUCTION_PROMPT
#     prompt = instrution_prompt + "\n\n" + prompt

#     images = row.get("images", None)
#     if images is None or args.ignore_image:
#         images_input = {}
#     else:
#         # NOTE(xk) convert the PIL images to RGB format, gemma processor only accepts RGB images
#         images = [image.convert("RGB") for image in images]

#         images_input = {
#             "multi_modal_data": {"image": images},
#         }
#         # NOTE(xk): the prompt format of different multi-modal llm is different
#         # check here for more: https://github.com/vllm-project/vllm/blob/9f414a12adb991d04d2adf0b80f1f115d6281fad/examples/offline_inference/vision_language.py#L208-L210
#         model_prompt_type = getattr(args, "model_prompt_type", None)
#         if model_prompt_type is None:
#             prompt = "<image>\n" * len(images) + prompt
#             messages = [
#                 [{"role": "system", "content": R1_SYSTEM_PROMPT}],
#                 [{"role": "user", "content": prompt}]
#             ]
#             prompt = messages
            
#         elif model_prompt_type == "gemma3":
#             prompt = (
#                 "<bos><start_of_turn>user\n"
#                 + f'{"<start_of_image>"*len(images)}'
#                 + f"{prompt}<end_of_turn>\n"
#                 + "<start_of_turn>model\n"
#             )

#         else:
#             print(f"Unknown model_prompt_type: {model_prompt_type}.")

#     return {
#         "prompt": prompt,
#         **images_input,
#     }
def build_messages(row, tokenizer, args):
    # 题干与选项
    question = row["question"]
    options = json.loads(row["options"]) if isinstance(row["options"], str) else row["options"]
    opts_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    # 图片转成 List[PIL.Image]
    # imgs_raw = row.get("images", None)
    # images = [] if imgs_raw is None or args.ignore_image else _ensure_pil_images(imgs_raw, base_dir=getattr(args, "data_dir", None))
    images = row.get("images", None)
    if images is None or args.ignore_image:
        images_input = {}
    else:
        # 如果已经是 PIL，就直接用；否则再做一次统一转换
        if len(images) > 0 and isinstance(images[0], Image.Image):
            images = [im.convert("RGB") for im in images]
        else:
            images = _ensure_pil_images(images, base_dir=getattr(args, "data_dir", None))
        images_input = {"multi_modal_data": {"image": images}}
    num_images = len(images)

    # user 文本：先占位符，再题面与规则提示（要求 <think> 与 <answer>）
    if num_images > 0:
        img_header = "\n".join(["<image>" for _ in range(num_images)])
        user_text = (
            f"{img_header}\n"
            f"Question: {question}\n"
            f"Options:\n{opts_text}\n\n"
            "Think step by step inside <think>...</think>, then give ONE capital letter inside <answer>...</answer>."
        )
    else:
        user_text = (
            f"Question: {question}\n"
            f"Options:\n{opts_text}\n\n"
            "Think step by step inside <think>...</think>, then give ONE capital letter inside <answer>...</answer>."
        )

    # chat messages（system 不放图片）
    messages = [
        {"role": "system", "content": R1_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    # 渲染成字符串 prompt（vLLM 的 offline generate 只吃字符串）
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if num_images > 0 and not args.ignore_image:
        return {"prompt": prompt_text, "multi_modal_data": {"image": images}}
    else:
        return {"prompt": prompt_text}


@click.command()
@click.option(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-VL-3B-Instruct",
    # default="Qwen/Qwen2.5-0.5B-Instruct",
    help="Model name or path",
    show_default=True,
)
@click.option(
    "--dp_size", type=int, default=1, help="Data parallel size", show_default=True
)
@click.option(
    "--tp_size", type=int, default=1, help="Tensor parallel size", show_default=True
)
@click.option(
    "--node_size", type=int, default=1, help="Total number of nodes", show_default=True
)
@click.option(
    "--node_rank",
    type=int,
    default=0,
    help="Rank of the current node",
    show_default=True,
)
@click.option(
    "--master_addr",
    type=str,
    default="",
    help="Master node IP address",
    show_default=True,
)
@click.option(
    "--master_port", type=int, default=0, help="Master node port", show_default=True
)
@click.option("--enforce_eager", is_flag=True, help="Enforce eager mode execution.")
@click.option("--trust_remote_code", is_flag=True, help="Trust remote code.")
@click.option("--max_model_len", type=int, default=None, help="Max model length.")
@click.option(
    "--gpu_memory_utilization",
    type=float,
    default=0.9,
    help="GPU memory utilization fraction.",
)
@click.option("--dtype", type=str, default="bfloat16", help="Model dtype.")
@click.option("--max_images", type=int, default=6, help="Max images per prompt.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
# sampling
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature",
    show_default=True,
)
@click.option(
    "--top_p", type=float, default=1.0, help="Top-p sampling", show_default=True
)
@click.option(
    "--max_tokens",
    type=int,
    default=4096,
    help="Max tokens to generate",
    show_default=True,
)
@click.option(
    "--n", type=int, default=1, help="Number of samples to generate", show_default=True
)
# chat template
@click.option("--model_prompt_type", type=str, default=None)
@click.option("--instruction_prompt", type=str, default=None)
# dataset
@click.option("--dataset_name", default="UCSC-VLAA/MedVLThinker-Eval")
@click.option("--subset", default=None)
@click.option("--split", default="test")
@click.option(
    "--num_proc", type=int, default=16, help="Number of processes for dataset loading."
)
@click.option("--dataset_size", type=int, default=None, help="Debug subset size.")
# inference
@click.option("--batch_size", default=256, type=int)
# output
@click.option("--output_dir", default="outputs/default_eval/", type=str)
@click.option("--overwrite", is_flag=True, help="Overwrite output directory.")
# debug
@click.option("--debug", is_flag=True)
# misc
@click.option("--ignore_image", is_flag=True, help="Ignore image inputs.")
def multiprocess(**kwargs):
    args = types.SimpleNamespace(**kwargs)

    output_dir = Path(args.output_dir)
    print(f"Output directory: {output_dir}, checking...")
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            print(f"Output directory '{output_dir}' already exists. Overwriting.")
            shutil.rmtree(output_dir)
        else:
            print(f"try to resume from existing output directory '{output_dir}'.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # save args
    args_file = output_dir / "args.json"
    with open(args_file, "w", encoding="utf-8") as f:
        json.dump(
            vars(args),
            f,
            indent=2,
            ensure_ascii=False,
        )

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port
        print(
            f"Although set those variables, we do not use them. Using master address: {dp_master_ip}, port: {dp_master_port}"
        )

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    if args.debug is True:
        print("In debug mode")
        main(
            dp_size=1,
            local_dp_rank=0,
            global_dp_rank=0,
            dp_master_ip=dp_master_ip,
            dp_master_port=dp_master_port,
            tp_size=1,
            args=args,
            barrier=None,
        )
        exit()

    from multiprocessing import Barrier, Process

    procs = []
    num_process = len(range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node))
    barrier = Barrier(num_process)
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            kwargs=dict(
                dp_size=dp_size,
                local_dp_rank=local_dp_rank,
                global_dp_rank=global_dp_rank,
                dp_master_ip=dp_master_ip,
                dp_master_port=dp_master_port,
                tp_size=tp_size,
                args=args,
                barrier=barrier,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        # proc.join(timeout=300)
        proc.join()
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode
    if exit_code == 0:
        merge_output(args.output_dir)

    exit(exit_code)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    multiprocess()