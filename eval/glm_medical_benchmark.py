import dotenv
dotenv.load_dotenv(override=True)

import json
import multiprocessing as mp
import os
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
from tqdm import trange
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


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
    gpu_ids = range(local_dp_rank * tp_size, (local_dp_rank + 1) * tp_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    print(f"Rank [{global_dp_rank}]: Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

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
    # This might fail if the model needs trust_remote_code=True, handle it gracefully if needed
    try:
        processor = AutoProcessor.from_pretrained(model, trust_remote_code=args.trust_remote_code)
    except Exception as e:
        print(f"Could not load processor for {model}. Error: {e}")
        processor = None

    # Check if dataset has at least one row before trying to build a prompt
    if len(ds) > 0:
        build_prompt(ds[0], processor, args)
    else:
        print(f"Rank [{global_dp_rank}]: Dataset is empty after initial load. Exiting.")
        if barrier:
            barrier.wait()
        return

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
        if barrier:
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
        # ↓↓↓ GLM 多模态设置：支持多图与高分辨率
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={"image": getattr(args, "max_images_per_prompt", 8)},
    )

    # Print the outputs.
    batch_size = args.batch_size
    for start_idx in trange(
        0,
        len(ds),
        batch_size,
        unit_scale=True,  # Set to True to show iterations/s
        desc=f"[Global DP Rank {global_dp_rank}] Processing dataset",
    ):
        end_idx = min(len(ds), start_idx + batch_size)

        ds_chunk = ds.select(range(start_idx, end_idx))

        # --- MODIFICATION: Filter out invalid prompts (e.g., due to small images) ---
        prompts = []
        valid_ds_chunk_rows = []
        for row in ds_chunk:
            prompt_data = build_prompt(row, processor, args)
            if prompt_data is not None:
                prompts.append(prompt_data)
                valid_ds_chunk_rows.append(row)

        # If the entire batch was skipped, continue to the next one
        if not prompts:
            continue
        # --- END MODIFICATION ---

        outputs = llm.generate(prompts, sampling_params=sampling_params)

        results = []

        # --- MODIFICATION: Zip with the filtered list of rows ---
        for idx, (row, output) in enumerate(zip(valid_ds_chunk_rows, outputs)):
            # --- END MODIFICATION ---
            # In each output, it consists of multiple rollouts,
            # by default it is 1.

            # metadata
            # dp_index needs to be based on original dataset index if possible,
            # but for simplicity, we keep it based on filtered chunk index.
            # A more robust solution might use row["dataset_index"].
            dp_index = start_idx + ds_chunk.to_dict()['dataset_index'].index(row['dataset_index'])
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
        # Use a more descriptive print statement
        if results:
            print(f"\nSaving {len(results)} results to '{out_file}'...")
            with open(out_file, "a", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

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
    print(f"Accuracy for dp_{global_dp_rank}: {result_acc}")
    with open(out_acc_file, "w", encoding="utf-8") as f:
        json.dump(result_acc, f, indent=2, ensure_ascii=False)
    print(f"Saved accuracy to '{out_acc_file}'.")

    # NOTE(xk) Wait for all processes to finish before exiting.
    # Otherwise, the main process (using pytorch dist) may exit before all processes finish writing.
    if barrier is not None:
        barrier.wait()


# def extract_answer(text: str) -> str:
#     """Extract the model’s final outputs."""
#     m = re.search(r"<answer>(.*?)</answer>", text, re.S)
#     return m.group(1).strip() if m else text.strip()

def extract_answer(text: str, options: dict | None = None) -> str:
    """
    优先返回字母(A-E)。若都解析不到，返回清洗后的全文作为兜底（兼容你原来的 grade 逻辑）。
    可选传入 options（形如 {'A':'...','B':'...'}）做文本匹配兜底。
    """
    if not isinstance(text, str):
        return ""

    t = text

    # 1) 去掉思维链
    t = re.sub(r"(?is)<think>.*?</think>", " ", t)

    # 2) 若有 box，优先抽取 box 内文
    m = re.search(r"(?is)<\|begin_of_box\|>(.*?)<\|end_of_box\|>", t)
    if m:
        t = m.group(1)

    # 3) 规范空白
    t_norm = " ".join(t.split())

    # 4) 万一真的有 <answer>…</answer>，先用这个
    m = re.search(r"(?is)<answer>\s*([A-E])(?:\s*[\.\)]\s*)?.*?</answer>", t_norm)
    if m:
        return m.group(1).upper()

    # 5) 常见英文/中文提示语
    patterns = [
        r"(?is)\banswer\s*(?:should\s*be|is|:|：)\s*([A-E])\b",
        r"(?is)\b(final|correct)\s+answer\s*(?:is|:|：)\s*([A-E])\b",
        r"(?is)\btherefore[,，]?\s*the\s*answer\s*(?:is|:|：)?\s*([A-E])\b",
        r"(?is)\boption\s*([A-E])\b",
    ]
    for pat in patterns:
        m = re.search(pat, t_norm)
        if m:
            return (m.groups()[-1]).upper()

    # 6) 在“answer/正确/最终”等关键词附近找形如 "B." 的模式
    m = re.search(r"(?is)(?:answer|final|correct)[^A-E]{0,50}\b([A-E])(?:\s*[\.\)])\b", t_norm)
    if m:
        return m.group(1).upper()

    # 7) 选项文本匹配兜底（可选；需传入 options）
    if options:
        low = t_norm.lower()
        best_key = None
        best_len = 0
        for k, v in (options.items() if hasattr(options, "items") else []):
            if not isinstance(v, str):
                continue
            vv = v.strip().lower()
            if vv and vv in low and len(vv) > best_len:
                best_key, best_len = k, len(vv)
        if best_key:
            return str(best_key).strip().upper()

    # 8) 最后手段：在结尾窗口里抓最后一个孤立的大写 A-E
    tail = t_norm[-200:]
    letters = re.findall(r"\b([A-E])\b", tail)
    if letters:
        return letters[-1].upper()

    # 9) 实在不行，返回全文（兼容你原 grade_answer 的兜底比较）
    return t_norm.strip()



def grade_answer(prediction, answer, answer_label=None):
    if answer_label is not None:
        if prediction.strip().lower() == f"{answer_label}. {answer}".strip().lower():
            return True
        elif prediction.strip().lower() == answer_label.strip().lower():
            return True

    if prediction.strip().lower() == answer.strip().lower():
        return True

    return False


def _glm_image_placeholders(n: int, modality: str = "image") -> str:
    if modality == "image":
        tok = "<|begin_of_image|><|image|><|end_of_image|>"
    else:
        tok = "<|begin_of_video|><|video|><|end_of_video|>"
    return tok * max(0, int(n))


def _glm_wrap_chat(prompt_text: str, vision_placeholders: str) -> str:
    # 手写兜底模板；GLM 官方建议 system+user+assistant
    return (
        "[gMASK]<sop><|system|>\nYou are a helpful assistant."
        "<|user|>\n"
        f"{vision_placeholders}{prompt_text}"
        "<|assistant|>assistant\n"
    )


def build_prompt(row, processor, args):
    # 这里原来调用 build_messages(row, args)；改为把 processor 也传进去
    messages = build_messages(row, processor, args)
    if messages and getattr(args, "debug", False):
        print(f"Prompt: {messages['prompt']}...")
    return messages


INSTRUCTION_PROMPT = (
    "You will solve a problem/request. You should provide your thoughts within <think> </think> tags before "
    "providing the answer.\nWrite your final answer within <|begin_of_box|> <|end_of_box|> tags with only the option letter."
)


def build_messages(row, processor, args):
    # 最小尺寸；GLM 对分辨率更友好，但这里做个兜底过滤，避免无效小图
    MIN_IMAGE_DIMENSION = 28

    # 文本部分
    question = row["question"]
    raw_options = row["options"]
    options = json.loads(raw_options)

    text_prompt = f"Question: {question}\n\nOptions:"
    for letter, option in options.items():
        text_prompt += f"\n\n{letter}. {option}"

    instrution_prompt = getattr(args, "instruction_prompt", None) or INSTRUCTION_PROMPT
    text_prompt = instrution_prompt + "\n\n" + text_prompt

    # 图像收集（数据集中一般已是 PIL.Image）
    images = row.get("images", None) or []
    images_input = {}

    if images and not args.ignore_image:
        # 尺寸兜底检查（你也可以把阈值改成 getattr(args,"min_image_side", 128) 更严格）
        for im in images:
            if im.width < MIN_IMAGE_DIMENSION or im.height < MIN_IMAGE_DIMENSION:
                print(
                    f"\n[WARNING] Skipping dataset_index {row.get('dataset_index', 'N/A')} "
                    f"due to small image (size: {im.size}). "
                    f"Minimum dimension required: {MIN_IMAGE_DIMENSION}px."
                )
                return None
        # 统一转 RGB
        images = [im.convert("RGB") for im in images]
        images_input = {"multi_modal_data": {"image": images}}
    else:
        images = []

    # 根据模型名决定提示词格式
    final_prompt = ""
    model_name_lower = args.model.lower()
    model_prompt_type = getattr(args, "model_prompt_type", None)

    # === GLM-4.5V 分支：支持多图 + think mode ===
    if "glm" in model_name_lower:
        prompt_built = None
        # 优先用 processor.apply_chat_template 来打开 think mode
        if processor is not None and hasattr(processor, "apply_chat_template"):
            try:
                content = []
                if images:
                    content += [{"type": "image", "image": im} for im in images]
                content += [{"type": "text", "text": text_prompt}]
                glmmessages = [{"role": "user", "content": content}]

                # 默认开启 think mode；若你之后想关，可给 args.enable_thinking=False
                enable_thinking = True if getattr(args, "enable_thinking", None) is None else bool(args.enable_thinking)

                prompt_built = processor.apply_chat_template(
                    glmmessages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except Exception:
                try:
                    prompt_built = processor.apply_chat_template(
                        glmmessages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": True},
                    )
                except Exception:
                    prompt_built = None

        # 如果上面失败，走手写兜底模板（会插入与图片数一致的占位符）
        if prompt_built is None:
            placeholders = _glm_image_placeholders(len(images), modality="image")
            prompt_built = _glm_wrap_chat(text_prompt, placeholders)

        # 安全检查：占位符个数是否与图片数一致（按 <|image|> 计数）
        if images:
            ph_count = prompt_built.count("<|image|>")
            if ph_count != len(images):
                print(f"[Warn] GLM placeholder({ph_count}) != images({len(images)})")

        final_prompt = prompt_built

    # === 保留原有 Qwen 分支（以防切回） ===
    elif "qwen" in model_name_lower:
        if images:
            vision_placeholders = "<|image_pad|>" * len(images)
            final_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{vision_placeholders}<|vision_end|>"
                f"{text_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            final_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{text_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

    # === 其他模型保持原样 ===
    elif model_prompt_type == "gemma3":
        image_placeholders = "<image>\n" * len(images) if images else ""
        final_prompt = (
            "<bos><start_of_turn>user\n"
            + f"{image_placeholders}{text_prompt}<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )
    else:
        image_placeholders = "<image>\n" * len(images) if images else ""
        final_prompt = image_placeholders + text_prompt

    return {
        "prompt": final_prompt,
        **images_input,
    }


@click.command()
@click.option(
    "--model",
    type=str,
    default="zai-org/GLM-4.5V",  # 默认改为 GLM-4.5V
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
            print(f"Killing process {proc.pid} that didn't stop within timeout.")
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
