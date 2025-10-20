#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import multiprocessing as mp
import os
import re
import shutil
import traceback
import types
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
from datasets import load_dataset
from PIL import Image
from tqdm import trange
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from transformers import AutoTokenizer


# -----------------------------
# Utils: parse JSON from model output
# -----------------------------
def parse_model_output(output_texts):
    """接受 list[str] 或 str；从模型输出里提取并解析首个 JSON 对象。"""
    if isinstance(output_texts, list):
        if not output_texts:
            return None
        raw = output_texts[0]
    elif isinstance(output_texts, str):
        raw = output_texts
    else:
        return None

    s = raw.strip()

    # 去掉 ```json / ``` 围栏
    if s.startswith("```"):
        s = s[3:]
        if s.startswith("json"):
            s = s[4:] if len(s) >= 4 and s[3] == '\n' else s.lstrip("json").lstrip()
        fence_end = s.find("```")
        if fence_end != -1:
            s = s[:fence_end].strip()

    # 取第一个 {...}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = (candidate
                       .replace("“", "\"").replace("”", "\"")
                       .replace("’", "'"))
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None
    return None


# -----------------------------
# Prompt (MCQ type classification) & builders
# -----------------------------
QUESTION_TYPE_PROMPT_TEMPLATE = r"""
[SYSTEM ROLE]
You are a strict biomedical MCQ taxonomist. Classify each question based ONLY on its STEM and OPTIONS (no images/captions/context will be provided).

[CATEGORIES — CHOOSE EXACTLY ONE]
1) Modality Recognition
2) Anatomy/Localization
3) Finding/Abnormality Identification
4) Disease Diagnosis
5) Lesion Grading
6) Next Step
7) Other Biological/Technical Attributes

[DECISION RULES]
- Finding vs Diagnosis: “What abnormality/sign is present?” → Finding; “Most likely diagnosis?” → Disease Diagnosis.
- Any request to identify image acquisition technique/sequence/stain/phase → Modality Recognition.
- Asking for structures/regions/locations → Anatomy/Localization.
- Staging, grading, severity, or scores → Lesion Grading.
- What to do next (test or treatment) → Next Step.
- If none of the above but still asks for biological/technical properties (e.g., receptor status, cell type) → Other Biological/Technical Attributes.

[OUTPUT FORMAT — STRICT JSON ONLY]
Return exactly:
{"question_type":"<ONE OF THE 7 CATEGORIES ABOVE>"}
No extra keys, no explanations.

[INPUTS]
STEM:
\"\"\"{STEM_TEXT}\"\"\"

OPTIONS:
{OPTIONS_BLOCK}

[CONDUCT]
Think silently. Use only the stem and options.
""".strip()

### NEW
def _as_str_list(opts) -> Optional[List[str]]:
    if opts is None:
        return None
    if isinstance(opts, list):
        return [str(x) for x in opts]
    if isinstance(opts, str):
        s = opts.strip()
        # 可能是 JSON 字符串或分号/换行分隔
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return [str(x) for x in j]
        except Exception:
            pass
        # 尝试按换行/分号/ | 分割
        parts = [p.strip() for p in re.split(r'[\n;\|]', s) if p.strip()]
        if parts:
            return parts
    return None

def extract_mcq_fields(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """优先从 generated_vqa 拿 {question, options}；否则从顶层 {question, options}。"""
    mcq = row.get("generated_vqa")
    q, opts = None, None

    if isinstance(mcq, dict):
        q = mcq.get("question") or mcq.get("stem") or mcq.get("prompt")
        opts = mcq.get("options") or mcq.get("choices")
    if q is None:
        q = row.get("question")
    if opts is None:
        opts = row.get("options") or row.get("choices")

    opts = _as_str_list(opts)
    if not q or not opts:
        return None
    return {"question": str(q), "options": opts}


### CHANGED
def build_prompt_from_row(row: Dict[str, Any]) -> Optional[str]:
    """构造分类 prompt（仅基于 stem+options）。缺少 MCQ 时返回 None。"""
    mcq = extract_mcq_fields(row)
    if not mcq:
        return None

    stem = mcq["question"].strip()
    # 把选项 A/B/C... 排版出来
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for i, opt in enumerate(mcq["options"]):
        prefix = f"{letters[i]}. " if i < len(letters) else f"{i+1}. "
        lines.append(prefix + str(opt).strip())
    options_block = "\n".join(lines) if lines else "(no options)"

    return (QUESTION_TYPE_PROMPT_TEMPLATE
            .replace("{STEM_TEXT}", stem)
            .replace("{OPTIONS_BLOCK}", options_block))



### NEW
def build_messages_glm_text(prompt_text: str, tokenizer, args) -> Dict[str, Any]:
    """GLM text-only chat message."""
    glmmessages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = None
    if tokenizer is not None:
        try:
            prompt = tokenizer.apply_chat_template(
                glmmessages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(args.enable_thinking),
            )
        except Exception:
            try:
                prompt = tokenizer.apply_chat_template(
                    glmmessages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": bool(args.enable_thinking)},
                )
            except Exception:
                prompt = None
    if prompt is None:
        # 极端兜底：简单拼接
        prompt = "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n" + prompt_text + "<|assistant|>assistant\n"
    return {"prompt": prompt}



# -----------------------------
# Core multiprocess runner
# -----------------------------
def _worker_entrypoint(
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
    try:
        # --- GPU selection (respect external mapping) ---
        orig_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if orig_visible:
            all_ids = [int(x) for x in orig_visible.split(",") if x.strip() != ""]
        else:
            import torch
            all_ids = list(range(torch.cuda.device_count()))

        start = local_dp_rank * tp_size
        end = (local_dp_rank + 1) * tp_size
        sel = all_ids[start:end]
        assert len(sel) == tp_size, f"Not enough GPUs: need {tp_size}, got {sel} from {all_ids}"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, sel))
        print(f"Rank [{global_dp_rank}]: Using physical GPUs {sel} (logical ids 0..{len(sel)-1})")

        # ---------- Load dataset ----------
        if args.data_files:
            files = [s.strip() for s in args.data_files.split(",") if s.strip()]
            def infer_builder(path: str) -> str:
                ext = Path(path).suffix.lower()
                if ext in [".json", ".jsonl"]:
                    return "json"
                if ext in [".parquet", ".pq"]:
                    return "parquet"
                return "json"
            builder = infer_builder(files[0])
            ds_all = load_dataset(builder, data_files=files, split="train")
        elif args.dataset_name:
            ds_all = load_dataset(args.dataset_name, args.subset)[args.split]
        else:
            raise ValueError("Please provide either --data_files or --dataset_name")

        if args.dataset_size:
            ds_all = ds_all.select(range(args.dataset_size))

        if "dataset_index" not in ds_all.column_names:
            ds_all = ds_all.add_column("dataset_index", list(range(len(ds_all))))

        prompts_per_rank = ceil(len(ds_all) / dp_size)
        start = global_dp_rank * prompts_per_rank
        end = min(start + prompts_per_rank, len(ds_all))
        ds = ds_all.select(range(start, end))

        # 输出目录/恢复
        output_dir = Path(args.output_dir) / "shards"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"dp_{global_dp_rank}.jsonl"

        if out_file.exists() and not args.overwrite:
            done_idx = set()
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if "dataset_index" in rec:
                            done_idx.add(rec["dataset_index"])
            before = len(ds)
            ds = ds.filter(lambda row: row["dataset_index"] not in done_idx,
                           num_proc=args.num_proc, keep_in_memory=True)
            after = len(ds)
            print(f"Rank [{global_dp_rank}]: resume filter {before} -> {after}")

        if len(ds) == 0:
            print(f"Rank [{global_dp_rank}]: no data, exit.")
            if barrier is not None:
                barrier.wait()
            return

        # --- tokenizer for GLM (only once per worker) ---
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # vLLM 初始化
        sampling_params = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        # GLM 的多模态预处理窗口（官方示例给出）
        mm_kwargs_glm = {
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        }

        llm = LLM(
            model=args.model,
            tensor_parallel_size=tp_size,
            enforce_eager=args.enforce_eager,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            seed=args.seed,
        )

        # 批推理
        for start_idx in trange(0, len(ds), args.batch_size,
                                unit_scale=args.batch_size,
                                desc=f"[Global DP Rank {global_dp_rank}] Type Classify"):
            end_idx = min(len(ds), start_idx + args.batch_size)
            ds_chunk = ds.select(range(start_idx, end_idx))

            # 构造 prompts（文本-only）
            built = []
            for row in ds_chunk:
                prompt_text = build_prompt_from_row(row)
                if prompt_text is None:
                    built.append(None)
                else:
                    built.append(build_messages_glm_text(prompt_text, tokenizer, args))

            # 缺 MCQ 的样本：直接写 Question Type = "Unknown"
            valid_pairs = [(row, msg) for row, msg in zip(ds_chunk, built) if msg is not None]
            skipped_rows = [row for row, msg in zip(ds_chunk, built) if msg is None]

            if skipped_rows:
                with open(out_file, "a", encoding="utf-8") as f:
                    for row in skipped_rows:
                        record = dict(row)
                        try:
                            imgs = record.get("images", None)
                            new_imgs = []
                            if imgs:
                                for it in imgs:
                                    if isinstance(it, dict):
                                        p = it.get("path", None)
                                        if p:
                                            new_imgs.append({"path": p})
                                    else:
                                        new_imgs.append({"path": str(it)})
                            record["images"] = new_imgs
                        except Exception:
                            record["images"] = []

                        record["Question Type"] = "Unknown"
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if not valid_pairs:
                continue

            rows_valid, messages = zip(*valid_pairs)
            outputs = llm.generate(list(messages), sampling_params=sampling_params)

            # 写出
            with open(out_file, "a", encoding="utf-8") as f:
                for row, out in zip(rows_valid, outputs):
                    texts = [o.text.strip() for o in out.outputs] if out.outputs else [""]
                    parsed = parse_model_output(texts)

                    record = dict(row)
                    try:
                        imgs = record.get("images", None)
                        new_imgs = []
                        if imgs:
                            for it in imgs:
                                if isinstance(it, dict):
                                    p = it.get("path", None)
                                    if p:
                                        new_imgs.append({"path": p})
                                else:
                                    new_imgs.append({"path": str(it)})
                        record["images"] = new_imgs
                    except Exception:
                        record["images"] = []

                    qtype = None
                    if isinstance(parsed, dict):
                        qtype = parsed.get("question_type")
                        if isinstance(qtype, str):
                            qtype = qtype.strip()

                    record["Question Type"] = qtype if qtype else "Unknown"
                    if args.keep_raw_output:
                        record["raw_model_output"] = texts[0] if texts else ""

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if barrier is not None:
            barrier.wait()

    except Exception as e:
        print(f"Rank [{global_dp_rank}] Exception: {e}")
        traceback.print_exc()
        if barrier is not None:
            try:
                barrier.wait()
            except Exception:
                pass
        raise


def merge_shards(shards_dir: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for p in sorted(shards_dir.glob("dp_*.jsonl")):
            with open(p, "r", encoding="utf-8") as r:
                shutil.copyfileobj(r, w)
    print(f"Merged -> {out_path} ({out_path.stat().st_size/1024/1024:.2f} MB)")


# -----------------------------
# CLI
# -----------------------------
@click.command()
# model / engine
@click.option("--model", type=str, default="zai-org/GLM-4.5V", show_default=True,
              help="GLM-4.5V（如需 FP8，可改为 zai-org/GLM-4.5V-FP8，需 vLLM≥0.10.2）")
@click.option("--enable_thinking", is_flag=True, default=False,
              help="开启 GLM Thinking 模式（默认关闭）。")
@click.option("--dp_size", type=int, default=1, show_default=True)
@click.option("--tp_size", type=int, default=1, show_default=True)
@click.option("--node_size", type=int, default=1, show_default=True)
@click.option("--node_rank", type=int, default=0, show_default=True)
@click.option("--master_addr", type=str, default="", show_default=True)
@click.option("--master_port", type=int, default=0, show_default=True)
@click.option("--enforce_eager", is_flag=True, help="Enforce eager mode.")
@click.option("--max_model_len", type=int, default=None)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--dtype", type=str, default="bfloat16")
@click.option("--seed", type=int, default=42)
# sampling
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--top_p", type=float, default=1.0, show_default=True)
@click.option("--max_tokens", type=int, default=1024, show_default=True)
@click.option("--n", type=int, default=1, show_default=True)
# data
@click.option("--dataset_name", default=None, help="HF dataset name (optional, alt to --data_files).")
@click.option("--subset", default=None)
@click.option("--split", default="train", show_default=True)
@click.option("--data_files", default=None,
              help="Comma-separated parquet/json/jsonl paths or glob (datasets.load_dataset auto-splits as 'train').")
@click.option("--num_proc", type=int, default=16, show_default=True)
@click.option("--dataset_size", type=int, default=None, help="Debug subset size.")
# inference
@click.option("--batch_size", default=16, type=int, show_default=True)
@click.option("--max_images_per_prompt", type=int, default=8, show_default=True,
              help="limit_mm_per_prompt['image']，须 ≥ 单样本最大图片数。")
@click.option("--min_image_side", type=int, default=28, show_default=True,
              help="最短边保护阈值（GLM 预处理要求最短边≥28）。")
@click.option("--small_image_policy", type=click.Choice(["skip", "pad", "resize"]),
              default="skip", show_default=True,
              help="遇到过小图：skip=跳过；pad=黑边补齐到阈值；resize=放大到阈值。")
# output
@click.option("--output_dir", default="outputs/med_vqa_verify/", type=str, show_default=True)
@click.option("--overwrite", is_flag=True, help="Overwrite shard outputs.")
@click.option("--merge_after", is_flag=True, help="Merge shards into results.jsonl after finishing.")
@click.option("--keep_raw_output", is_flag=True, help="Keep raw_model_output for debugging.")
# misc
@click.option("--ignore_image", is_flag=True, help="保留参数占位；GLM 支持无图，但本任务通常有图。")
@click.option("--debug_single", is_flag=True, help="Run single-process for quick debug.")
def main(**kwargs):
    args = types.SimpleNamespace(**kwargs)

    out_root = Path(args.output_dir)
    if out_root.exists() and any(out_root.iterdir()) and args.overwrite:
        print(f"Overwrite output_dir: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # DP master（未使用，只占位）
    if args.node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port
        print(f"(Unused) DP master set to {dp_master_ip}:{dp_master_port}")

    assert args.dp_size % args.node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = args.dp_size // args.node_size

    if args.debug_single:
        _worker_entrypoint(
            dp_size=1, local_dp_rank=0, global_dp_rank=0,
            dp_master_ip=dp_master_ip, dp_master_port=dp_master_port,
            tp_size=1, args=args, barrier=None,
        )
        if args.merge_after:
            merge_shards(out_root / "shards", out_root / "results.jsonl")
        return

    from multiprocessing import Barrier, Process
    ranks = list(range(args.node_rank * dp_per_node, (args.node_rank + 1) * dp_per_node))
    barrier = Barrier(len(ranks))
    procs = []
    for local_dp_rank, global_dp_rank in enumerate(ranks):
        p = mp.Process(
            target=_worker_entrypoint,
            kwargs=dict(
                dp_size=args.dp_size,
                local_dp_rank=local_dp_rank,
                global_dp_rank=global_dp_rank,
                dp_master_ip=dp_master_ip,
                dp_master_port=dp_master_port,
                tp_size=args.tp_size,
                args=args,
                barrier=barrier,
            ),
        )
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode:
            exit_code = p.exitcode

    if exit_code == 0 and args.merge_after:
        merge_shards(out_root / "shards", out_root / "results.jsonl")

    exit(exit_code)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
