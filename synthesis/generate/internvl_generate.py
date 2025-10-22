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
# Parsing utils
# -----------------------------
def parse_model_output(output_texts):
    """
    接受 list[str] 或 str；从模型输出里提取并解析首个 JSON 对象。
    允许 fenced code block（```json 或 ```）或纯文本 JSON。
    """
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
# Prompt (rubric) & message builder (用于“生成MCQ”)
# -----------------------------
RUBRIC_PROMPT_TEMPLATE = r"""
[SYSTEM ROLE]
You are an expert medical-education item writer. Your job is to generate a high-quality multiple-choice question (MCQ) from a biomedical figure (the image) and its accompanying paper caption/context. The MCQ must be self-contained, clinically valid, and solvable by carefully inspecting the image together with general domain knowledge implicitly derivable from the caption/context—WITHOUT quoting the caption or revealing the answer verbatim.

[NON-NEGOTIABLE RULES]
1) Do NOT write “according to the caption/description/text” or similar. The stem must read as a standalone exam question.
2) Do NOT copy answer text verbatim from the caption; paraphrase and compress.
3) Exactly ONE best answer. Distractors must be plausible and mutually exclusive with the key.
4) Use only information supported by the image and facts that a competent clinician could infer from the caption/context; no speculative claims.
5) Keep clinical terminology precise; avoid brand names/PHI; no patient identifiers.
6) Output MUST follow the JSON schema below—no extra keys, no commentary.
7) Do NOT include chain-of-thought. Keep rationales concise and factual is NOT required—only output the JSON.

[RUBRIC (Self-check before you output)]
Essential Criteria (must PASS)
- E1. Stem is self-contained; no mention of “caption/description”; no hidden assumptions.
- E2. Image–content alignment: the question requires inspecting specific visual features.
- E3. Caption-derived facts are integrated implicitly (paraphrased) but do not leak the answer.
- E4. Single correct option; remaining options are incorrect for a clear clinical reason.
- E5. Medical correctness: terminology, anatomy, modality, and pathophysiology are accurate.

Important Criteria (strongly recommended)
- I1. Cognitive level: ≥ application (identification, interpretation, next step, best explanation).
- I2. Distractors: near-misses, common confusions, or plausible alternatives—not trivial.
- I3. Parallelism: options have similar length/structure; avoid “all/none of the above.”
- I4. Difficulty labeled (Easy/Moderate/Hard) with a brief justification.  (Do NOT include this in the output JSON.)
- I5. Balanced scope: focuses on one primary concept (finding, diagnosis, step, location).

Optional Criteria (nice to have)
- O1. Localizes the key finding (e.g., lobe/segment/organ subregion) if appropriate.
- O2. Uses quantitative details (size/scale/grade/stage) only when clearly supported.

[ALLOWED QUESTION ARCHETYPES]
Pick ONE that best fits the image + derivable facts:
- Finding identification (“Which abnormality is present?”)
- Best diagnosis / most likely explanation
- Next best step (diagnostic or management)
- Localization (“Which structure/region is affected?”)
- Modality/sequence recognition (e.g., T1 vs T2, phase, stain)

[GENERATION WORKFLOW]
Step 1 — (Privately) derive a few concise, paraphrased facts from captions/contexts that a clinician could reasonably infer.
Step 2 — Choose an archetype so that both image inspection and those facts are needed.
Step 3 — Write a self-contained stem that integrates the derived facts implicitly (no mention of “caption/description/text”) and does NOT reveal the answer.
Step 4 — Author 5 options (A–E): one correct, four high-quality distractors (near-miss/opposite/irrelevant-but-plausible). Keep options parallel.
Step 5 — Run the RUBRIC. If any Essential item fails, regenerate (internally). Output only JSON.

[OUTPUT FORMAT — STRICT JSON]
Return exactly one JSON object with keys:
- "question": string
- "options": object with keys "A","B","C","D","E"
- "answer": one of "A","B","C","D","E"

[SOURCE MATERIAL] (allowed vocabulary source; do NOT quote literally if it would leak the diagnosis)
CAPTIONS (one per image, in order):
\"\"\"{CAPTION_BLOCK}\"\"\"

CONTEXTS (one per image, in order):
\"\"\"{CONTEXT_BLOCK}\"\"\"

Think silently. Output ONLY the JSON object.
""".strip()

def join_captions(captions: Optional[List[str]]) -> str:
    if not captions:
        return "N/A"
    lines = []
    for i, cap in enumerate(captions, start=1):
        cap = (cap or "").strip()
        lines.append(f"[Image {i}] {cap}" if cap else f"[Image {i}] (empty)")
    return "\n".join(lines)


def join_contexts(contexts: Optional[List[List[str]]]) -> str:
    if not contexts:
        return "No additional context is provided."
    lines = []
    for i, ctx_list in enumerate(contexts, start=1):
        if ctx_list:
            ctx_text = " ".join([s.strip() for s in ctx_list if s and s.strip()])
            ctx_text = ctx_text if ctx_text else "(empty)"
        else:
            ctx_text = "(empty)"
        lines.append(f"[Image {i}] {ctx_text}")
    return "\n".join(lines)


def build_prompt_from_row(row: Dict[str, Any]) -> str:
    """
    用于“生成 MCQ”的主提示（rubric 模板 + captions/contexts）。
    """
    captions = row.get("caption", None)
    contexts = row.get("context", None)
    cap_block = join_captions(captions)
    ctx_block = join_contexts(contexts)
    prompt = RUBRIC_PROMPT_TEMPLATE.replace("{CAPTION_BLOCK}", cap_block).replace("{CONTEXT_BLOCK}", ctx_block)
    return prompt

# -----------------------------
# InternVL: message builder（<image>×N + chat template）
# -----------------------------
def _collect_pil_images(row: Dict[str, Any], max_images: int) -> List[Image.Image]:
    images_field = row.get("images", None)
    pil_images: List[Image.Image] = []
    if not images_field:
        return pil_images
    for it in images_field[:max_images]:
        img = None
        if isinstance(it, Image.Image):
            img = it.convert("RGB")
        elif isinstance(it, dict):
            b = it.get("bytes"); p = it.get("path")
            if b is not None:
                try:
                    img = Image.open(BytesIO(b)).convert("RGB")
                except Exception:
                    img = None
            if img is None and p:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    img = None
        else:
            try:
                img = Image.open(str(it)).convert("RGB")
            except Exception:
                img = None
        if img is not None:
            pil_images.append(img)
    return pil_images

def build_messages_internvl(row: Dict[str, Any],
                            prompt_text: str,
                            tokenizer,
                            args) -> Dict[str, Any]:
    """
    InternVL: 用 <image> 占位符 + chat template；确保 prompt 是 str。
    支持多图：占位符重复 N 次，并在 multi_modal_data 传入同样数量的 PIL 图片。
    """
    pil_images: List[Image.Image] = [] if args.ignore_image else _collect_pil_images(row, args.max_images_per_prompt)
    vision_placeholders = ("<image>\n" * len(pil_images)) if pil_images else ""
    user_text = f"{vision_placeholders}{prompt_text}"

    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    mm = {"multi_modal_data": {"image": pil_images}} if pil_images else {}
    return {"prompt": prompt, **mm}

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
                           num_proc=args.num_proc,
                           keep_in_memory=True)
            after = len(ds)
            print(f"Rank [{global_dp_rank}]: resume filter {before} -> {after}")

        if len(ds) == 0:
            print(f"Rank [{global_dp_rank}]: no data, exit.")
            if barrier is not None:
                barrier.wait()
            return
        # --- tokenizer for InternVL (only once per worker) ---
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # InternVL 停止符（不同变体略有差异；这里采用通用集合，None 会被过滤）
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        # vLLM 采样参数
        sampling_params = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop_token_ids=stop_token_ids if stop_token_ids else None,
        )

        # vLLM 初始化（InternVL 建议 trust_remote_code=True）
        llm = LLM(
            model=args.model,
            tensor_parallel_size=tp_size,
            enforce_eager=args.enforce_eager,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            seed=args.seed,
            limit_mm_per_prompt={"image": args.max_images_per_prompt},
        )

        # 批推理
        for start_idx in trange(0, len(ds), args.batch_size,
                                unit_scale=args.batch_size,
                                desc=f"[Global DP Rank {global_dp_rank}] Inference"):
            end_idx = min(len(ds), start_idx + args.batch_size)
            ds_chunk = ds.select(range(start_idx, end_idx))

            # 先构造文本 prompt，再用 InternVL 的 chat template 封装
            msg_list = []
            for row in ds_chunk:
                prompt_core = build_prompt_from_row(row)
                msg = build_messages_internvl(row, prompt_core, tokenizer, args)
                msg_list.append(msg)

            outputs = llm.generate(msg_list, sampling_params=sampling_params)

            # 写出
            with open(out_file, "a", encoding="utf-8") as f:
                for row, out in zip(ds_chunk, outputs):
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

                    record["generated_vqa"] = parsed  # {"question":..., "options":{...}, "answer":"A"}
                    if args.keep_raw_output:
                        record["raw_model_output"] = texts[0] if texts else ""

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 等待所有进程
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
@click.option("--model", type=str, default="OpenGVLab/InternVL3-8B-Instruct", show_default=True,
              help="InternVL 模型，如 OpenGVLab/InternVL3-8B-Instruct 或 InternVL3.5 变体。")
@click.option("--dp_size", type=int, default=1, show_default=True)
@click.option("--tp_size", type=int, default=1, show_default=True)
@click.option("--node_size", type=int, default=1, show_default=True)
@click.option("--node_rank", type=int, default=0, show_default=True)
@click.option("--master_addr", type=str, default="", show_default=True)
@click.option("--master_port", type=int, default=0, show_default=True)
@click.option("--enforce_eager", is_flag=True, help="Enforce eager mode.")
@click.option("--trust_remote_code", is_flag=True, help="[已内置 True] InternVL 推荐开启（保持此开关仅为兼容旧参数）。")
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
              help="Comma-separated parquet/json/jsonl paths or glob; auto-selects datasets builder.")
@click.option("--num_proc", type=int, default=16, show_default=True)
@click.option("--dataset_size", type=int, default=None, help="Debug subset size.")
# inference
@click.option("--batch_size", default=16, type=int, show_default=True)
# output
@click.option("--output_dir", default="outputs/med_vqa_vllm_internvl/", type=str, show_default=True)
@click.option("--overwrite", is_flag=True, help="Overwrite shard outputs.")
@click.option("--merge_after", is_flag=True, help="Merge shards into results.jsonl after finishing.")
@click.option("--keep_raw_output", is_flag=True, help="Keep raw_model_output for debugging.")
# misc
@click.option("--ignore_image", is_flag=True, help="Ignore images (text-only).")
@click.option("--debug_single", is_flag=True, help="Run single-process for quick debug.")
@click.option("--max_images_per_prompt", type=int, default=8, show_default=True,
              help="传给 vLLM 的上限，须 ≥ 单样本图片数最大值。")

def main(**kwargs):
    args = types.SimpleNamespace(**kwargs)

    out_root = Path(args.output_dir)
    if out_root.exists() and any(out_root.iterdir()) and args.overwrite:
        print(f"Overwrite output_dir: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 保存参数
    with open(out_root / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # DP master
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
            dp_size=1,
            local_dp_rank=0,
            global_dp_rank=0,
            dp_master_ip=dp_master_ip,
            dp_master_port=dp_master_port,
            tp_size=1,
            args=args,
            barrier=None,
        )
        if args.merge_after:
            merge_shards(out_root / "shards", out_root / "results.jsonl")
        return

    from multiprocessing import Barrier, Process
    procs = []
    ranks = list(range(args.node_rank * dp_per_node, (args.node_rank + 1) * dp_per_node))
    barrier = Barrier(len(ranks))

    for local_dp_rank, global_dp_rank in enumerate(ranks):
        proc = Process(
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
        proc.start()
        procs.append(proc)

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