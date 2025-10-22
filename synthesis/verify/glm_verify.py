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
# Prompt (verification rubric) & builders
# -----------------------------

VERIFICATION_PROMPT_TEMPLATE = r"""                                                                                                                         [SYSTEM ROLE]                                                                                                                                               You are a biomedical MCQ verifier with two distinct roles, executed in order:                                                                               1.  **The Referee (for Essential Criteria):** You are an objective, rule-based official. Your job is to fairly determine if the absolute minimum standards for usability are met.
2.  **The Critic (for Bonus Criteria):** After the essential check, you become a relentless perfectionist. Your default assumption is that the MCQ is NOT excellent. Your goal is to deny bonus points unless perfection is demonstrated beyond any doubt.

Judge each MCQ *only* using FIGURE(s)+CAPTION+CONTEXT.

[DATASET POLICY — MUST ENFORCE]
- Stem is self-contained and MUST NOT say "caption" or "context".
- Paraphrasing allowed, but DO NOT introduce unsupported clinical facts (age/sex/history/location/findings/diagnoses) beyond CAPTION/CONTEXT or clearly visible image cues.
- If CAPTION names a diagnosis, do NOT restate that diagnosis verbatim in the stem.
- Exactly one correct option; all options must be the same semantic type.
- Clinical correctness (modality/stain/anatomy/pathophysiology) must be supported by sources.

[SCORING MODEL]
Output ONLY {"rubric":[...]}.
Each item has:
- idx (1..N), title, description (ONE sentence starting with "Essential/Important/Optional/Pitfall Criteria: ..."),
- category: Essential | Important | Optional | Pitfall,
- weight: Essential=5; Important∈{3,4}; Optional∈{1,2}; Pitfall∈{-1,-2},
- score: Essential/Important/Optional ∈ {0, weight}; Pitfall ∈ {0, weight} (0 = not triggered),
- notes: ≤ 12 words (brief reason; no chain-of-thought).

**Scoring Mindset:**
- **Essential Items:** Award full points if the rule is met. Be a fair and impartial referee.
- **Important/Optional Items (Bonus Points):** **These are bonus points, not entitlements.** The score is **0 by default.** You must find **irrefutable evidence of perfection** to award points. If there is *any* subjective room for improvement, the score remains 0.

[FIXED ESSENTIAL ITEMS — MUST INCLUDE with EXACT titles]
1) Stem Self-contained
2) Vocabulary Constraint
3) Diagnosis Leak
4) Single Correct Option
5) Option Type Consistency
6) Clinical Validity
7) Image–Text Consistency

[BONUS CRITERIA FOR EXCELLENCE — ZERO-TOLERANCE JUDGEMENT]
(You must assess against a diverse set of 4–8 items. For this section, your mindset is "guilty until proven innocent." **A single, minor flaw in any sub-point means an instant score of 0 for the entire item.**)
- Plausible Distractors (Important, 3–4)
  • **Every single** distractor must be a strong, clinically relevant alternative given the sources.
  • Each must differ from the key by exactly ONE clear axis.
  • **If you can imagine a slightly more plausible distractor that wasn't used, score 0.** There must be no weak links.
- Parallel Options (Important, 3)
  • Grammatical structure, length, and specificity must be **rigorously uniform**.
  • **Any noticeable outlier** in form (e.g., one starts with a verb, others with nouns) or length fails this. No unique cues on the key.
- Stem Concision (Optional, 1–2)
  • Stem must be ≤ 2 sentences AND ≤ ~35 words.
  • If the stem can be rephrased to be even **slightly more elegant or direct** without losing critical meaning, **score 0**.
- Clarity and Focus (Optional, 2)
  • The stem poses a single, perfectly unambiguous question.
  • If the question could be worded **any more clearly or is even slightly awkward**, **score 0**.
- Answer Field Validity (Important, 3)
  • 'answer' exists, is one of A–E, and exactly matches an option key.
  • No duplicate options; all option strings non-empty.
- JSON Schema Compliance (Important, 3)
  • MCQ has exactly required keys {question, options{A..E}, answer}; no extras/missing.
- Forbidden Terms (Pitfall, –2)
  • Stem contains the exact word 'caption' or 'context'.
- Synonym Drift (Pitfall, –1)
  • Stem introduces a **specific** clinical fact that is absent from sources and not visible in the image.
- Multiple Keys (Pitfall, –2)
  • >1 option is reasonably correct **given sources**.
- Medical Inaccuracy (Pitfall, –2)
  • Any statement directly contradicts sources.

[HOW TO JUDGE]
- For Essentials, be a referee. For the Bonus section, be a rival looking for a weakness.
- Use only FIGURE(s)+CAPTION+CONTEXT. If an MCQ asserts anything unsupported, fail the relevant item.

[OUTPUT FORMAT — STRICT JSON ONLY]
Return exactly:
{
  "rubric": [
    {"idx":1,"title":"...","description":"Essential Criteria: ...","category":"Essential","weight":5,"score":0|5,"notes":"..."},
    ...
  ]
}
No totals. No extra keys. No commentary.

[INPUTS]
IMAGES: <attach in order>
MCQ:
{MCQ_JSON}

CAPTION:
\"\"\"{CAPTION_BLOCK}\"\"\"

CONTEXT:
\"\"\"{CONTEXT_BLOCK}\"\"\"

[CONDUCT]
If inputs are insufficient to judge, output {"error":"insufficient_evidence"} ONLY.
Think silently. Output ONLY the JSON object above.
""".strip()

def join_captions(captions: Optional[List[str]]) -> str:
    if not captions:
        return "Image(s) Public Caption: N/A"
    first = captions[0] if isinstance(captions, list) else captions
    text = (first or "").strip()
    if not text:
        return "Image(s) Public Caption: N/A"
    return f"Image(s) Public Caption: {text}"


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


def build_prompt_from_row(row: Dict[str, Any]) -> Optional[str]:
    """把 generated_vqa + caption/context 拼到验证模板里；若缺 MCQ 则返回 None。"""
    mcq = row.get("generated_vqa")
    if not mcq:
        return None
    try:
        mcq_json = json.dumps(mcq, ensure_ascii=False)
    except Exception:
        return None

    cap_block = join_captions(row.get("caption", None))
    ctx_block = join_contexts(row.get("context", None))

    return (VERIFICATION_PROMPT_TEMPLATE
            .replace("{MCQ_JSON}", mcq_json)
            .replace("{CAPTION_BLOCK}", cap_block)
            .replace("{CONTEXT_BLOCK}", ctx_block))

# ========== GLM chat template helpers ==========
def _make_glm_image_placeholders(num_images: int, modality: str = "image") -> str:
    if modality == "image":
        token = "<|begin_of_image|><|image|><|end_of_image|>"
    else:
        token = "<|begin_of_video|><|video|><|end_of_video|>"
    return token * num_images


def _wrap_glm_chat(prompt_text: str, vision_placeholders: str) -> str:
    # 官方建议格式；system 可以简单，核心是 user 放完整验证指令与占位符
    return (
        "[gMASK]<sop><|system|>\nYou are a helpful assistant."
        "<|user|>\n"
        f"{vision_placeholders}{prompt_text}"
        "<|assistant|>assistant\n"
    )

def _collect_pil_images(row: Dict[str, Any], args) -> List[Image.Image]:
    """收集 PIL 图片，加入最短边保护：skip / pad / resize."""
    images_field = row.get("images", None)
    pil_images: List[Image.Image] = []
    if not images_field:
        return pil_images

    for it in images_field[:args.max_images_per_prompt]:
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
        if img is None:
            continue

        w, h = img.size
        if min(w, h) < args.min_image_side:
            if args.small_image_policy == "skip":
                print(f"[Warn] skip tiny image {w}x{h} (<{args.min_image_side})")
                continue
            elif args.small_image_policy == "pad":
                new_w, new_h = max(w, args.min_image_side), max(h, args.min_image_side)
                canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
                canvas.paste(img, (0, 0))
                img = canvas
            else:  # resize
                new_w, new_h = max(w, args.min_image_side), max(h, args.min_image_side)
                img = img.resize((new_w, new_h), Image.BICUBIC)

        pil_images.append(img)

    return pil_images

def build_messages_glm(row: Dict[str, Any],
                       prompt_text: str,
                       tokenizer,
                       args) -> Dict[str, Any]:
    """GLM-4.5V: 支持多图，占位符与图片数一致；默认关闭 thinking。"""
    pil_images = _collect_pil_images(row, args)

    # 先尝试 tokenizer.apply_chat_template（优先硬关 thinking）
    prompt = None
    if tokenizer is not None:
        try:
            # messages: 单轮对话；多图 -> 多个 image 块
            content = []
            if pil_images:
                content += [{"type": "image", "image": img} for img in pil_images]
            content += [{"type": "text", "text": prompt_text}]
            glmmessages = [{"role": "user", "content": content}]

            # 大多数 transformers 版本支持 enable_thinking；否则走 kwargs 兜底
            prompt = tokenizer.apply_chat_template(
                glmmessages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(args.enable_thinking),
            )
        except Exception:
            # 部分模板走 chat_template_kwargs
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
        # 回退：手写模板（思考模式默认关闭；模板中不带 <think>）
        placeholders = _make_glm_image_placeholders(len(pil_images), modality="image")
        prompt = _wrap_glm_chat(prompt_text, placeholders)

    # 安全检查：占位符个数 == 图片数（按 <|image|> 统计）
    if pil_images:
        ph_count = prompt.count("<|image|>")
        if ph_count != len(pil_images):
            print(f"[Warn] placeholder({ph_count}) != images({len(pil_images)})")

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
            limit_mm_per_prompt={"image": args.max_images_per_prompt},
            mm_processor_kwargs=mm_kwargs_glm,
        )

        # 批推理
        for start_idx in trange(0, len(ds), args.batch_size,
                                unit_scale=args.batch_size,
                                desc=f"[Global DP Rank {global_dp_rank}] Verification"):
            end_idx = min(len(ds), start_idx + args.batch_size)
            ds_chunk = ds.select(range(start_idx, end_idx))

            # 构造 prompts
            built = []
            for row in ds_chunk:
                prompt_text = build_prompt_from_row(row)
                if prompt_text is None:
                    built.append(None)
                else:
                    built.append(build_messages_glm(row, prompt_text, tokenizer, args))

            # 过滤掉 None（缺 MCQ 的）
            valid_pairs = [(row, msg) for row, msg in zip(ds_chunk, built) if msg is not None]
            skipped_rows = [row for row, msg in zip(ds_chunk, built) if msg is None]

            # 先把缺 MCQ 的样本写出去（标注 error）
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
                        record["rubric_verification"] = {"error": "missing_mcq"}
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

                    record["rubric_verification"] = parsed
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