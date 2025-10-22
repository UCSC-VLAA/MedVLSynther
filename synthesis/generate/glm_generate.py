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

# NEW: tokenizer用于在离线模式下关闭thinking
try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# -----------------------------
# Parsing utils
# -----------------------------
# --- NEW: 清洗 <think> / <answer> 标签 ---
_ANSWER_PAT = re.compile(r"(?is)<think>.*?</think>\s*<answer>(.*?)</answer>")
_ANSWER_ONLY_PAT = re.compile(r"(?is)<answer>(.*?)</answer>")

def strip_reasoning_answer_tags(text: str) -> tuple[str, bool]:
    """
    若存在 <think>...</think><answer>...</answer>，只返回 <answer> 内文本。
    若仅有 <answer>...</answer> 也返回其中内容。
    否则原样返回。
    返回: (clean_text, detected_flag)
    """
    if not text:
        return text, False
    m = _ANSWER_PAT.search(text)
    if m:
        return m.group(1).strip(), True
    m2 = _ANSWER_ONLY_PAT.search(text)
    if m2:
        return m2.group(1).strip(), True
    return text, False

def parse_model_output(output_texts):
    if isinstance(output_texts, list):
        if not output_texts:
            return None
        raw = output_texts[0]
    elif isinstance(output_texts, str):
        raw = output_texts
    else:
        return None

    s = raw.strip()

    if s.startswith("```"):
        s = s[3:]
        if s.startswith("json"):
            s = s[4:] if len(s) >= 4 and s[3] == '\n' else s.lstrip("json").lstrip()
        fence_end = s.find("```")
        if fence_end != -1:
            s = s[:fence_end].strip()

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
# Prompt (rubric) & message builder
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

[SOURCE MATERIAL]
CAPTIONS:
\"\"\"{CAPTION_BLOCK}\"\"\"

CONTEXTS:
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

# ========== GLM chat template helpers ==========
def _make_glm_image_placeholders(num_images: int, modality: str = "image") -> str:
    if modality == "image":
        token = "<|begin_of_image|><|image|><|end_of_image|>"
    else:
        token = "<|begin_of_video|><|video|><|end_of_video|>"
    return token * num_images


def _wrap_glm_chat(prompt_text: str, vision_placeholders: str) -> str:
    # 官方vLLM示例格式（非思考模式下同样可用）
    return (
        "[gMASK]<sop><|system|>\nYou are a helpful assistant."
        "<|user|>\n"
        f"{vision_placeholders}{prompt_text}"
        "<|assistant|>assistant\n"
    )


def build_prompt_from_row(row: Dict[str, Any]) -> str:
    captions = row.get("caption", None)
    contexts = row.get("context", None)
    cap_block = join_captions(captions)
    ctx_block = join_contexts(contexts)
    return RUBRIC_PROMPT_TEMPLATE.replace("{CAPTION_BLOCK}", cap_block).replace("{CONTEXT_BLOCK}", ctx_block)

def build_messages(row: Dict[str, Any], args, tokenizer=None) -> Dict[str, Any]:
    # 1) collect images
    images_field = row.get("images", None)
    pil_images = []
    if images_field and not args.ignore_image:
        for it in images_field:
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
                w, h = img.size
                if min(w, h) < args.min_image_side:
                    if args.small_image_policy == "skip":
                        print(f"[Warn] skip tiny image {w}x{h} (<{args.min_image_side})")
                        continue
                    elif args.small_image_policy == "pad":
                        from PIL import Image
                        new_w, new_h = max(w, args.min_image_side), max(h, args.min_image_side)
                        canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
                        canvas.paste(img, (0, 0))
                        img = canvas
                    else:  # resize
                        new_w, new_h = max(w, args.min_image_side), max(h, args.min_image_side)
                        img = img.resize((new_w, new_h), Image.BICUBIC)
                pil_images.append(img)
    prompt_core = build_prompt_from_row(row)
    # 决定占位风格
    style = args.placeholder_style
    if style == "auto":
        mn = (args.model or "").lower()
        if "glm-4.5" in mn or "glm_4.5" in mn or "glm-45" in mn:
            style = "glm45"
        else:
            style = "legacy"

    # ------- GLM 专用 -------
    if style == "glm45":
        if pil_images:
            placeholders = _make_glm_image_placeholders(len(pil_images), modality=args.glm_modality)
        else:
            placeholders = ""
        # 优先：如果可用tokenizer，则用chat template并硬关thinking
        prompt = None
        if tokenizer is not None:
            try:
                glmmessages = [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image", "image": img} for img in pil_images] +
                            [{"type": "text", "text": prompt_core}]
                        ) if pil_images else [{"type": "text", "text": prompt_core}]
                    }
                ]
                # 关闭thinking（若transformers版本支持）
                prompt = tokenizer.apply_chat_template(
                    glmmessages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(args.enable_thinking),  # False -> 关闭
                )
                # 有些预览版用 chat_template_kwargs 接口：
                if "{enable_thinking" in (getattr(tokenizer, "chat_template", "") or "") and args.enable_thinking is False and "enable_thinking=False" not in prompt:
                    # 部分模板不吃enable_thinking位置参数，这里再兜底一次
                    prompt = tokenizer.apply_chat_template(
                        glmmessages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False},
                    )
            except Exception:
                prompt = None  # 回退到手写模板

        if prompt is None:
            # 回退：手写不带thinking的GLM模板
            prompt = _wrap_glm_chat(prompt_core, placeholders)

        mm = {"multi_modal_data": {"image": pil_images}} if pil_images else {}
        return {"prompt": prompt, **mm}

    # ------- 其他/历史风格（保持兼容） -------
    if pil_images:
        prompt = "<image>\n" * len(pil_images) + prompt_core
        mm = {"multi_modal_data": {"image": pil_images}}
    else:
        prompt = prompt_core
        mm = {}

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
        # --- GPU mapping ---
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
                if ext in [".json", ".jsonl"]: return "json"
                if ext in [".parquet", ".pq"]: return "parquet"
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

        # vLLM 初始化
        sampling_params = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        # GLM推荐的多模态预处理窗口（官方示例给出的size/fps）
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
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
            limit_mm_per_prompt={"image": args.max_images_per_prompt},
            mm_processor_kwargs=mm_kwargs_glm if args.placeholder_style in ("glm45", "auto") else {
                "min_pixels": 28 * 28, "max_pixels": 1280 * 28 * 28, "fps": 1
            },
        )

        # tokenizer（用于apply_chat_template关闭thinking）
        tokenizer = None
        if _HAS_TRANSFORMERS and (args.placeholder_style in ("glm45", "auto")):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model, trust_remote_code=args.trust_remote_code
                )
            except Exception as _e:
                tokenizer = None
                print(f"[Warn] Failed to load tokenizer for {args.model}: {_e}")
        # 批推理
        for start_idx in trange(0, len(ds), args.batch_size,
                                unit_scale=args.batch_size,
                                desc=f"[Global DP Rank {global_dp_rank}] Inference"):
            end_idx = min(len(ds), start_idx + args.batch_size)
            ds_chunk = ds.select(range(start_idx, end_idx))

            # 构造 prompts/messages
            messages = [build_messages(row, args, tokenizer=tokenizer) for row in ds_chunk]
            # 关闭thinking：离线generate不支持extra_body，这里已在build_messages中通过chat template/模板关闭
            outputs = llm.generate(messages, sampling_params=sampling_params)

            # 写出
            with open(out_file, "a", encoding="utf-8") as f:
                for row, out in zip(ds_chunk, outputs):
                    texts = [o.text.strip() for o in out.outputs] if out.outputs else [""]
                    # NEW: 清洗 <think>/<answer>，并记录是否触发
                    clean_texts = []
                    detected_thinking = False
                    for t in texts:
                        ct, found = strip_reasoning_answer_tags(t)
                        clean_texts.append(ct)
                        detected_thinking = detected_thinking or found

                    if detected_thinking:
                        print(f"[Warn] Rank [{global_dp_rank}]: Detected <think>/<answer>; using <answer> content only for raw output & parsing.")

                    # 之后的JSON解析基于“清洗后”的文本
                    parsed = parse_model_output(clean_texts)

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

                    record["generated_vqa"] = parsed
                    if args.keep_raw_output:
                        record["raw_model_output"] = (clean_texts[0] if detected_thinking else (texts[0] if texts else ""))

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
@click.option("--min_image_side", type=int, default=28, show_default=True,
              help="小于该最短边像素的图片将按 small_image_policy 处理")
@click.option("--small_image_policy", type=click.Choice(["skip","pad","resize"]),
              default="skip", show_default=True,
              help="遇到过小图的策略：skip=跳过; pad=边缘填充到阈值; resize=直接放大到阈值")

@click.option("--model", type=str, default="zai-org/GLM-4.5V", show_default=True,
              help="GLM 模型名；若配合 --glm_precision=fp8 且模型不是 *-FP8，将自动替换为 FP8 版本。")
@click.option("--glm_precision", type=click.Choice(["bf16", "fp8"]), default="bf16", show_default=True,
              help="选择GLM精度：bf16使用 'zai-org/GLM-4.5V'；fp8使用 'zai-org/GLM-4.5V-FP8'。")
@click.option("--enable_thinking", is_flag=True, default=False,
              help="开启GLM Thinking Mode（默认关闭）。")
@click.option("--glm_modality", type=click.Choice(["image", "video"]), default="image", show_default=True)
@click.option("--dp_size", type=int, default=1, show_default=True)
@click.option("--tp_size", type=int, default=1, show_default=True)
@click.option("--node_size", type=int, default=1, show_default=True)
@click.option("--node_rank", type=int, default=0, show_default=True)
@click.option("--master_addr", type=str, default="", show_default=True)
@click.option("--master_port", type=int, default=0, show_default=True)
@click.option("--enforce_eager", is_flag=True, help="Enforce eager mode.")
@click.option("--trust_remote_code", is_flag=True)
@click.option("--max_model_len", type=int, default=None)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--dtype", type=str, default="bfloat16",
              help="计算精度（对FP8权重一般设置为'auto'或fp16/bf16均可；保持默认即可）。")
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
              help="Comma-separated parquet/json/jsonl paths or glob.")
@click.option("--num_proc", type=int, default=16, show_default=True)
@click.option("--dataset_size", type=int, default=None, help="Debug subset size.")
# inference
@click.option("--batch_size", default=16, type=int, show_default=True)
# output
@click.option("--output_dir", default="outputs/med_vqa_vllm/", type=str, show_default=True)
@click.option("--overwrite", is_flag=True, help="Overwrite shard outputs.")
@click.option("--merge_after", is_flag=True, help="Merge shards into results.jsonl after finishing.")
@click.option("--keep_raw_output", is_flag=True, help="Keep raw_model_output for debugging.")
# misc
@click.option("--ignore_image", is_flag=True, help="Ignore images (text-only).")
@click.option("--debug_single", is_flag=True, help="Run single-process for quick debug.")
@click.option("--placeholder_style", type=click.Choice(["glm45", "legacy", "auto"]),
              default="glm45", show_default=True,
              help="glm45: <|begin_of_image|><|image|><|end_of_image|>; legacy: <image>; auto: 根据模型名猜。")
@click.option("--max_images_per_prompt", type=int, default=8, show_default=True,
              help="传给 vLLM 的上限，须 ≥ 单样本图片数最大值。")

def main(**kwargs):
    args = types.SimpleNamespace(**kwargs)

    # 自动对齐模型与精度
    model_l = (args.model or "").lower()
    if "glm-4.5v" in model_l or "glm_4.5v" in model_l:
        if args.glm_precision == "fp8" and ("fp8" not in model_l):
            print("[Info] glm_precision=fp8 -> 使用 zai-org/GLM-4.5V-FP8")
            args.model = "zai-org/GLM-4.5V-FP8"
            if args.dtype == "bfloat16":
                args.dtype = "auto"  # FP8权重下一般使用auto更稳妥
        if args.glm_precision == "bf16" and ("fp8" in model_l):
            print("[Info] glm_precision=bf16，但模型是FP8量化版本；如需BF16请改为 zai-org/GLM-4.5V")

    out_root = Path(args.output_dir)
    if out_root.exists() and any(out_root.iterdir()) and args.overwrite:
        print(f"Overwrite output_dir: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

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