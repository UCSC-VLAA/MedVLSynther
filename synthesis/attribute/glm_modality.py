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
# Allowed label sets
# -----------------------------
MODALITY_LABELS = [
    "Colposcopy",
    "CT (Computed Tomography)",
    "Digital Photography",
    "Fundus Photography",
    "Infrared Reflectance Imaging",
    "MR (Magnetic Resonance Imaging)",
    "OCT (Optical Coherence Tomography)",
    "Dermoscopy",
    "Endoscopy",
    "Microscopy Images",
    "X-Ray",
    "Ultrasound",
]

ANATOMY_LABELS = [
    "Lung",
    "Mammary Gland",
    "Hand",
    "Upper Limb",
    "Eye",
    "Uterus",
    "Intestine",
    "Skin",
    "Shoulder",
    "Kidney",
    "Gallbladder",
    "Pancreas",
    "Spleen",
    "Liver",
    "Pelvic",
    "Ovary",
    "Blood Vessel",
    "Spine",
    "Urinary System",
    "Adipose Tissue",
    "Muscle Tissue",
    "Oral Cavity",
    "Knee",
    "Foot",
    "Lower Limb",
]

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower()) if isinstance(s, str) else ""

# simple canonicalization (case/spacing insensitive, exact by normalized string)
def _canonicalize(val: Optional[str], allowed: List[str]) -> Optional[str]:
    if not isinstance(val, str):
        return None
    nv = _norm(val)
    table = { _norm(a): a for a in allowed }
    return table.get(nv, None)

# -----------------------------
# Prompt for (modality, anatomy)
# -----------------------------
LABEL_PROMPT_TEMPLATE = r"""
[SYSTEM ROLE]
You are a strict biomedical MCQ visual taxonomist. Use the provided medical image(s) together with the STEM and OPTIONS to classify this question.

[TASK]
Choose exactly ONE label for each of:
- modality (from the allowed list)
- anatomy (from the allowed list)

[ALLOWED MODALITY LABELS]
{MODALITY_BLOCK}

[ALLOWED ANATOMY LABELS]
{ANATOMY_BLOCK}

[OUTPUT FORMAT — STRICT JSON ONLY]
Return exactly:
{{
  "modality": "<ONE MODALITY FROM THE LIST>",
  "anatomy": "<ONE ANATOMY FROM THE LIST>"
}}
No extra keys. No explanations. No code fences.

[INPUTS]
STEM:
\"\"\"{STEM_TEXT}\"\"\"

OPTIONS:
{OPTIONS_BLOCK}

[CONDUCT]
Think silently. Output ONLY the JSON object above.
""".strip()

def _format_allowed_block(items: List[str]) -> str:
    return "\n".join(f"- {x}" for x in items)

def _format_options_block(options_any: Any) -> str:
    # options may be dict, list, or JSON str
    if isinstance(options_any, str):
        try:
            obj = json.loads(options_any)
            options_any = obj
        except Exception:
            pass
    if isinstance(options_any, dict):
        pairs = [(k, options_any[k]) for k in sorted(options_any.keys())]
    elif isinstance(options_any, list):
        # assume ["A. xxx", "B. yyy"] or ["xxx", "yyy"]
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        pairs = []
        for i, v in enumerate(options_any):
            k = letters[i] if i < len(letters) else str(i+1)
            text = re.sub(r"^[A-Za-z]\s*[.)]\s*", "", str(v)).strip()
            pairs.append((k, text))
    else:
        return "(options unavailable)"
    return "\n".join(f"{k}. {str(v)}" for k, v in pairs)

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

    s = (raw or "").strip()

    # 去掉三引号围栏（含```json）
    if s.startswith("```"):
        s2 = s[3:]
        if s2.startswith("json"):
            s2 = s2[4:] if len(s2) >= 4 and s2[3] == '\n' else s2.lstrip("json").lstrip()
        fence_end = s2.find("```")
        if fence_end != -1:
            s = s2[:fence_end].strip()

    # 抓第一个 {...}
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
# Build prompt from row (MCQ)
# -----------------------------
def _extract_mcq_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    兼容两种字段：
      1) 直接字段: row["question"], row["options"]
      2) 组合对象: row["generated_vqa"] 里有 question/options
    """
    q, opts = None, None
    if "question" in row and "options" in row:
        q, opts = row.get("question"), row.get("options")
    elif "generated_vqa" in row and isinstance(row["generated_vqa"], dict):
        q = row["generated_vqa"].get("question")
        opts = row["generated_vqa"].get("options")
    if q is None or opts is None:
        return None
    return {"question": q, "options": opts}

def build_prompt_from_row(row: Dict[str, Any]) -> Optional[str]:
    mcq = _extract_mcq_from_row(row)
    if mcq is None:
        return None

    question = str(mcq["question"])
    options_block = _format_options_block(mcq["options"])
    mod_block = _format_allowed_block(MODALITY_LABELS)
    anat_block = _format_allowed_block(ANATOMY_LABELS)

    return (LABEL_PROMPT_TEMPLATE
            .replace("{STEM_TEXT}", question)
            .replace("{OPTIONS_BLOCK}", options_block)
            .replace("{MODALITY_BLOCK}", mod_block)
            .replace("{ANATOMY_BLOCK}", anat_block))

# ========== GLM chat template helpers ==========
def _make_glm_image_placeholders(num_images: int, modality: str = "image") -> str:
    if modality == "image":
        token = "<|begin_of_image|><|image|><|end_of_image|>"
    else:
        token = "<|begin_of_video|><|video|><|end_of_video|>"
    return token * num_images

def _wrap_glm_chat(prompt_text: str, vision_placeholders: str) -> str:
    # system 简单，核心 user 放完整指令与占位符，思考模式（<think>）不出现
    return (
        "[gMASK]<sop><|system|>\nYou are a helpful assistant."
        "<|user|>\n"
        f"{vision_placeholders}{prompt_text}"
        "<|assistant|>assistant\n"
    )

def _collect_pil_images(row: Dict[str, Any], args) -> List[Image.Image]:
    """收集 PIL 图片，加入最短边保护：skip / pad / resize。"""
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

    # 先尝试 tokenizer.apply_chat_template（显式关闭思考）
    prompt = None
    if tokenizer is not None:
        try:
            content = []
            if pil_images:
                content += [{"type": "image", "image": img} for img in pil_images]
            content += [{"type": "text", "text": prompt_text}]
            glmmessages = [{"role": "user", "content": content}]

            prompt = tokenizer.apply_chat_template(
                glmmessages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # 关键：关思考
            )
        except Exception:
            try:
                prompt = tokenizer.apply_chat_template(
                    glmmessages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
            except Exception:
                prompt = None

    if prompt is None:
        # 回退：手写模板（不包含 <think>）
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

        mm_kwargs_glm = {"size": {"shortest_edge": 12544, "longest_edge": 47040000}, "fps": 1}

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
                                desc=f"[Global DP Rank {global_dp_rank}] Modality/Anatomy Tagging"):
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

                        record = dict(row)
                        record["modality"] = None
                        record["anatomy"] = None
                        record["parse_error"] = "missing_mcq"
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

                    # 解析并规范化
                    mod = parsed.get("modality") if isinstance(parsed, dict) else None
                    anat = parsed.get("anatomy") if isinstance(parsed, dict) else None
                    mod = _canonicalize(mod, MODALITY_LABELS)
                    anat = _canonicalize(anat, ANATOMY_LABELS)

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

                    record["modality"] = mod
                    record["anatomy"] = anat
                    if mod is None or anat is None:
                        record["parse_error"] = "invalid_or_missing_label"
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
@click.option("--model", type=str, default="zai-org/GLM-4.5V", show_default=True)
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
              help="最短边保护阈值。")
@click.option("--small_image_policy", type=click.Choice(["skip", "pad", "resize"]),
              default="skip", show_default=True,
              help="遇到过小图：skip=跳过；pad=黑边补齐到阈值；resize=放大到阈值。")
# output
@click.option("--output_dir", default="outputs/modality_anatomy/", type=str, show_default=True)
@click.option("--overwrite", is_flag=True, help="Overwrite shard outputs.")
@click.option("--merge_after", is_flag=True, help="Merge shards into results.jsonl after finishing.")
@click.option("--keep_raw_output", is_flag=True, help="Keep raw_model_output for debugging.")
# misc
@click.option("--ignore_image", is_flag=True, help="若设为 True，将忽略图像，仅按文本分类。默认不忽略。")
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
