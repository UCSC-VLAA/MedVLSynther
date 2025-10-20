#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, sys, unicodedata
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, List

# ---------- 文本规范化 & 签名 ----------
def _norm_text(s: Any) -> str:
    """大小写不敏感、空白折叠、NFKC 规范化"""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()

def _options_signature(opts: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """将 options 映射成 (letter, normalized_text) 的有序 tuple（按字母排序）"""
    items = []
    for k, v in sorted(opts.items(), key=lambda kv: kv[0]):
        items.append((str(k), _norm_text(v)))
    return tuple(items)

def _vqa_signature(question: str, options: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return _norm_text(question), _options_signature(options)

# ---------- 从 prompts 解析 Question 与 Options ----------
_Q_RE = re.compile(r"Question:\s*(.*?)\n\s*\n\s*Options:", re.S)
def _extract_q_and_options_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """从 eval prompts 中提取 Question 与 Options"""
    if not isinstance(prompt, str):
        return None, None
    m = _Q_RE.search(prompt)
    if not m:
        return None, None
    question = m.group(1).strip()
    rest = prompt[m.end():]

    # 选项块截止到 <|assistant|>（若无则到串末尾）
    end_tag = "<|assistant|>"
    end_idx = rest.find(end_tag)
    options_block = rest[:end_idx] if end_idx != -1 else rest

    # 解析 A./B./...；选项之间常以空行分隔
    options_block = options_block.strip()
    opts: Dict[str, str] = {}

    # 尝试更鲁棒的正则（在下一个空行+新字母或结束处截断）
    pat = re.compile(r"^\s*([A-Z])\.\s*(.*?)\s*(?=(?:\n\s*\n\s*[A-Z]\.|$))", re.M | re.S)
    for g in pat.finditer(options_block):
        letter = g.group(1)
        text = g.group(2).strip()
        if text:
            opts[letter] = text

    # 兜底：逐行
    if not opts:
        for line in options_block.splitlines():
            line = line.strip()
            if not line:
                continue
            m2 = re.match(r"^([A-Z])\.\s*(.+)$", line)
            if m2:
                opts[m2.group(1)] = m2.group(2).strip()

    return (question if question else None), (opts if opts else None)

# ---------- 从 output_text 抽取 <think>... ----------
def _extract_think(output_text: str) -> Optional[str]:
    if not isinstance(output_text, str):
        return None
    m = re.search(r"<think>\s*(.*?)\s*</think>", output_text, re.S | re.I)
    if not m:
        return None
    return m.group(1).strip()

# ---------- 读取 JSONL ----------
def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as e:
                print(f"[warn] {path}:{ln} JSON 解析失败: {e}", file=sys.stderr)

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="合并 eval 正确样本的 reasoning 到元数据 JSONL（按 Question+Options 匹配）")
    ap.add_argument("--eval_jsonl", required=True, help="eval_results.jsonl 路径")
    ap.add_argument("--meta_jsonl", required=True, help="元数据 jsonl 路径（包含 generated_vqa）")
    ap.add_argument("--out_jsonl", required=True, help="输出的新 jsonl 路径")
    ap.add_argument("--use_answer_to_break_ties", action="store_true",
                    help="如发生一题多条元数据同签名，优先用答案字母/文本进一步筛选")
    args = ap.parse_args()

    # 1) 读取元数据并建索引（按 Question+Options 签名）
    meta_index: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List[Dict[str, Any]]] = defaultdict(list)
    meta_count = 0
    for ln, obj in _iter_jsonl(args.meta_jsonl):
        gvqa = obj.get("generated_vqa", {})
        q = gvqa.get("question", None)
        opts = gvqa.get("options", None)
        if isinstance(q, str) and isinstance(opts, dict) and opts:
            sig = _vqa_signature(q, opts)
            meta_index[sig].append(obj)
            meta_count += 1
        else:
            # 跳过不含题/选项的行
            continue
    print(f"[info] 载入元数据 {meta_count} 条，构建签名桶 {len(meta_index)} 个。", file=sys.stderr)

    # 2) 遍历 eval_results
    total = 0
    correct = 0
    matched = 0
    written = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for ln, ej in _iter_jsonl(args.eval_jsonl):
            total += 1

            pops = ej.get("parsed_outputs", [])
            if not isinstance(pops, list) or not pops:
                continue

            # 找第一个 is_correct==True
            chosen = None
            for it in pops:
                if isinstance(it, dict) and it.get("is_correct", False) is True and "output_text" in it:
                    chosen = it
                    break
            if not chosen:
                continue
            correct += 1

            reasoning = _extract_think(chosen.get("output_text", ""))
            if not reasoning:
                continue

            prompt = ej.get("prompts", "")
            q, opts = _extract_q_and_options_from_prompt(prompt)
            if not q or not opts:
                continue

            sig = _vqa_signature(q, opts)
            cand_list = meta_index.get(sig, [])

            if not cand_list:
                # 没匹配到，跳过
                continue

            # 若有多候选，尝试用答案 disambiguate（可选）
            sel_obj = None
            if len(cand_list) == 1 or not args.use_answer_to_break_ties:
                sel_obj = cand_list[0]
            else:
                want_letter = ej.get("answer_label", None)
                want_text = ej.get("answer", None)
                # 优先用 letter，再用文本
                for c in cand_list:
                    gvqa = c.get("generated_vqa", {})
                    if want_letter and isinstance(gvqa.get("answer"), str) and gvqa["answer"].strip() == want_letter.strip():
                        sel_obj = c
                        break
                if sel_obj is None and want_text:
                    for c in cand_list:
                        # 如果元数据没存答案文本，只能跳过
                        # （此处可根据你的元数据结构自行扩展）
                        pass

                if sel_obj is None:
                    # 仍无法消歧，就取第一个
                    sel_obj = cand_list[0]

            matched += 1

            # 复制一份，新增/覆盖顶层 reasoning 键（不动原索引）
            out_obj = dict(sel_obj)
            out_obj["reasoning"] = reasoning

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"[done] eval总数={total} | 正确样本={correct} | 匹配成功={matched} | 新jsonl写入={written} 条")
    # 按你的要求，这里报告新的 jsonl 条目数：
    print(written)

if __name__ == "__main__":
    main()
