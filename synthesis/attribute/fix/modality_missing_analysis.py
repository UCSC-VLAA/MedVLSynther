#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List
from collections import Counter

# --------- IO ---------
def load_jsonl(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue

# --------- parsing helpers ---------
BOX_RE = re.compile(r"<\|begin_of_box\|>(.*)<\|end_of_box\|>", re.DOTALL | re.IGNORECASE)

def between_braces(s: str) -> Optional[str]:
    """取第一个 '{' 到最后一个 '}' 的子串"""
    try:
        i = s.index("{")
        j = s.rindex("}")
        if i < j:
            return s[i:j+1]
    except ValueError:
        return None
    return None

TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

def try_parse_json_loose(s: str) -> Optional[Dict[str, Any]]:
    """
    尝试把各种“近似 JSON”的字符串修正为可解析 JSON：
      - 去掉 box 包裹
      - 截取 {...}
      - 修掉尾逗号、单引号
    """
    if s is None:
        return None
    if not isinstance(s, str):
        # 有些记录直接就是 dict
        if isinstance(s, dict):
            return s
        return None

    raw = s.strip()
    # 去 box 包壳
    m = BOX_RE.search(raw)
    if m:
        raw = m.group(1).strip()

    # 截 {...}
    sub = between_braces(raw) or raw

    # 先直接尝试
    try:
        return json.loads(sub)
    except Exception:
        pass

    # 修尾逗号
    sub2 = TRAILING_COMMA_RE.sub(r"\1", sub)

    # 单引号替换（粗暴但实用）
    if "'" in sub2 and '"' not in sub2:
        sub2 = sub2.replace("'", '"')
    # 再试
    try:
        return json.loads(sub2)
    except Exception:
        # 正则兜底读键值
        return None

# 兜底：正则直接抠 "modality" / "anatomy" 的值（大小写不敏感）
VAL_RE = re.compile(r'(?i)"?\b(modality|anatomy)\b"?\s*:\s*"([^"]+)"')

def extract_labels_from_raw(raw_model_output: Any) -> Tuple[Optional[str], Optional[str], bool]:
    """
    返回 (modality_raw, anatomy_raw, parsed_as_json)
    parsed_as_json 表示是否成功以 JSON 方式解析（便于评估原始质量）
    """
    # 先 json 解析
    js = try_parse_json_loose(raw_model_output)
    if isinstance(js, dict):
        # 大小写与别名不处理，这里只看原值“有多少种可能”
        m = js.get("modality")
        a = js.get("anatomy")
        # 有些键大小写不同
        if m is None:
            m = js.get("image_modality") or js.get("Modality") or js.get("MODALITY")
        if a is None:
            a = js.get("organ") or js.get("Anatomy") or js.get("ANATOMY")
        def norm(x):
            if x is None: return None
            s = str(x).strip()
            return s if s else None
        return norm(m), norm(a), True

    # 再正则兜底
    if isinstance(raw_model_output, str):
        raw = raw_model_output.strip()
        # 去 box
        mbox = BOX_RE.search(raw)
        if mbox:
            raw = mbox.group(1)
        mods = {}; anas = {}
        for k, v in VAL_RE.findall(raw):
            key = k.lower()
            v = v.strip()
            if not v: continue
            if key == "modality": mods[v] = True
            elif key == "anatomy": anas[v] = True
        mv = next(iter(mods.keys()), None)
        av = next(iter(anas.keys()), None)
        if mv or av:
            return mv, av, False

    return None, None, False

# --------- main ---------
def main():
    ap = argparse.ArgumentParser(description="Scan missing-modality/anatomy JSONL and enumerate raw_model_output label variants.")
    ap.add_argument("--input", required=True, help="dumped jsonl of missing entries")
    ap.add_argument("--top", type=int, default=50, help="print top-N variants for each field")
    args = ap.parse_args()

    total = 0
    parsed_json_ok = 0

    mod_counter = Counter()
    ana_counter = Counter()

    mod_missing_any = 0
    ana_missing_any = 0

    bad_raw_samples: List[str] = []

    for rec in load_jsonl(args.input):
        total += 1
        rmo = rec.get("raw_model_output", None)
        mod, ana, ok_json = extract_labels_from_raw(rmo)
        if ok_json:
            parsed_json_ok += 1
        # modality
        if mod is None:
            mod_missing_any += 1
        else:
            mod_counter[mod] += 1
        # anatomy
        if ana is None:
            ana_missing_any += 1
        else:
            ana_counter[ana] += 1

        if (mod is None and ana is None) and len(bad_raw_samples) < 5:
            bad_raw_samples.append(str(rmo)[:300])

    print("========== Scan raw_model_output (missing entries) ==========")
    print(f"Input file          : {args.input}")
    print(f"Total lines         : {total}")
    print(f"JSON parsed OK      : {parsed_json_ok}")
    print("")

    # ---- Modality ----
    print("---- Modality candidates ----")
    print(f"Distinct labels     : {len(mod_counter)}")
    if mod_counter:
        for lab, cnt in mod_counter.most_common(args.top):
            print(f"  - {lab}: {cnt}")
    print(f"Unrecoverable (no modality found in raw): {mod_missing_any}")
    print("")

    # ---- Anatomy ----
    print("---- Anatomy candidates ----")
    print(f"Distinct labels     : {len(ana_counter)}")
    if ana_counter:
        for lab, cnt in ana_counter.most_common(args.top):
            print(f"  - {lab}: {cnt}")
    print(f"Unrecoverable (no anatomy found in raw): {ana_missing_any}")

    if bad_raw_samples:
        print("\n-- Samples of raw_model_output that yielded nothing (truncated) --")
        for i, s in enumerate(bad_raw_samples, 1):
            print(f"[{i}] {s}")

if __name__ == "__main__":
    main()
