#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, Optional

# ===== 白名单（新增 "Brain"）=====
ANATOMY_WHITELIST = [
    "Lung","Mammary Gland","Hand","Upper Limb","Eye","Uterus","Intestine","Skin",
    "Shoulder","Kidney","Gallbladder","Pancreas","Spleen","Liver","Pelvic","Ovary",
    "Blood Vessel","Spine","Urinary System","Adipose Tissue","Muscle Tissue",
    "Oral Cavity","Knee","Foot","Lower Limb",
    # 新增
    "Brain", "Heart"
]

WL_LOWER = {w.lower(): w for w in ANATOMY_WHITELIST}

# ===== 常见别名/变体 -> 白名单项（大小写不敏感）=====
ALIAS = {
    # 直接对应白名单的大小写/变体
    "small intestine": "Intestine",
    "ovaries": "Ovary",
    "ovarian follicle": "Ovary",
    "pelvis": "Pelvic",
    "pelvic cavity": "Pelvic",
    "spinal cord": "Spine",
    "ibat": "Adipose Tissue",
    "interscapular brown adipose tissue (ibat)": "Adipose Tissue",
    "tooth": "Oral Cavity",

    # “血管”范畴：更细的血管名并到 Blood Vessel
    "umbilical vein": "Blood Vessel",
    "thoracic aorta": "Blood Vessel",

    # “肌肉组织”
    "upper trapezius": "Muscle Tissue",

    # 眼相关（保守并入 Eye）
    "orbit": "Eye",

    # 新增 Brain 及其常见下位词
    "brain": "Brain",
    "brainstem": "Brain",
    "cerebral hemispheres": "Brain",
    "thalamus": "Brain",
    "amygdala": "Brain",
    "pituitary gland": "Brain",

    # add heart
    "heart": "Heart"
}

# 尽量从 <|begin_of_box|>{...}<|end_of_box|> 中取 JSON；退化到 '"anatomy": "..."' 的 KV 正则
BOX_RE = re.compile(r"\{.*?\}", re.DOTALL)
ANA_KV_RE = re.compile(r'"anatomy"\s*:\s*"([^"]+)"', re.IGNORECASE)

def extract_anatomy(raw: Any) -> Optional[str]:
    if not raw:
        return None
    s = str(raw)
    m = BOX_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            val = obj.get("anatomy", None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            pass
    m2 = ANA_KV_RE.search(s)
    if m2:
        v = m2.group(1).strip()
        if v:
            return v
    return None

def normalize_anatomy(raw_val: Optional[str]) -> (str, str):
    """
    返回 (final_label, source_tag)
    - 命中白名单（不区分大小写） -> 对应白名单项
    - 命中 ALIAS -> 对应白名单项
    - 否则 -> "Other"
    """
    if not raw_val:
        return ("Other", "no-anatomy-in-raw")
    key = raw_val.strip().lower()

    # 白名单直中
    if key in WL_LOWER:
        return (WL_LOWER[key], "whitelist-direct")

    # 别名映射
    if key in ALIAS:
        return (ALIAS[key], "mapped-alias")

    # 一些更松弛的安全映射（尽量少做以免误并）
    LOOSE = {
        "lymph node": "Other",      # 避免误并
        "lymph nodes": "Other",
        "blood vessel": "Blood Vessel",
        "vein": "Blood Vessel",
        "artery": "Blood Vessel",
    }
    if key in LOOSE:
        return (LOOSE[key], "loose-map")

    return ("Other", "unmapped")

def main():
    ap = argparse.ArgumentParser(description="Fix missing anatomy using raw_model_output + canonical mapping (with Brain added).")
    ap.add_argument("--input", required=True, help="missing.jsonl（anatomy 缺失条目）")
    ap.add_argument("--output", required=True, help="输出修复后的 JSONL")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    total = fixed = to_other = 0
    buckets: Dict[str, int] = {}

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            total += 1

            raw = rec.get("raw_model_output")
            raw_ana = extract_anatomy(raw)
            canon, src = normalize_anatomy(raw_ana)

            rec["anatomy"] = canon
            rec["anatomy_raw_extracted"] = raw_ana
            rec["anatomy_source"] = src

            if canon == "Other":
                to_other += 1
            else:
                fixed += 1
                buckets[canon] = buckets.get(canon, 0) + 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("========== Anatomy Fix Summary ==========")
    print(f"Input lines               : {total}")
    print(f"Mapped to whitelist       : {fixed}")
    print(f"Tagged as 'Other'         : {to_other}")
    if buckets:
        print("\n-- Mapped distribution --")
        for k, v in sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k}: {v}")
    print(f"\n[OK] Wrote -> {out_path}")

if __name__ == "__main__":
    main()
