#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, Optional

WHITELIST = [
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

# 映射表（大小写不敏感）
CANON_MAP = {
    # OCT 系
    "oct": "OCT (Optical Coherence Tomography)",
    "optical coherence tomography": "OCT (Optical Coherence Tomography)",
    "optical coherence tomography angiography (octa)": "OCT (Optical Coherence Tomography)",
    "casia2 anterior segment imaging": "OCT (Optical Coherence Tomography)",

    # CT 系
    "ct": "CT (Computed Tomography)",
    "ct angiography": "CT (Computed Tomography)",
    "micro-computed tomography (µct)": "CT (Computed Tomography)",
    "micro-computed tomography (μct)": "CT (Computed Tomography)",
    "micro computed tomography": "CT (Computed Tomography)",

    # X-Ray 系
    "x-ray": "X-Ray",
    "x ray": "X-Ray",
    "xray": "X-Ray",
    "digital subtraction angiography": "X-Ray",
    "mammography": "X-Ray",
    "digital tomosynthesis": "X-Ray",
    "uterine arteriography": "X-Ray",

    # Fundus 系（眼底）
    "fundus autofluorescence": "Fundus Photography",
    "fundus autofluorescence imaging": "Fundus Photography",
    "fluorescein angiography": "Fundus Photography",
    "fundus fluorescein angiography": "Fundus Photography",
    "indocyanine green angiography": "Fundus Photography",
    "indocyanine green angiography (icga)": "Fundus Photography",
    "ultrawidefield fluorescein angiography (uwf-fa)": "Fundus Photography",

    # 数码摄影系
    "digital photography": "Digital Photography",
    "slit-lamp imaging": "Digital Photography",
    "gonioscopy": "Digital Photography",

    # 内镜
    "endoscopy": "Endoscopy",
    "probe-based confocal laser endomicroscopy (pcle)": "Endoscopy",

    # 直接等于白名单中的其它项（大小写不同）
    "infrared reflectance imaging": "Infrared Reflectance Imaging",
    "mr (magnetic resonance imaging)": "MR (Magnetic Resonance Imaging)",
    "ultrasound": "Ultrasound",
    "microscopy images": "Microscopy Images",
    "colposcopy": "Colposcopy",
    "dermoscopy": "Dermoscopy",
    "fundus photography": "Fundus Photography",
}

WL_LOWER = {w.lower(): w for w in WHITELIST}

BOX_RE = re.compile(r"\{.*?\}", re.DOTALL)
MODALITY_KV_RE = re.compile(r'"modality"\s*:\s*"([^"]+)"', re.IGNORECASE)

def extract_modality(raw: Any) -> Optional[str]:
    """从 raw_model_output 里尽力抽出 modality 字段（返回原始字符串）。"""
    if not raw:
        return None
    s = str(raw)
    # 优先找 JSON 块
    m = BOX_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            val = obj.get("modality", None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            pass
    # 退而求其次：KV 正则
    m2 = MODALITY_KV_RE.search(s)
    if m2:
        v = m2.group(1).strip()
        if v:
            return v
    return None

def normalize_modality(raw_val: Optional[str]) -> (str, str):
    """
    规范化为白名单；返回 (final_modality, source_note)
    - 成功映射或直接命中白名单：返回白名单项
    - 否则：返回 "Other"
    """
    if not raw_val:
        return ("Other", "no-modality-in-raw")

    key = raw_val.strip().lower()

    # 直接命中白名单（不区分大小写）
    if key in WL_LOWER:
        return (WL_LOWER[key], "whitelist-direct")

    # 命中别名映射
    if key in CANON_MAP:
        return (CANON_MAP[key], "mapped-alias")

    # 常见简写再宽松一层（比如 'mri' -> MR，'ct scan' -> CT）
    loose = {
        "mri": "MR (Magnetic Resonance Imaging)",
        "ct scan": "CT (Computed Tomography)",
        "pet-ct": "Other",          # 保持 Other
        "pet/ct": "Other",
        "pet (computed tomography)": "Other",
        "pet": "Other",
        "spect": "Other",
        "eeg": "Other",
        "ecg": "Other",
        "ekg": "Other",
        "nirs": "Other",
        "phase-contrast imaging": "Other",  # 之前建议：不盲目并到显微
        "nirf": "Other",
        "near-infrared fluorescence (nirf)": "Other",
        "[18f]fdg pet": "Other",
    }
    if key in loose:
        return (loose[key], "loose-map")

    return ("Other", "unmapped")

def main():
    ap = argparse.ArgumentParser(description="Fix missing modality using raw_model_output + canonical mapping.")
    ap.add_argument("--input", required=True, help="missing.jsonl（modality 缺失条目）")
    ap.add_argument("--output", required=True, help="输出修复后的 JSONL")
    ap.add_argument("--keep-trace", action="store_true", help="在记录里保留 modality_raw_extracted 与 modality_source 字段（默认保留）")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    total = 0
    fixed = 0
    to_other = 0
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
            raw_mod = extract_modality(raw)
            canon, src = normalize_modality(raw_mod)

            # 写回/补齐 modality
            # 如果原本有 modality 且为空/None，也覆盖；如果有非空值，这里也可以选择覆盖（更一致）
            rec["modality"] = canon
            rec["modality_raw_extracted"] = raw_mod
            rec["modality_source"] = src

            if canon == "Other":
                to_other += 1
            else:
                fixed += 1
                buckets[canon] = buckets.get(canon, 0) + 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("========== Modality Fix Summary ==========")
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
