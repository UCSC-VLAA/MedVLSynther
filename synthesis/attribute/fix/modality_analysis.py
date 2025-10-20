#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

MODALITY_LABELS = [
    "Colposcopy","CT (Computed Tomography)","Digital Photography","Fundus Photography",
    "Infrared Reflectance Imaging","MR (Magnetic Resonance Imaging)","OCT (Optical Coherence Tomography)",
    "Dermoscopy","Endoscopy","Microscopy Images","X-Ray","Ultrasound", "Other"
]
ANATOMY_LABELS = [
    "Lung","Mammary Gland","Hand","Upper Limb","Eye","Uterus","Intestine","Skin","Shoulder","Kidney",
    "Gallbladder","Pancreas","Spleen","Liver","Pelvic","Ovary","Blood Vessel","Spine","Urinary System",
    "Adipose Tissue","Muscle Tissue","Oral Cavity","Knee","Foot","Lower Limb", "Brain", "Heart", "Other"
]

def _canon_map(whitelist: List[str]) -> Dict[str, str]:
    return {re.sub(r"\s+"," ",w.strip().lower()): w for w in whitelist}
MOD_CANON = _canon_map(MODALITY_LABELS)
ANA_CANON = _canon_map(ANATOMY_LABELS)

def load_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try: yield json.loads(s)
            except Exception: continue

def _norm_q(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _extract_wds_members_from_paths(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        if not p: continue
        p = str(p).strip()
        m = p.split("::",1)[1] if "::" in p else os.path.basename(p)
        out.append(m.lower())
    return out

def image_members(rec: Dict[str, Any]) -> List[str]:
    paths = []
    imgs = rec.get("images")
    if isinstance(imgs, list):
        for e in imgs:
            if isinstance(e, dict) and e.get("path"): paths.append(e["path"])
            elif isinstance(e, str): paths.append(e)
    v = rec.get("image_wds_path", None)
    if isinstance(v, list): paths += v
    elif isinstance(v, str): paths.append(v)
    mems = _extract_wds_members_from_paths(paths)
    if mems: return sorted(set(mems))
    fns = rec.get("image_file_name", None)
    if isinstance(fns, list) and fns: return sorted({str(x).lower() for x in fns if x})
    if isinstance(fns, str) and fns: return [fns.strip().lower()]
    return []

def make_key_qimg(rec: Dict[str, Any]) -> Tuple[str, Tuple[str,...]]:
    return (_norm_q(rec.get("question") or (rec.get("generated_vqa") or {}).get("question") or ""),
            tuple(image_members(rec)))

def _canon_label(raw: Any, table: Dict[str,str]) -> Optional[str]:
    if raw is None: return None
    s = str(raw).strip()
    if not s: return None
    s_norm = re.sub(r"\s+"," ", s.lower())
    return table.get(s_norm)  # None if OOV

def _options_str(rec: Dict[str,Any]) -> str:
    opts = rec.get("options")
    if isinstance(opts, dict):
        try: return json.dumps(opts, ensure_ascii=False)
        except Exception: return str(opts)
    return "" if opts is None else str(opts)

def analyze(path: str, show_missing: int = 5, dump_missing: Optional[str] = None):
    total = 0
    mod_present = ana_present = 0
    mod_counter = Counter(); ana_counter = Counter()
    mod_oov = Counter(); ana_oov = Counter()

    key2mods = defaultdict(set); key2anas = defaultdict(set)

    missing_modality = []   # records missing modality
    missing_anatomy  = []   # records missing anatomy

    for rec in load_jsonl(path):
        total += 1
        key = make_key_qimg(rec)

        # modality
        raw_m = rec.get("modality", None)
        if raw_m is None or str(raw_m).strip()=="":
            missing_modality.append(rec)
        else:
            mod_present += 1
            canon_m = _canon_label(raw_m, MOD_CANON)
            if canon_m is not None:
                mod_counter[canon_m] += 1
                key2mods[key].add(canon_m)
            else:
                mod_oov[str(raw_m)] += 1

        # anatomy
        raw_a = rec.get("anatomy", None)
        if raw_a is None or str(raw_a).strip()=="":
            missing_anatomy.append(rec)
        else:
            ana_present += 1
            canon_a = _canon_label(raw_a, ANA_CANON)
            if canon_a is not None:
                ana_counter[canon_a] += 1
                key2anas[key].add(canon_a)
            else:
                ana_oov[str(raw_a)] += 1

    mod_conflicts = [(k, list(v)) for k,v in key2mods.items() if len(v)>1]
    ana_conflicts = [(k, list(v)) for k,v in key2anas.items() if len(v)>1]

    # -------- report --------
    print("========== Summary ==========")
    print(f"File                    : {path}")
    print(f"Total lines             : {total}\n")

    print("========== Modality ==========")
    print(f"Records w/ modality     : {mod_present}")
    if mod_counter:
        print(f"Distinct (whitelist)    : {len(mod_counter)} / {len(MODALITY_LABELS)} possible")
        for lab,cnt in mod_counter.most_common():
            r = cnt / mod_present if mod_present else 0.0
            print(f"  - {lab}: {cnt} ({r:.2%})")
    if mod_oov:
        print("\n[OOV modality values]")
        for v,c in mod_oov.most_common(10): print(f"  * '{v}' -> {c}")
    print(f"\nMissing modality        : {len(missing_modality)}")
    print(f"Conflicts (same key, different modality): {len(mod_conflicts)}\n")

    print("========== Anatomy ==========")
    print(f"Records w/ anatomy      : {ana_present}")
    if ana_counter:
        print(f"Distinct (whitelist)    : {len(ana_counter)} / {len(ANATOMY_LABELS)} possible")
        for lab,cnt in ana_counter.most_common():
            r = cnt / ana_present if ana_present else 0.0
            print(f"  - {lab}: {cnt} ({r:.2%})")
    if ana_oov:
        print("\n[OOV anatomy values]")
        for v,c in ana_oov.most_common(10): print(f"  * '{v}' -> {c}")
    print(f"\nMissing anatomy         : {len(missing_anatomy)}")
    print(f"Conflicts (same key, different anatomy): {len(ana_conflicts)}\n")

    # ---- print missing examples ----
    def _brief(rec):
        return {
            "dataset_index": rec.get("dataset_index"),
            "images": image_members(rec),
            "question": rec.get("question") or (rec.get("generated_vqa") or {}).get("question"),
            "options": _options_str(rec),
            "modality": rec.get("modality"),
            "anatomy": rec.get("anatomy"),
            "raw_model_output": rec.get("raw_model_output"),
        }

    if show_missing > 0 and missing_modality:
        print(f"-- Examples missing MODALITY (showing up to {show_missing}) --")
        for rec in missing_modality[:show_missing]:
            print(json.dumps(_brief(rec), ensure_ascii=False))

    if show_missing > 0 and missing_anatomy:
        print(f"\n-- Examples missing ANATOMY (showing up to {show_missing}) --")
        for rec in missing_anatomy[:show_missing]:
            print(json.dumps(_brief(rec), ensure_ascii=False))

    # ---- optional dump ----
    if dump_missing:
        outp = Path(dump_missing)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for rec in missing_modality:
                b = _brief(rec); b["missing"] = "modality"
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
            for rec in missing_anatomy:
                b = _brief(rec); b["missing"] = b.get("missing","") + ("+anatomy" if "missing" in b else "anatomy")
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
        print(f"\n[OK] Dumped missing cases -> {outp}")

def main():
    ap = argparse.ArgumentParser(description="Analyze modality/anatomy coverage and print/dump missing entries.")
    ap.add_argument("--input", required=True, help="JSONL path")
    ap.add_argument("--show-missing", type=int, default=5, help="print up to N missing examples per field")
    ap.add_argument("--dump-missing", type=str, default=None, help="optional JSONL to dump all missing records")
    args = ap.parse_args()
    analyze(args.input, show_missing=args.show_missing, dump_missing=args.dump_missing)

if __name__ == "__main__":
    main()
