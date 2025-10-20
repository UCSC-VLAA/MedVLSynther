#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, Counter

# ---------- IO ----------

def iter_jsonl_paths(path_like: str) -> Iterable[Path]:
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".jsonl":
        yield p; return
    if p.is_dir():
        for f in sorted(p.rglob("*.jsonl")):
            if f.suffix.lower() == ".jsonl":
                yield f
        return
    for f in sorted(Path().glob(path_like)):
        if f.is_file() and f.suffix.lower() == ".jsonl":
            yield f

def load_line(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _normalize_list_str(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]

def _norm_field(x) -> Optional[str]:
    if x is None: return None
    s = str(x).strip()
    if not s or s.lower() in ("none","null","nan"):
        return None
    return s

def per_record_images(rec: Dict[str, Any]) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
    aid = _norm_field(rec.get("article_accession_id", None))
    ids = _normalize_list_str(rec.get("image_id", []))
    names = _normalize_list_str(rec.get("image_file_name", []))
    n = max(len(ids), len(names))
    out = []
    for i in range(n or 1):  
        img_id = _norm_field(ids[i]) if i < len(ids) else None
        img_nm = _norm_field(names[i]) if i < len(names) else None
        if img_id is None and img_nm is None:
            out.append((aid, "__UNKNOWN__", None))
        else:
            out.append((aid, img_id, img_nm))
    return out

def get_qtype(rec: Dict[str, Any]) -> str:
    qtype = rec.get("Question Type")
    if qtype is None:
        qtype = rec.get("question_type")
    qtype = str(qtype).strip() if qtype is not None else ""
    return qtype if qtype else "<UNKNOWN>"

def get_modality(rec: Dict[str, Any]) -> str:
    m = rec.get("modality")
    m = str(m).strip() if isinstance(m, str) else ""
    return m if m else "<UNKNOWN>"

def get_anatomy(rec: Dict[str, Any]) -> str:
    a = rec.get("anatomy")
    a = str(a).strip() if isinstance(a, str) else ""
    return a if a else "<UNKNOWN>"

# ---------- table formatting ----------

def format_table(rows: List[Dict[str, Any]], field_name: str) -> str:
    headers = [field_name, "#Questions", "%", "#Images(unique)", "AvgImgs/Q", "#Q@1", "#Q@2", "#Q@3", "#Q@4", "#Q@5", "#Q@6+"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        widths[field_name] = max(widths[field_name], len(str(r[field_name])))
        widths["#Questions"] = max(widths["#Questions"], len(str(r["#Questions"])))
        widths["%"]         = max(widths["%"], len(f'{r["%"]:.2f}'))
        widths["#Images(unique)"] = max(widths["#Images(unique)"], len(str(r["#Images(unique)"])))
        widths["AvgImgs/Q"] = max(widths["AvgImgs/Q"], len(f'{r["AvgImgs/Q"]:.2f}'))
        for k in ["#Q@1","#Q@2","#Q@3","#Q@4","#Q@5","#Q@6+"]:
            widths[k] = max(widths[k], len(str(r[k])))

    lines = []
    line_hdr = "  ".join([
        headers[0].ljust(widths[field_name]),
        headers[1].rjust(widths["#Questions"]),
        headers[2].rjust(widths["%"]),
        headers[3].rjust(widths["#Images(unique)"]),
        headers[4].rjust(widths["AvgImgs/Q"]),
        headers[5].rjust(widths["#Q@1"]),
        headers[6].rjust(widths["#Q@2"]),
        headers[7].rjust(widths["#Q@3"]),
        headers[8].rjust(widths["#Q@4"]),
        headers[9].rjust(widths["#Q@5"]),
        headers[10].rjust(widths["#Q@6+"]),
    ])
    lines.append(line_hdr)
    lines.append("-" * len(line_hdr))

    for r in rows:
        lines.append("  ".join([
            str(r[field_name]).ljust(widths[field_name]),
            str(r["#Questions"]).rjust(widths["#Questions"]),
            f'{r["%"]:.2f}'.rjust(widths["%"]),
            str(r["#Images(unique)"]).rjust(widths["#Images(unique)"]),
            f'{r["AvgImgs/Q"]:.2f}'.rjust(widths["AvgImgs/Q"]),
            str(r["#Q@1"]).rjust(widths["#Q@1"]),
            str(r["#Q@2"]).rjust(widths["#Q@2"]),
            str(r["#Q@3"]).rjust(widths["#Q@3"]),
            str(r["#Q@4"]).rjust(widths["#Q@4"]),
            str(r["#Q@5"]).rjust(widths["#Q@5"]),
            str(r["#Q@6+"]).rjust(widths["#Q@6+"]),
        ]))

    return "\n".join(lines)

# ---------- aggregation using yesterday's unique key ----------

def summarize_by_field(files: List[Path], field_name: str, pick_label):
    by_label = defaultdict(lambda: {
        "q_cnt": 0,
        "per_record_dist": Counter(),    # N -> #Q
        "unique_imgs": set(),            # set of (aid,id,name)
    })
    global_unique_imgs = set()
    total_q = 0

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = load_line(line)
                if rec is None:
                    continue
                total_q += 1
                label = pick_label(rec)
                imgs = per_record_images(rec)

                by_label[label]["q_cnt"] += 1

                k = len(imgs)
                if k >= 6: k = 6
                by_label[label]["per_record_dist"][k] += 1

                for im in imgs:
                    by_label[label]["unique_imgs"].add(im)
                    global_unique_imgs.add(im)

    rows = []
    ordered = sorted(by_label.items(), key=lambda kv: kv[1]["q_cnt"], reverse=True)
    for label, v in ordered:
        q_cnt = v["q_cnt"]
        uniq_imgs = len(v["unique_imgs"])
        pct = (100.0 * q_cnt / total_q) if total_q else 0.0
        avg = (uniq_imgs / q_cnt) if q_cnt else 0.0
        dist = v["per_record_dist"]
        rows.append({
            field_name: label,
            "#Questions": q_cnt,
            "%": pct,
            "#Images(unique)": uniq_imgs,
            "AvgImgs/Q": avg,
            "#Q@1": dist.get(1, 0),
            "#Q@2": dist.get(2, 0),
            "#Q@3": dist.get(3, 0),
            "#Q@4": dist.get(4, 0),
            "#Q@5": dist.get(5, 0),
            "#Q@6+": dist.get(6, 0),
        })

    # TOTAL
    total_unique = len(global_unique_imgs)
    total_avg = (total_unique / total_q) if total_q else 0.0
    total_dist = Counter()
    for _, v in by_label.items():
        total_dist.update(v["per_record_dist"])
    rows.append({
        field_name: "TOTAL",
        "#Questions": total_q,
        "%": 100.0,
        "#Images(unique)": total_unique,
        "AvgImgs/Q": total_avg,
        "#Q@1": total_dist.get(1, 0),
        "#Q@2": total_dist.get(2, 0),
        "#Q@3": total_dist.get(3, 0),
        "#Q@4": total_dist.get(4, 0),
        "#Q@5": total_dist.get(5, 0),
        "#Q@6+": total_dist.get(6, 0),
    })

    return rows

def main():
    ap = argparse.ArgumentParser(description="Summarize with UNIQUE image by (aid,image_id,image_file_name) for Question Type / modality / anatomy + 1–6 image distribution.")
    ap.add_argument("--input", required=True, help="JSONL path")
    ap.add_argument("--field", choices=["qtype","modality","anatomy","all"], default="all",
                    help="which field to analyze, default all")
    ap.add_argument("--out-csv-prefix", default=None,
                    help="optional：export CSV (*_qtype.csv / *_modality.csv / *_anatomy.csv)")
    args = ap.parse_args()

    files = list(iter_jsonl_paths(args.input))
    if not files:
        print("[ERR] No jsonl found."); return

    want = ["qtype","modality","anatomy"] if args.field == "all" else [args.field]
    outputs = {}

    if "qtype" in want:
        rows = summarize_by_field(files, "Question Type", get_qtype)
        print(format_table(rows, "Question Type"))
        outputs["qtype"] = rows

    if "modality" in want:
        rows = summarize_by_field(files, "modality", get_modality)
        print("\n" + format_table(rows, "modality"))
        outputs["modality"] = rows

    if "anatomy" in want:
        rows = summarize_by_field(files, "anatomy", get_anatomy)
        print("\n" + format_table(rows, "anatomy"))
        outputs["anatomy"] = rows

    # CSV
    if args.out_csv_prefix:
        import csv
        base = Path(args.out_csv_prefix)
        for key, rows in outputs.items():
            outp = base.with_name(base.name + f"_{key}.csv")
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([key,"#Questions","%","#Images(unique)","AvgImgs/Q","#Q@1","#Q@2","#Q@3","#Q@4","#Q@5","#Q@6+"])
                for r in rows:
                    w.writerow([
                        r[key], r["#Questions"], f'{r["%"]:.2f}', r["#Images(unique)"],
                        f'{r["AvgImgs/Q"]:.4f}', r["#Q@1"], r["#Q@2"], r["#Q@3"], r["#Q@4"], r["#Q@5"], r["#Q@6+"]
                    ])
            print(f"[OK] CSV saved -> {outp}")

if __name__ == "__main__":
    main()
