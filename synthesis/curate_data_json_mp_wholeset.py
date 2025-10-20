#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, tarfile, argparse, random, math
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------------- Allowed labels ----------------
CLIN_SECONDARIES = [
    'x-ray radiography', 'optical coherence tomography', 'endoscopy', 
    'intraoral imaging', 'angiography', 'procedural image', 'skull', 'patient photo', 
    'functional magnetic resonance', 'magnetic resonance', 'eye', 'mammography', 
    'electrocardiography', 'clinical imaging', 'skin lesion', 'ultrasound', 
    'specimen', 'computerized tomography', 'laryngoscopy', 'teeth', 
    'intraoperative image', 'surgical procedure', 'brain'
]


# 另外允许的 Microscopy：
MICRO_ALLOWED_SECONDARY = {'light microscopy'}

# 需要“均衡”的类别仅为上面的 24 个 Clinical Imaging 的二级标签：
# BALANCE_CATEGORIES = CLIN_SECONDARIES[:]  # 如要把 'light microscopy' 当第25类，只需 append 即可

BALANCE_CATEGORIES = CLIN_SECONDARIES + ['light microscopy']

# K 分布
K_DIST = {1: 0.90, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02}
K_LIST = [1,2,3,4,5,6]

# ---------------- Utils ----------------
def normalize_caption(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "no caption found":
        return None
    return s

def as_list_str(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [("" if v is None else str(v)) for v in x]
    return [str(x)]

def to_lower_set(seq):
    return {s.strip().lower() for s in seq if isinstance(s, str)}

def collect_context_list(rec_context, image_id):
    """兼容 dict{image_id:[...]} / list[{image_id:[...]}]，返回 list[str] 去重保序"""
    iid = str(image_id)
    out, seen = [], set()

    def _push(v):
        if v is None: v = ""
        v = str(v)
        if v not in seen:
            seen.add(v); out.append(v)

    if isinstance(rec_context, dict):
        for k, v in rec_context.items():
            if str(k) == iid:
                if isinstance(v, list):
                    for it in v: _push(it)
                else:
                    _push(v)
    elif isinstance(rec_context, list):
        for el in rec_context:
            if isinstance(el, dict):
                for k, v in el.items():
                    if str(k) == iid:
                        if isinstance(v, list):
                            for it in v: _push(it)
                        else:
                            _push(v)
    return out

def relpath_under(base_dir, abs_path):
    try:
        return os.path.relpath(abs_path, base_dir)
    except Exception:
        return abs_path

# ---------------- Worker: parse one tar ----------------
def process_tar(args):
    """
    解析一个 tar，返回 list[mini_rec]。
    mini_rec（每图一条）包含：
      - image_id
      - article_accession_id, article_title
      - image_file_name, image_panel_type, image_panel_subtype
      - image_primary_label(list), image_secondary_label(list), image_size
      - caption（该图自己的 caption）
      - context（该图自己的 context 列表）
      - image_wds_tar（相对 base 的 tar 路径）
      - image_wds_member（jpg 成员路径）
      - image_wds_path（拼接 "tar::member"）
    """
    tar_fp, base_dir = args
    out = []

    rel_tar = relpath_under(base_dir, tar_fp)
    try:
        with tarfile.open(tar_fp, "r") as tar:
            json_members = [m for m in tar.getmembers() if m.name.endswith(".json")]
            json_members.sort(key=lambda m: m.name)

            for jm in json_members:
                try:
                    raw = tar.extractfile(jm).read().decode("utf-8", "ignore")
                    rec = json.loads(raw)
                except Exception:
                    continue

                iid = rec.get("image_cluster_id")
                if not iid:
                    continue

                # caption（image_set 里找本图的）
                cap = None
                for imeta in (rec.get("image_set") or []):
                    if imeta and imeta.get("image_id") == iid:
                        cap = normalize_caption(imeta.get("caption"))
                        break

                ctx = collect_context_list(rec.get("image_context", {}), iid)

                jpg_member = os.path.splitext(jm.name)[0] + ".jpg"
                wds_member = jpg_member
                wds_tar = rel_tar
                wds_path = f"{wds_tar}::{wds_member}"

                primary = as_list_str(rec.get("image_primary_label"))
                secondary = as_list_str(rec.get("image_secondary_label"))

                mini = {
                    "image_id": iid,
                    "article_accession_id": rec.get("article_accession_id") or rec.get("pmid") or "",
                    "article_title": rec.get("article_title") or "",
                    "image_file_name": rec.get("image_file_name"),
                    "image_panel_type": rec.get("image_panel_type"),
                    "image_panel_subtype": rec.get("image_panel_subtype"),
                    "image_primary_label": primary,
                    "image_secondary_label": secondary,
                    "image_size": rec.get("image_size") or [None, None],
                    "caption": cap,
                    "context": ctx,
                    "image_wds_tar": wds_tar,
                    "image_wds_member": wds_member,
                    "image_wds_path": wds_path,
                }
                out.append(mini)
    except (tarfile.ReadError, EOFError):
        return []
    return out

# ---------------- Build index (parallel) ----------------
def build_mini_index_parallel(base_dir, workers):
    # 收集 tar
    tars = []
    for sub in ["commercial", "noncommercial", "other"]:
        d = os.path.join(base_dir, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".tar"):
                    tars.append(os.path.join(d, f))
    tars.sort()

    minis = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_tar, (t, base_dir)) for t in tars]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Indexing TARs (parallel)"):
            minis.extend(fut.result())

    # 两个索引：image_id -> mini； article -> list[mini]
    by_image = {m["image_id"]: m for m in minis}
    by_article = defaultdict(list)
    for m in minis:
        art = m["article_accession_id"]
        by_article[art].append(m)
    return by_image, by_article

# ---------------- Label filter ----------------
def is_allowed_image(mini):
    prim = to_lower_set(mini.get("image_primary_label") or [])
    sec  = to_lower_set(mini.get("image_secondary_label") or [])
    # Clinical Imaging + 24 任意其一
    if "clinical imaging" in prim and (sec & set(CLIN_SECONDARIES)):
        return True
    # Microscopy + light microscopy（允许，但它不参与 24 类均衡）
    if "microscopy" in prim and ("light microscopy" in sec):
        return True
    return False

def image_categories(mini):
    """返回该图属于哪些“均衡类别”（即 24 个 Clinical secondaries，取交集）。"""
    sec = to_lower_set(mini.get("image_secondary_label") or [])
    return list((sec & set(BALANCE_CATEGORIES)))

# ---------------- Discover groups (full-use) ----------------
def discover_groups_by_category(by_article, by_image, K_max=6):
    """
    在每篇文章内部，用“caption 完全相同”的图片作为一个组；只使用 is_allowed_image=True 的图片；
    组的类别 = 组内所有图片 secondary 的并集 ∩ BALANCE_CATEGORIES。
    返回:
      per_cat_groups[K][cat] = [ tuple(sorted_ids), ... ]
      全局可用统计 available_per_cat[K][cat] = 数量
    """
    per_cat_groups = {k: defaultdict(list) for k in K_LIST}
    # 遍历文章
    for art, items in by_article.items():
        # 先筛掉不允许的图；且必须有 caption
        allowed_items = [m for m in items if is_allowed_image(m) and m.get("caption")]
        if not allowed_items:
            continue

        # caption -> [image_id,...]
        cap2ids = defaultdict(list)
        for it in allowed_items:
            cap2ids[it["caption"]].append(it["image_id"])

        # 对每个 caption 组，形成 full-use 组（不截断）
        for cap, ids in cap2ids.items():
            K = len(ids)
            if K < 1 or K > K_max:
                continue
            ids_sorted = tuple(sorted(ids))
            # 组的类别：组内所有图的 secondary 之并集 ∩ 24 类
            cats = set()
            for iid in ids_sorted:
                cats.update(image_categories(by_image[iid]))
            if not cats:
                # 这组虽然允许（可能是 microscopy），但不属于 24 个均衡类；跳过
                continue
            for c in cats:
                per_cat_groups[K][c].append(ids_sorted)

    # 做个统计用
    available = {k: {c: len(vs) for c, vs in per_cat_groups[k].items()} for k in K_LIST}
    return per_cat_groups, available

# ---------------- Targets ----------------
def split_evenly_total(N, cats):
    """把 N 平均分到 len(cats) 个类别；余数给前面的若干个。"""
    q, r = divmod(N, len(cats))
    per = {c: q for c in cats}
    for c in cats[:r]:
        per[c] += 1
    return per

def compute_targets_per_cat(Nc, dist):
    """给单个类别 Nc，算其 K 配额（四舍五入后把误差补到 K=1）。"""
    t = {k: int(round(Nc * p)) for k, p in dist.items()}
    delta = Nc - sum(t.values())
    if delta != 0:
        t[1] = max(0, t.get(1, 0) + delta)
    return t

# ---------------- Sampling ----------------
def sample_by_category(per_cat_groups, per_cat_N, seed):
    """
    per_cat_groups: 来自 discover_groups_by_category 的 per_cat_groups[K][cat] -> [group]
    per_cat_N: 每个类别的目标条数（已是 N/24 均分）
    返回：selected_groups 汇总为 list[tuple(sorted_ids)]，以及按类/按K统计
    """
    rng = random.Random(seed)
    used_groups = set()  # 去重：一个组只能被一个类别拿走
    selected = defaultdict(lambda: defaultdict(list))  # selected[cat][K] = [group,...]

    for cat in BALANCE_CATEGORIES:
        Nc = per_cat_N.get(cat, 0)
        if Nc <= 0:
            continue
        targets_K = compute_targets_per_cat(Nc, K_DIST)

        for K in K_LIST:
            need = targets_K.get(K, 0)
            if need <= 0:
                continue
            cand = per_cat_groups.get(K, {}).get(cat, [])
            if not cand:
                continue
            # 去掉已用组
            cand = [g for g in cand if g not in used_groups]
            if not cand:
                continue
            if len(cand) <= need:
                pick = cand
            else:
                pick = rng.sample(cand, need)
            for g in pick:
                used_groups.add(g)
            selected[cat][K].extend(pick)

    # 展开
    all_groups = []
    for cat in BALANCE_CATEGORIES:
        for K in K_LIST:
            all_groups.extend(selected[cat][K])

    # 统计
    stat_cat = {cat: sum(len(selected[cat][K]) for K in K_LIST) for cat in BALANCE_CATEGORIES}
    stat_K = Counter({K: 0 for K in K_LIST})
    for cat in BALANCE_CATEGORIES:
        for K in K_LIST:
            stat_K[K] += len(selected[cat][K])

    return all_groups, selected, stat_cat, dict(stat_K)

# ---------------- Assemble ----------------
def assemble_records(groups, by_image):
    final = []
    for ids_sorted in groups:
        ids = list(ids_sorted)
        m0 = by_image[ids[0]]
        rec = {
            "image_file_name": [],
            "image_panel_type": [],
            "image_panel_subtype": [],
            "image_primary_label": [],
            "image_secondary_label": [],
            "image_size": [],
            "caption": [],
            "context": [],
            "image_id": [],
            "image_wds_tar": [],
            "image_wds_member": [],
            "image_wds_path": [],
            "article_accession_id": m0.get("article_accession_id"),
            "article_title": m0.get("article_title"),
        }
        cap0 = by_image[ids[0]].get("caption", "")
        for iid in ids:
            m = by_image[iid]
            rec["image_id"].append(iid)
            rec["image_file_name"].append(m.get("image_file_name"))
            rec["image_panel_type"].append(m.get("image_panel_type"))
            rec["image_panel_subtype"].append(m.get("image_panel_subtype"))
            rec["image_primary_label"].append(m.get("image_primary_label") or [])
            rec["image_secondary_label"].append(m.get("image_secondary_label") or [])
            rec["image_size"].append(m.get("image_size") or [None, None])
            rec["caption"].append(cap0)
            rec["context"].append(m.get("context") or [])
            rec["image_wds_tar"].append(m.get("image_wds_tar"))
            rec["image_wds_member"].append(m.get("image_wds_member"))
            rec["image_wds_path"].append(m.get("image_wds_path"))
        final.append(rec)
    return final

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Full-dataset parallel sampler with per-category balance & K-distribution.")
    ap.add_argument("--base", required=True, help="WebDataset 根目录（含 commercial/noncommercial/other）")
    ap.add_argument("--out", required=True, help="输出 JSON 文件路径")
    ap.add_argument("--total", type=int, default=50, help="目标样本总数 N（按 24 个二级类均分）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="并行进程数")
    args = ap.parse_args()

    # 1) 并行索引
    print("--- Phase 1: Building mini index in parallel ---")
    by_image, by_article = build_mini_index_parallel(args.base, args.workers)
    print(f"  images indexed: {len(by_image)}  |  articles: {len(by_article)}")

    # 2) 组发现（文章内、caption 完全相同、full-use、不截断），并按 24 类挂接候选
    print("\n--- Phase 2: Discovering full-use groups per category ---")
    per_cat_groups, available = discover_groups_by_category(by_article, by_image, K_max=6)
    print("  Available groups per K (per-category counts shown as total sums):")
    for K in K_LIST:
        totalK = sum(available.get(K, {}).values())
        print(f"    K={K}: total groups={totalK}")

    # 3) 24 类均分 N
    print("\n--- Phase 3: Targets ---")
    per_cat_N = split_evenly_total(args.total, BALANCE_CATEGORIES)
    print(f"  Per-category N (sum={sum(per_cat_N.values())}):")
    # 只打印前几项避免刷屏
    preview = list(per_cat_N.items())[:8]
    for k,v in preview:
        print(f"    {k}: {v}")
    if len(per_cat_N) > 8:
        print("    ...")

    # 4) 采样（同一组不可重复分配；某类某K不足 => 拿到多少算多少）
    print("\n--- Phase 4: Sampling by category & K ---")
    groups, selected, stat_cat, stat_K = sample_by_category(per_cat_groups, per_cat_N, seed=args.seed)
    print(f"  Selected groups total: {len(groups)}")
    print("  Selected per K:", stat_K)
    print("  Selected per category (top few):")
    for c, n in list(stat_cat.items())[:8]:
        print(f"    {c}: {n}")
    if len(stat_cat) > 8:
        print("    ...")

    # 5) 组装并保存
    print("\n--- Phase 5: Assembling final JSON records ---")
    final = assemble_records(groups, by_image)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print("\n======================================================")
    print("🎉 Done")
    print(f"- Saved to: {os.path.abspath(args.out)}")
    print(f"- Records:  {len(final)}   (target N={args.total}, 某些类/K 不足则会少)")
    print("======================================================")

if __name__ == "__main__":
    main()
