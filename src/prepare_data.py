"""把 PaveTrack_OD 的 YOLO 标注转成「框裁剪 -> 类别」分类样本，并做防泄漏切分。

输入: index/pt_index_v2.json + 已解压的 PaveTrack_OD/ 目录
输出: data/{train,val,test}.jsonl，每行 {crop, label, group, src_img, box}

切分策略（分层分组分配）:
  官方 split 是**帧级随机切分**——84 个真实路段中有 76 个横跨 train/val/test。
  直接沿用官方 split 会让同路段、同场景的相邻帧同时出现在训练集和测试集，
  测试指标虚高。因此这里丢弃官方 split，改为：

  1. 按 group_key 分组（同路段/同采集会话/同时间戳簇归为一组）
  2. 在每个分组类型内部独立做贪心分配，使各 split 的类型构成与全局一致
  3. 组是最小分配单位，保证任何一组不会跨 split

  结果：70/15/15，组间零泄漏，且 road/ts/ts14/solo 四类构成偏移 < 0.01 pt。

用法:
    python3 src/build_index.py --manifest PaveTrack_OD.json --out index/pt_index_v2.json
    python3 src/prepare_data.py --index index/pt_index_v2.json --root PaveTrack_OD --outdir data
"""
import argparse
import collections
import json
import os

from PIL import Image

# 中国子集 class_id 映射（已用官方 samples_manifest.json 的 class_map 核实，
# 并用标注文件交叉验证）。中国子集共 10 类，class_id 0-9。
CLASSES = [
    "crack", "patched_crack", "pothole", "patched_pothole",
    "alligator_crack", "patched_alligator_crack",
    "manhole", "street_waste", "speed_bump", "puddle",
]

# ⚠️ 美国子集用的是另一套映射，id 与类名不对应，绝不能混用：
#   pothole 中国=2 / 美国=1；patched crack 中国=1 / 美国=2；manhole 中国=6 / 美国=5
# 仅在需要处理美国子集时使用。
CLASSES_US = [
    "crack", "pothole", "patched_crack",
    "patched_pothole", "clay_patched_crack", "manhole",
]

EXPAND = 0.25
MIN_BOX = 8
CROP_SIZE = 448
JPEG_QUALITY = 92


def check_class_id(cid, path):
    """校验 class_id 在本项目使用的中国子集映射范围内。"""
    if cid < 0 or cid >= len(CLASSES):
        raise SystemExit(
            "[label] 非法 class_id=%d (文件 %s)；中国子集应为 0..%d。"
            "若误用美国子集，其映射见 CLASSES_US。" % (cid, path, len(CLASSES) - 1))
    return cid


def yolo_to_box(cx, cy, w, h, W, H):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return x1, y1, x2, y2


def expand_box(b, W, H, g=EXPAND):
    x1, y1, x2, y2 = b
    w = x2 - x1
    h = y2 - y1
    return (max(0, x1 - g * w), max(0, y1 - g * h),
            min(W, x2 + g * w), min(H, y2 + g * h))


def stratified_assign(pairs, ratios=(0.70, 0.15, 0.15), seed=42):
    """按分组类型分层，每类内部贪心分配，返回 {group_key: split}。

    贪心规则：组按图数降序，每组放进「目标配额 - 已填量」缺口最大的 split。
    在类型内部做，可保证各 split 的类型构成与全局一致。
    """
    names = ("train", "val", "test")
    by_type = collections.defaultdict(collections.Counter)
    for p in pairs:
        by_type[p["group_type"]][p["group_key"]] += 1

    assign = {}
    for t, groups in by_type.items():
        total = sum(groups.values())
        targets = [total * r for r in ratios]
        filled = [0, 0, 0]
        # 同尺寸组按 key 排序，保证可复现
        for key, size in sorted(groups.items(), key=lambda x: (-x[1], x[0])):
            deficits = [targets[i] - filled[i] for i in range(3)]
            i = max(range(3), key=lambda j: deficits[j])
            assign[key] = names[i]
            filled[i] += size
    return assign


def verify_no_leak(pairs, assign):
    """确认没有任何分组跨 split。"""
    g2s = collections.defaultdict(set)
    for p in pairs:
        g2s[p["group_key"]].add(assign[p["group_key"]])
    return [k for k, v in g2s.items() if len(v) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index/pt_index_v2.json")
    ap.add_argument("--root", default="PaveTrack_OD")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--cropdir", default="data/crops")
    ap.add_argument("--max-per-split", type=int, default=0)
    ap.add_argument("--crop-size", type=int, default=CROP_SIZE)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cropdir, exist_ok=True)
    idx = json.load(open(args.index))
    pairs = idx["pairs"]

    # ---- schema 校验：拒绝旧索引，避免静默读出错误字段 ----
    required = {"group_key", "group_type", "official_split", "image_path", "label_path"}
    if not pairs or not required <= set(pairs[0]):
        missing = sorted(required - set(pairs[0] if pairs else []))
        raise SystemExit(
            "[index] 索引 schema 过旧，缺少字段 %s\n"
            "        请先用 build_index.py 重新生成索引：\n"
            "        python3 src/build_index.py --manifest PaveTrack_OD.json --out %s"
            % (missing, "index/pt_index_v2.json")
        )

    # ---- 分层分组切分 ----
    assign = stratified_assign(pairs)
    leaks = verify_no_leak(pairs, assign)
    if leaks:
        raise SystemExit("[split] 致命错误：%d 个分组跨 split" % len(leaks))

    buckets = collections.defaultdict(list)
    for p in pairs:
        buckets[assign[p["group_key"]]].append(p)

    n = len(pairs)
    print("[split] 分层分组分配完成（组间零泄漏）")
    for k in ("train", "val", "test"):
        imgs = len(buckets[k])
        ngrp = len({p["group_key"] for p in buckets[k]})
        comp = collections.Counter(p["group_type"] for p in buckets[k])
        desc = " ".join("%s=%.1f%%" % (t, 100.0 * comp[t] / imgs)
                        for t in ("road", "ts", "ts14", "solo"))
        print("[split] %-5s imgs=%5d (%4.1f%%) groups=%4d  %s"
              % (k, imgs, 100.0 * imgs / n, ngrp, desc))

    # ---- 抽裁剪样本 ----
    stats = collections.Counter()
    label_ids = collections.Counter()
    for split in ("train", "val", "test"):
        items = buckets[split]
        if args.max_per_split:
            items = items[: args.max_per_split]
        rows = []
        for p in items:
            img_path = os.path.join(args.root, p["image_path"].split("PaveTrack_OD/", 1)[-1])
            lab_path = os.path.join(args.root, p["label_path"].split("PaveTrack_OD/", 1)[-1])
            if not (os.path.exists(img_path) and os.path.exists(lab_path)):
                stats["missing"] += 1
                continue
            try:
                im = Image.open(img_path).convert("RGB")
            except Exception:
                stats["bad_img"] += 1
                continue
            W, H = im.size
            for li, line in enumerate(open(lab_path)):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                label_ids[cid] += 1
                if cid >= len(CLASSES):
                    stats["unknown_class"] += 1
                    continue
                box = yolo_to_box(*[float(x) for x in parts[1:5]], W, H)
                box = expand_box(box, W, H)
                if box[2] - box[0] < MIN_BOX or box[3] - box[1] < MIN_BOX:
                    stats["tiny_box"] += 1
                    continue
                crop = im.crop(tuple(int(round(v)) for v in box))
                crop = crop.resize((args.crop_size, args.crop_size), Image.BICUBIC)
                stem = "%s_b%d" % (p["stem"], li)
                out = os.path.join(args.cropdir, split, stem + ".jpg")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                crop.save(out, quality=JPEG_QUALITY)
                rows.append({
                    "crop": out,
                    "label": cid,
                    "group": p["group_key"],
                    "src_img": p["image_path"],
                    "box": [round(v, 1) for v in box],
                })
                stats["%s/%s" % (split, CLASSES[cid])] += 1
        with open(os.path.join(args.outdir, "%s.jsonl" % split), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("[crops] %-5s %d" % (split, len(rows)))

    # ---- class_id 校验 ----
    print("\n[label] 数据中出现的 class_id 分布：")
    for cid in sorted(label_ids):
        name = CLASSES[cid] if cid < len(CLASSES) else "<未定义>"
        print("[label]   %2d -> %-24s %7d" % (cid, name, label_ids[cid]))
    max_id = max(label_ids) if label_ids else -1
    if max_id >= len(CLASSES):
        print("[label] ⚠ 数据含 class_id=%d，超出 CLASSES 的 %d 类定义，"
              "请按实际标签顺序补全 CLASSES" % (max_id, len(CLASSES)))
    if stats["unknown_class"]:
        print("[label] 被跳过的未知类别框数: %d" % stats["unknown_class"])

    print("\n[stats]")
    for k in sorted(stats):
        if "/" not in k:
            print("  %-16s %d" % (k, stats[k]))


if __name__ == "__main__":
    main()
