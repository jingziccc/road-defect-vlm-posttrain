"""诊断路段切分：对比多种 road_id 解析口径下的切分结果，定位 bug 真正成因。

用法:
    python3 src/analyze_split.py --index index/pt_index_v2.json
"""
import argparse
import collections
import json
import re

# 路段式文件名: 5~6 位数字路段号 + 日期时间，如 000001-2021-09-07_10-11-27-711.jpg
ROAD_RE = re.compile(r"^(\d{5,6})-\d{4}-\d{2}-\d{2}_")
# 时间戳式: 超长数字前缀 + 毫秒时间戳，如 1617949663266913869455049206564-1617949272040.jpg
TS_RE = re.compile(r"^(\d{13,})-\d{9,}")
# 纯数字名，如 168368595491.jpg
NUM_RE = re.compile(r"^(\d+)\.jpg$")


def classify(stem):
    """返回 (类别, road_id 或 None)。None 表示该图没有路段归属，应视为独立样本。"""
    name = stem + ".jpg"
    m = ROAD_RE.match(name)
    if m:
        return "road", m.group(1)
    m = TS_RE.match(name)
    if m:
        return "ts", None
    m = NUM_RE.match(name)
    if m:
        return "num", None
    return "other", None


def split_stats(pairs, keyfn, label):
    """按 keyfn 给出的分组键做 conflict->test 切分，返回各 split 图数。"""
    ks = collections.defaultdict(set)
    for p in pairs:
        k = keyfn(p)
        if k is not None:
            ks[k].add(p["split"])
    conflict = {k for k, s in ks.items() if len(s) > 1}

    buckets = collections.Counter()
    for p in pairs:
        k = keyfn(p)
        if k is not None and k in conflict:
            buckets["test"] += 1
        else:
            buckets[p["split"]] += 1
    tot = sum(buckets.values())
    print("  %-28s keys=%-5d conflict=%-4d | train %6d (%4.1f%%) val %5d test %6d"
          % (label, len(ks), len(conflict),
             buckets["train"], 100.0 * buckets["train"] / tot,
             buckets["val"], buckets["test"]))
    return buckets, conflict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index/pt_index_v2.json")
    args = ap.parse_args()

    pairs = json.load(open(args.index))["pairs"]
    # 兼容 v2 索引：官方 split 字段名为 official_split
    for p in pairs:
        if "split" not in p and "official_split" in p:
            p["split"] = p["official_split"]
    print("pairs = %d\n" % len(pairs))

    # ---- 1. 文件名形态分布 ----
    kinds = collections.Counter()
    for p in pairs:
        k, _ = classify(p["stem"])
        kinds[k] += 1
    print("=== 文件名形态 ===")
    for k, v in kinds.most_common():
        print("  %-6s %6d (%.1f%%)" % (k, v, 100.0 * v / len(pairs)))

    # ---- 2. 各种 road_id 口径下的切分 ----
    print("\n=== 切分结果对比 ===")

    def old_key(p):
        # 现有实现: ^(\d+)- ，失败则 split('-')[0]
        m = re.match(r"^(\d+)-", p["stem"] + ".jpg")
        return m.group(1) if m else (p["stem"] + ".jpg").split("-")[0]

    def new_key(p):
        # 修正实现: 只有路段式文件名才有路段归属
        k, rid = classify(p["stem"])
        return rid

    def new_key_plus_ts(p):
        # 修正 + 时间戳式也按前缀分组（仅用于对比，不推荐）
        k, rid = classify(p["stem"])
        if rid is not None:
            return rid
        if k == "ts":
            return TS_RE.match(p["stem"] + ".jpg").group(1)
        return None

    old_b, old_c = split_stats(pairs, old_key, "现有 ^(\\d+)- (有 bug)")
    new_b, new_c = split_stats(pairs, new_key, "修正: 仅路段式")
    ts_b, ts_c = split_stats(pairs, new_key_plus_ts, "修正+时间戳也分组")

    # ---- 3. 现有实现把哪些"伪路段"卷进 conflict ----
    print("\n=== 现有实现的 conflict 键构成 ===")
    cnt = collections.Counter()
    for k in old_c:
        if ROAD_RE.match(k + "-2021-01-01_00-00-00.jpg"):
            cnt["真实路段(5-6位)"] += 1
        elif len(k) >= 13:
            cnt["时间戳(长数字)"] += 1
        else:
            cnt["其他"] += 1
    for k, v in cnt.most_common():
        print("  %-18s %d" % (k, v))

    # ---- 4. 真实路段的跨 split 情况 ----
    print("\n=== 真实路段跨 split 明细 ===")
    road_split = collections.defaultdict(set)
    road_imgs = collections.Counter()
    for p in pairs:
        k, rid = classify(p["stem"])
        if rid is not None:
            road_split[rid].add(p["split"])
            road_imgs[rid] += 1
    conf = [r for r, s in road_split.items() if len(s) > 1]
    print("  真实路段数: %d" % len(road_split))
    print("  跨 split 路段: %d (%.1f%%)" % (len(conf), 100.0 * len(conf) / len(road_split)))
    print("  这些路段含图: %d (%.1f%%)"
          % (sum(road_imgs[r] for r in conf),
             100.0 * sum(road_imgs[r] for r in conf) / len(pairs)))
    print("  非路段式图数: %d (%.1f%%)"
          % (len(pairs) - sum(road_imgs.values()),
             100.0 * (len(pairs) - sum(road_imgs.values())) / len(pairs)))

    print("\n  跨 split 路段 Top10 (按图数):")
    for r in sorted(conf, key=lambda x: -road_imgs[x])[:10]:
        print("    %s: %d 图, splits=%s" % (r, road_imgs[r], sorted(road_split[r])))


if __name__ == "__main__":
    main()
