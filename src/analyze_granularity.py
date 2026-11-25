"""比较不同分组粒度下的防泄漏切分可行性。

现有 bug 的根因不只是文件名解析错误：真实路段只有 84 个，其中 76 个跨官方 split。
在这么粗的粒度上做「冲突路段整体归 test」，无论怎么解析文件名都会把 70% 数据推进 test。
本脚本测试更细的会话粒度（road+date / 完整时间戳 / 哈希去重）能否得到可用切分。

用法:
    python3 src/analyze_granularity.py --index index/pt_index_v2.json
"""
import argparse
import collections
import json
import re

ROAD_RE = re.compile(r"^(\d{5,6})-(\d{4})-(\d{2})-(\d{2})_")
TS_RE = re.compile(r"^(\d{13,})-(\d{9,})")


def keys_for(stem):
    """返回该图在多种粒度下的分组键；None 表示无归属。"""
    name = stem + ".jpg"
    out = {}
    m = ROAD_RE.match(name)
    if m:
        out["road"] = m.group(1)
        out["road_date"] = "%s-%s%s%s" % (m.group(1), m.group(2), m.group(3), m.group(4))
        out["full"] = stem            # 每帧独立
        return out
    m = TS_RE.match(name)
    if m:
        out["ts_prefix"] = m.group(1)
        out["full"] = stem
        return out
    out["full"] = stem
    return out


def evaluate(pairs, granularity, policy):
    """policy: 'conflict2test' 或 'majority' 或 'drop'"""
    ks = collections.defaultdict(collections.Counter)
    for p in pairs:
        k = keys_for(p["stem"]).get(granularity)
        if k is not None:
            ks[k][p["split"]] += 1

    buckets = collections.Counter()
    n_conflict = 0
    for p in pairs:
        k = keys_for(p["stem"]).get(granularity)
        if k is None:
            buckets[p["split"]] += 1
            continue
        cnt = ks[k]
        if len(cnt) > 1:
            n_conflict += 1 if p is pairs[0] else 0
            if policy == "conflict2test":
                buckets["test"] += 1
            elif policy == "majority":
                buckets[cnt.most_common(1)[0][0]] += 1
            elif policy == "drop":
                pass
        else:
            buckets[next(iter(cnt))] += 1
    n_conf = sum(1 for k, c in ks.items() if len(c) > 1)
    tot = sum(buckets.values()) or 1
    return {
        "gran": granularity, "policy": policy,
        "keys": len(ks), "conflict": n_conf,
        "train": buckets["train"], "val": buckets["val"], "test": buckets["test"],
        "train_pct": 100.0 * buckets["train"] / tot,
        "total": tot,
    }


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

    print("=== 各粒度 x 各策略 的切分结果 ===")
    print("%-14s %-14s %6s %8s %8s %7s %8s %7s" %
          ("granularity", "policy", "keys", "conflict", "train", "val", "test", "train%"))
    for gran in ("road", "road_date", "ts_prefix", "full"):
        for policy in ("conflict2test", "majority", "drop"):
            r = evaluate(pairs, gran, policy)
            print("%-14s %-14s %6d %8d %8d %7d %8d %6.1f%%" %
                  (r["gran"], r["policy"], r["keys"], r["conflict"],
                   r["train"], r["val"], r["test"], r["train_pct"]))
        print()

    # road_date 粒度下跨 split 的比例
    print("=== road_date 粒度跨 split 明细 ===")
    ks = collections.defaultdict(collections.Counter)
    imgs = collections.Counter()
    for p in pairs:
        k = keys_for(p["stem"]).get("road_date")
        if k:
            ks[k][p["split"]] += 1
            imgs[k] += 1
    conf = [k for k, c in ks.items() if len(c) > 1]
    print("  会话数: %d" % len(ks))
    print("  跨 split: %d (%.1f%%)" % (len(conf), 100.0 * len(conf) / len(ks)))
    print("  跨 split 会话含图: %d (%.1f%%)" %
          (sum(imgs[k] for k in conf), 100.0 * sum(imgs[k] for k in conf) / len(pairs)))
    print("  会话图数分布: 中位 %d, 最大 %d" %
          (sorted(imgs.values())[len(imgs) // 2], max(imgs.values())))


if __name__ == "__main__":
    main()
