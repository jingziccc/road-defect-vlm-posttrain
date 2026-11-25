"""从 PaveTrack_OD 清单 JSON 构建索引：图像/标签配对 + 分组键提取。

用法:
    python3 src/build_index.py --manifest PaveTrack_OD.json --out index/pt_index_v2.json

分组键（group_key）用于防泄漏切分：同组的图必然进同一个 split。
PaveTrack 的文件名有 4 种形态，各自的分组粒度不同：

| 形态 | 样例 | 分组键 | 组数 | 图数 |
|---|---|---|---|---|
| 路段式 | 000001-2021-09-07_10-11-27-711.jpg | road:000001 | 84 | 34,662 |
| 时间戳式 | 1617949663266913869455049206564-1617949272040.jpg | ts:<长前缀> | 768 | 6,542 |
| 14位时间戳 | 20151209152420JSaP5pf8istAOPpcx4IYwJGvpDeUS1OT.jpg | ts14:20151209152420 | 471 | 528 |
| 独立图 | 168368595491.jpg / <32位HEX>.jpg | solo:<stem> | 8,268 | 8,268 |

注意：不能只用 `^(\d+)-` 提取路段号。时间戳式文件名的前缀是设备生成的超长 id，
不是路段号，误当作路段会产生 2,266 个「伪路段」，其中 440 个是时间戳型。
"""
import argparse
import collections
import json
import re

CLASSES = [
    "crack", "patched_crack", "pothole", "patched_pothole",
    "alligator_crack", "patched_alligator_crack",
    "manhole", "street_waste", "speed_bump", "puddle",
    "clay_patched_crack",          # 第 11 类，仅 245 图 / 328 框
]

# 路段式: 5~6 位路段号 + 日期时间
ROAD_RE = re.compile(r"^(\d{5,6})-\d{4}-\d{2}-\d{2}_")
# 时间戳式: 13 位以上长数字 + 9 位以上毫秒时间戳
TS_RE = re.compile(r"^(\d{13,})-\d{9,}")
# 14 位时间戳开头（精确到秒）
TS14_RE = re.compile(r"^(\d{14})")


def walk(node, path=()):
    p = path + (node.get("name"),)
    yield node, p
    for c in (node.get("children") or []):
        yield from walk(c, p)


def group_key(fname):
    """返回 (分组键, 分组类型)。同键的图必须进同一个 split，否则同路段/同场景泄漏。"""
    m = ROAD_RE.match(fname)
    if m:
        return "road:" + m.group(1), "road"
    m = TS_RE.match(fname)
    if m:
        return "ts:" + m.group(1), "ts"
    m = TS14_RE.match(fname)
    if m:
        return "ts14:" + m.group(1), "ts14"
    return "solo:" + fname.rsplit(".", 1)[0], "solo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="PaveTrack_OD.json")
    ap.add_argument("--out", default="index/pt_index_v2.json")
    args = ap.parse_args()

    d = json.load(open(args.manifest))
    root = d["children"][0]

    imgs, labs = {}, {}
    for n, p in walk(root):
        if n.get("type") != "file":
            continue
        name = n["name"]
        parent = "/".join(p[:-1])
        if name.endswith(".jpg"):
            imgs[(parent, name[:-4])] = {
                "name": name, "path": n["path"], "size": n.get("size", 0),
                "split": p[3] if len(p) > 3 else "",
            }
        elif name.endswith(".txt"):
            labs[(parent, name[:-4])] = {
                "name": name, "path": n["path"], "size": n.get("size", 0),
            }

    pairs = []
    for (iparent, stem), im in imgs.items():
        lparent = iparent.replace("/images/", "/labels/")
        lb = labs.get((lparent, stem))
        if lb is None:
            continue
        gk, gt = group_key(im["name"])
        pairs.append({
            "stem": stem,
            "official_split": im["split"],   # 保留官方 split 以便审计，不直接用于训练
            "group_key": gk,
            "group_type": gt,
            "image_path": im["path"],
            "label_path": lb["path"],
            "image_size": im["size"],
            "label_size": lb["size"],
        })

    # ---- 审计输出 ----
    gtypes = collections.Counter(p["group_type"] for p in pairs)
    gcount = collections.defaultdict(set)
    for p in pairs:
        gcount[p["group_type"]].add(p["group_key"])

    print("[index] pairs=%d" % len(pairs))
    print("[index] %-8s %8s %8s" % ("type", "groups", "images"))
    for t in ("road", "ts", "ts14", "solo"):
        print("[index] %-8s %8d %8d" % (t, len(gcount[t]), gtypes[t]))

    # 官方 split 在组粒度上的跨 split 情况（仅审计，不影响训练切分）
    gs = collections.defaultdict(set)
    for p in pairs:
        gs[p["group_key"]].add(p["official_split"])
    n_conf = sum(1 for v in gs.values() if len(v) > 1)
    print("[index] 官方 split 跨组冲突: %d / %d 组 (%.1f%%)"
          % (n_conf, len(gs), 100.0 * n_conf / len(gs)))
    print("[index] → 官方 split 是帧级随机切分，不可直接用于防泄漏训练，"
          "改由 prepare_data.py 按 group_key 重新分层分配")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({
        "pairs": pairs,
        "classes": CLASSES,
        "note": "official_split 仅用于审计；训练切分见 prepare_data.py 的分层分配",
    }, open(args.out, "w"), ensure_ascii=False)
    print("[index] 写出 %s" % args.out)


if __name__ == "__main__":
    import os
    main()
