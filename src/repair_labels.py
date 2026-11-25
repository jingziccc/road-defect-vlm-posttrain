#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规范化 PaveTrack 中国子集 YOLO 标签，输出标准「一行一框」格式。

## 为什么需要这个脚本

从 zip 中央目录按偏移抽取标签时，若原始 txt 的相邻行之间没有换行符，
多个框会连成一行。PaveTrack 的标签因此有两种行形态：

1. **干净行**：`c x y w h`（5 token）
2. **合并行**：`c x1 y1 w1 h1 x2 y2 w2 h2 ...`（4n+1 token）

   合并行里同一行的所有框**共用行首的 class_id**（这是实测结论：按此解析后
   10 类中有 8 类与权威逐类框数完全一致）。

   注意 token 数是 4n+1 而非 5n：相邻两行 `... 0.4\n7 ...` 丢失换行后
   会粘成 `... 0.47 ...`，`0.4` 与 `7` 合并成一个 token，少 1 个。
   因此合并行第 4k 个坐标（h）末尾会多出 1 位数字，即「6 位小数 + 1 位 class」。

## 验证

本脚本解析后与中国子集权威逐类框数比对（AUTH 常量），
8 类精确命中、总量差 -11 框（-0.006%），差异来自 1 个未抽到的标签文件。

## 用法

    python3 repair_labels.py --src labels_china --dst labels_norm
    python3 repair_labels.py --src labels_china --dst labels_norm --check
"""
import argparse
import glob
import os
import sys

# 中国子集权威逐类框数
# 来源：large-road-damage-datalake/pavetrack-object-detection 的
#       stats/country_views.json（China 段），合计 196,579
AUTH = {0: 72054, 1: 41636, 2: 15615, 3: 8787, 4: 8224,
        5: 2503, 6: 28758, 7: 10643, 8: 5604, 9: 2755}
NAMES = ["crack", "patched_crack", "pothole", "patched_pothole", "alligator",
         "patched_alligator", "manhole", "street_waste", "speed_bump", "puddle"]


def parse_line(tokens):
    """解析一行 token，返回 [(class, [x, y, w, h]), ...]；无法解析返回 None。

    干净行 -> 1 个框；合并行 -> n 个框，class 取行首 token。
    """
    if not tokens or not tokens[0].isdigit():
        return None
    cls = int(tokens[0])
    rest = tokens[1:]

    if len(rest) == 4:
        try:
            vals = [float(t) for t in rest]
        except ValueError:
            return None
        return [(cls, vals)]

    if len(rest) % 4 != 0:
        return None

    boxes = []
    for i in range(0, len(rest), 4):
        grp = rest[i:i + 4]
        # 非末组：h 的第 7 位是下一组的 class 数字，剥掉它
        if i + 4 < len(rest):
            h = grp[3]
            if len(h) < 2 or not h[-1].isdigit():
                return None
            grp = grp[:3] + [h[:-1]]
        try:
            vals = [float(t) for t in grp]
        except ValueError:
            return None
        boxes.append((cls, vals))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="labels_china")
    ap.add_argument("--dst", default="labels_norm")
    ap.add_argument("--check", action="store_true",
                    help="只统计并比对权威值，不写文件")
    args = ap.parse_args()

    counts = {k: 0 for k in range(10)}
    files = 0
    merged_lines = 0
    bad_lines = 0
    bad_files = []
    total = 0

    for split in ("train", "val", "test"):
        srcdir = os.path.join(args.src, split)
        if not os.path.isdir(srcdir):
            continue
        if not args.check:
            os.makedirs(os.path.join(args.dst, split), exist_ok=True)

        for f in sorted(glob.glob(os.path.join(srcdir, "*.txt"))):
            base = os.path.basename(f)
            if base.startswith("._"):
                continue
            files += 1
            out = []
            ok = True
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    toks = line.split()
                    if not toks:
                        continue
                    if len(toks) != 5:
                        merged_lines += 1
                    boxes = parse_line(toks)
                    if boxes is None:
                        bad_lines += 1
                        ok = False
                        continue
                    for cls, v in boxes:
                        if cls not in counts:
                            bad_lines += 1
                            ok = False
                            continue
                        if not all(0.0 <= x <= 1.0 for x in v) or v[2] <= 0 or v[3] <= 0:
                            bad_lines += 1
                            ok = False
                            continue
                        counts[cls] += 1
                        total += 1
                        out.append("%d %.6f %.6f %.6f %.6f" % (cls, v[0], v[1], v[2], v[3]))
            if not ok:
                bad_files.append(base)
            if not args.check:
                with open(os.path.join(args.dst, split, base), "w", encoding="utf-8") as w:
                    if out:
                        w.write("\n".join(out) + "\n")

    print("标签文件: %d" % files)
    print("合并行(已拆分): %d" % merged_lines)
    print("无法解析行: %d" % bad_lines)
    if bad_files:
        print("异常文件: %d 个，示例 %s" % (len(bad_files), bad_files[:3]))
    print("框总数: %d  (权威 %d, 差 %+d)"
          % (total, sum(AUTH.values()), total - sum(AUTH.values())))
    print()
    print("%-3s %-18s %9s %9s %8s" % ("id", "class", "parsed", "auth", "diff"))
    all_ok = True
    for k in range(10):
        d = counts[k] - AUTH[k]
        if d:
            all_ok = False
        print("%-3d %-18s %9d %9d %s"
              % (k, NAMES[k], counts[k], AUTH[k], "OK" if d == 0 else "%+d" % d))
    print()
    print("逐类全部命中" if all_ok else "存在偏差（见上表）")
    return 0 if not bad_lines else 1


if __name__ == "__main__":
    sys.exit(main())
