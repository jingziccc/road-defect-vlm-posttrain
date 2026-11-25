"""统一评测：Top-2 Margin 路由 + 双视角 CoT 复核 + 误改率审计。

指标口径（与简历/项目文档一致）:
- 分类层: Macro-F1、每类 P/R/F1
- 路由层: 复核覆盖率、难例命中率(Margin 组 vs 随机组)
- 纠错层: 净修正数 = 错->对 - 对->错; 误改率 = 对->错 / 复核数
- 成本层: 强模型调用占比、单图平均耗时

用法:
    python src/evaluate.py --pred out/stage1_test.jsonl --route out/route_test.jsonl \
        --refine out/stage2_test.jsonl --out out/metrics_test.json
"""
import argparse
import collections
import json


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prf(gold, pred, n_class):
    """返回每类 P/R/F1 与 Macro-F1。"""
    out = {}
    for c in range(n_class):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"P": prec, "R": rec, "F1": f1, "support": tp + fn}
    macro = sum(v["F1"] for v in out.values()) / n_class
    return out, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="一阶段预测 jsonl: {crop,label,pred,p1,p2}")
    ap.add_argument("--route", default=None, help="路由结果: {crop,routed(bool),margin}")
    ap.add_argument("--refine", default=None, help="二阶段复核: {crop,pred}")
    ap.add_argument("--n-class", type=int, default=10)
    ap.add_argument("--out", default="out/metrics.json")
    args = ap.parse_args()

    base = load(args.pred)
    gold = [r["label"] for r in base]
    pred1 = [r["pred"] for r in base]
    per_cls, macro1 = prf(gold, pred1, args.n_class)

    res = {"n": len(base), "stage1": {"per_class": per_cls, "macro_f1": macro1}}

    # 路由层
    routed = set()
    if args.route:
        rr = load(args.route)
        routed = {r["crop"] for r in rr if r.get("routed")}
        margins = [r.get("margin") for r in rr if r.get("margin") is not None]
        res["route"] = {
            "coverage": len(routed) / len(base) if base else 0.0,
            "n_routed": len(routed),
            "margin_mean": sum(margins) / len(margins) if margins else None,
        }

    # 纠错层
    if args.refine:
        ref = {r["crop"]: r["pred"] for r in load(args.refine)}
        final = list(pred1)
        fix = broke = 0
        for i, r in enumerate(base):
            c = r["crop"]
            if c in ref and c in routed:
                new = ref[c]
                if pred1[i] != gold[i] and new == gold[i]:
                    fix += 1
                elif pred1[i] == gold[i] and new != gold[i]:
                    broke += 1
                final[i] = new
        per_cls2, macro2 = prf(gold, final, args.n_class)
        n_routed = len(routed) if routed else 0
        res["stage2"] = {
            "per_class": per_cls2,
            "macro_f1": macro2,
            "macro_f1_delta": macro2 - macro1,
            "net_fix": fix - broke,
            "fix": fix,
            "broke": broke,
            "mis_fix_rate": (broke / n_routed) if n_routed else None,
        }

    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "stage1"}, ensure_ascii=False, indent=2)[:2000])
    print("stage1 macro_f1 = %.4f" % macro1)
    if "stage2" in res:
        print("stage2 macro_f1 = %.4f (delta %+.4f)" % (res["stage2"]["macro_f1"], res["stage2"]["macro_f1_delta"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
