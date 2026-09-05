# -*- coding: utf-8 -*-
"""
10_分段修正（Top-L 由 k-core 保护，整体排序由 RRF 决定）
========================================================
为什么这样做：
  - XGBoost(回归)：整体 Spearman 最好；
  - k-core：前 100 名最稳（Facebook 尤甚）；
  - RRF：介于两者之间。
  因此：
    1) 先用 RRF 给出全体节点的整体顺序；
    2) 单独构造“前 L 名”名单：k-core 前 L 与 RRF 前 L 的交集优先，
       再补 k-core 独有、最后补 RRF 独有；
    3) 其余节点按 RRF 顺序接在后面。
  这样可以保住整体排序，同时让最关键的 top-L 不再被 XGBoost 的
  个别错误顶部节点拖垮。

运行：python 10_topL_corrected.py
输出：results/leave_one_out_topL_detail.csv
      results/leave_one_out_topL_summary.csv
"""

import os
import sys
import importlib.util
import csv
import statistics
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_local_lib = os.path.join(ROOT, "work", "figlibs")
if os.path.isdir(_local_lib):
    sys.path.append(_local_lib)

_spec = importlib.util.spec_from_file_location(
    "m06", os.path.join(ROOT, "outputs", "code", "06_leave_one_out_xgboost.py")
)
m06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m06)

import xgboost as xgb


def rrf_scores(score_lists, k=60):
    n = len(score_lists[0])
    total = [0.0] * n
    for scores in score_lists:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        rank = [0] * n
        for pos, i in enumerate(order, start=1):
            rank[i] = pos
        for i in range(n):
            total[i] += 1.0 / (k + rank[i])
    return total


def topL_corrected_order(rrf_vals, kcore_vals, L):
    """返回一个最终排列（下标列表）。"""
    n = len(rrf_vals)
    rrf_order = sorted(range(n), key=lambda i: rrf_vals[i], reverse=True)
    kc_order = sorted(range(n), key=lambda i: kcore_vals[i], reverse=True)

    top_rrf = rrf_order[:L]
    top_kc = kc_order[:L]
    inter = [i for i in top_rrf if i in set(top_kc)]
    only_kc = [i for i in top_kc if i not in set(top_rrf)]
    only_rrf = [i for i in top_rrf if i not in set(top_kc)]

    top_segment = inter + only_kc + only_rrf
    rest = [i for i in rrf_order if i not in top_segment]
    return top_segment + rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--L", type=int, default=100,
                        help="由 k-core 保护的前 L 名")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    print("=" * 76)
    print(f"分段修正 Top-L（L={args.L}），留一网络交叉验证")
    print("=" * 76)

    data = {net: m06.prepare_network(net) for net in m06.NETWORKS}
    detail = []

    for test_net in m06.NETWORKS:
        train_nets = [n for n in m06.NETWORKS if n != test_net]
        Xtr, ytr = [], []
        for net in train_nets:
            Xtr.extend(data[net]["X"])
            ytr.extend(data[net]["y_rank"])

        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=args.seed, n_jobs=-1,
        )
        model.fit(Xtr, ytr)

        test = data[test_net]
        pred_xgb = list(model.predict(test["X"]))
        rrf = rrf_scores([pred_xgb, test["baselines"]["k-core"]])
        kcore = test["baselines"]["k-core"]

        final_order = topL_corrected_order(rrf, kcore, args.L)
        # 把“最终排列位置”转成分数，位置越靠前分数越大
        n = len(final_order)
        final_scores = [0.0] * n
        for pos, i in enumerate(final_order):
            final_scores[i] = n - pos

        print(f"\n测试网络 = {test_net}")
        for label, vals in [
            ("k-core", kcore),
            ("RRF(XGB+k-core)", rrf),
            ("TopL-corrected", final_scores),
        ]:
            m = m06.evaluate(vals, test["y"], args.topk)
            detail.append([test_net, label, round(m["spearman"], 4),
                           round(m["kendall"], 4), round(m["ndcg"], 4),
                           m["top_hit"]])
            print(f"  {label:<18} Spearman={m['spearman']:.4f} "
                  f"Kendall={m['kendall']:.4f} NDCG={m['ndcg']:.4f} "
                  f"Top={m['top_hit']}")

    detail_csv = os.path.join(m06.RESULT_DIR, "leave_one_out_topL_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["test_network", "method", "spearman", "kendall",
                    "ndcg_top", "top_hit"])
        w.writerows(detail)

    methods = sorted({r[1] for r in detail})
    summary_csv = os.path.join(m06.RESULT_DIR, "leave_one_out_topL_summary.csv")
    print("\n汇总（4 网络均值 ± 标准差）")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "spearman", "kendall", "ndcg", "top_hit"])
        for method in methods:
            rows = [r for r in detail if r[1] == method]
            vals = {}
            for idx, metric in [(2, "spearman"), (3, "kendall"),
                                (4, "ndcg"), (5, "top_hit")]:
                xs = [r[idx] for r in rows]
                vals[metric] = (statistics.mean(xs), statistics.stdev(xs))
            w.writerow([
                method,
                f"{vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}",
                f"{vals['kendall'][0]:.4f}±{vals['kendall'][1]:.4f}",
                f"{vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}",
                f"{vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}",
            ])
            print(f"{method:<18}"
                  f" Spearman={vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}"
                  f" NDCG={vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}"
                  f" Top={vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}")
    print("\n文件：", detail_csv, " / ", summary_csv)


if __name__ == "__main__":
    main()
