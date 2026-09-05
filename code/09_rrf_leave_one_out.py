# -*- coding: utf-8 -*-
"""
09_倒数排名融合 RRF（留一网络交叉验证）
=======================================
方法：
  对每个测试网络，先分别得到“XGBoost(回归)排序”和“k-core 排序”，
  每个节点在两种方法中都有一个名次 rank_m（1 为最重要）。
  RRF 分数：
      RRF_score(i) = 1/(k + rank_xgb(i)) + 1/(k + rank_kcore(i))
  其中 k=60 是标准平滑常数。
  它比“名次直接取平均”更稳健：即使 XGBoost 把某个错误节点排到第 1，
  该节点在 k-core 里名次很差时，RRF 分数不会特别高。

运行：python 09_rrf_leave_one_out.py
输出：results/leave_one_out_rrf_detail.csv
      results/leave_one_out_rrf_summary.csv
"""

import os
import sys
import importlib.util
import csv
import statistics
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# 如果运行机器上已装有 networkx/xgboost，则不需要 work/figlibs；
# 这里仅在本地依赖目录存在且可用时加入搜索路径。
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
    """score_lists: 多个方法的原始分数列表，值越大越重要。返回 RRF 分数。"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--k", type=int, default=60,
                        help="RRF 平滑常数，标准值 60")
    parser.add_argument("--add_degree", action="store_true",
                        help="额外把 degree 也加入 RRF（默认只用 XGBoost+k-core）")
    args = parser.parse_args()

    print("=" * 76)
    print(f"RRF 倒数排名融合（k={args.k}），留一网络交叉验证")
    if args.add_degree:
        print("融合成员：XGBoost + k-core + degree")
    else:
        print("融合成员：XGBoost + k-core")
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
        pred_xgb = [float(v) for v in model.predict(test["X"])]
        members = [pred_xgb, test["baselines"]["k-core"]]
        if args.add_degree:
            members.append(test["baselines"]["degree"])

        fusion = rrf_scores(members, k=args.k)
        name = "RRF(XGB+k-core)"
        if args.add_degree:
            name = "RRF(XGB+k-core+degree)"

        print(f"\n测试网络 = {test_net}")
        for label, vals in [
            ("XGBoost", pred_xgb),
            ("k-core", test["baselines"]["k-core"]),
            (name, fusion),
        ]:
            m = m06.evaluate(vals, test["y"], args.topk)
            detail.append([test_net, label, round(m["spearman"], 4),
                           round(m["kendall"], 4), round(m["ndcg"], 4),
                           m["top_hit"]])
            print(f"  {label:<26} Spearman={m['spearman']:.4f} "
                  f"Kendall={m['kendall']:.4f} NDCG={m['ndcg']:.4f} "
                  f"Top={m['top_hit']}")

    detail_csv = os.path.join(m06.RESULT_DIR, "leave_one_out_rrf_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["test_network", "method", "spearman", "kendall",
                       "ndcg_top", "top_hit"])
        wcsv.writerows(detail)

    methods = sorted({r[1] for r in detail})
    summary_csv = os.path.join(m06.RESULT_DIR, "leave_one_out_rrf_summary.csv")
    print("\n汇总（4 网络均值 ± 标准差）")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["method", "spearman", "kendall", "ndcg", "top_hit"])
        for method in methods:
            rows = [r for r in detail if r[1] == method]
            vals = {}
            for idx, metric in [(2, "spearman"), (3, "kendall"),
                                (4, "ndcg"), (5, "top_hit")]:
                xs = [r[idx] for r in rows]
                vals[metric] = (statistics.mean(xs), statistics.stdev(xs))
            wcsv.writerow([
                method,
                f"{vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}",
                f"{vals['kendall'][0]:.4f}±{vals['kendall'][1]:.4f}",
                f"{vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}",
                f"{vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}",
            ])
            print(f"{method:<26}"
                  f" Spearman={vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}"
                  f" NDCG={vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}"
                  f" Top={vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}")
    print("\n文件：", detail_csv, " / ", summary_csv)


if __name__ == "__main__":
    main()
