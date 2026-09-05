# -*- coding: utf-8 -*-
"""
08_XGBoost + k-core 排名融合（留一网络交叉验证）
================================================
动机：
  回归目标 XGBoost：整体 Spearman 好，但 Facebook top-100 差；
  rank 目标 XGBoost：top-100 好，但整体 Spearman 崩溃。
  这里把“回归版 XGBoost 排序”和“k-core 排序”做等权排名融合，
  希望同时保住整体排序与顶部稳定性。

运行：python 08_fusion_leave_one_out.py
输出：results/leave_one_out_fusion_detail.csv
      results/leave_one_out_fusion_summary.csv
"""

import os
import sys
import importlib.util
import csv
import statistics6

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(os.path.join(ROOT, "work", "figlibs"))

# 复用 06 脚本里的数据准备与评价函数（不重复实现）
_spec = importlib.util.spec_from_file_location(
    "m06", os.path.join(ROOT, "outputs", "code", "06_leave_one_out_xgboost.py")
)
m06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m06)

import networkx as nx
import xgboost as xgb


def rank_fraction_desc(values):
    """把一个方法的分数转成 0~1 的相对名次；值越大名次越靠前。"""
    ranks = m06.average_ranks(values)
    n = len(values)
    return [(r - 1.0) / max(n - 1, 1) for r in ranks]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--w_xgb", type=float, default=0.5,
                        help="融合权重：XGBoost 的权重，k-core 权重 = 1-w_xgb")
    args = parser.parse_args()
    w = args.w_xgb

    print("=" * 76)
    print(f"XGBoost(回归) + k-core 融合（w_xgb={w}），留一网络交叉验证")
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

        r_xgb = rank_fraction_desc(pred_xgb)
        r_kcore = rank_fraction_desc(test["baselines"]["k-core"])
        r_deg = rank_fraction_desc(test["baselines"]["degree"])
        fusion_ck = [w * a + (1 - w) * b for a, b in zip(r_xgb, r_kcore)]
        fusion_cd = [w * a + (1 - w) * b for a, b in zip(r_xgb, r_deg)]

        print(f"\n测试网络 = {test_net}")
        for name, vals in [
            ("XGBoost", pred_xgb),
            ("k-core", test["baselines"]["k-core"]),
            ("degree", test["baselines"]["degree"]),
            ("Fusion(XGB+k-core)", fusion_ck),
            ("Fusion(XGB+degree)", fusion_cd),
        ]:
            m = m06.evaluate(vals, test["y"], args.topk)
            detail.append([test_net, name, round(m["spearman"], 4),
                           round(m["kendall"], 4), round(m["ndcg"], 4),
                           m["top_hit"]])
            print(f"  {name:<22} Spearman={m['spearman']:.4f} "
                  f"Kendall={m['kendall']:.4f} NDCG={m['ndcg']:.4f} "
                  f"Top={m['top_hit']}")

    detail_csv = os.path.join(m06.RESULT_DIR, "leave_one_out_fusion_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["test_network", "method", "spearman", "kendall",
                       "ndcg_top", "top_hit"])
        wcsv.writerows(detail)

    methods = ["XGBoost", "k-core", "degree",
               "Fusion(XGB+k-core)", "Fusion(XGB+degree)"]
    summary_csv = os.path.join(m06.RESULT_DIR, "leave_one_out_fusion_summary.csv")
    print("\n论文主表候选：4 网络均值 ± 标准差")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["method", "spearman", "kendall", "ndcg", "top_hit"])
        for method in methods:
            rows = [r for r in detail if r[1] == method]
            vals = {}
            for idx, name in [(2, "spearman"), (3, "kendall"),
                              (4, "ndcg"), (5, "top_hit")]:
                xs = [r[idx] for r in rows]
                vals[name] = (statistics.mean(xs), statistics.stdev(xs))
            wcsv.writerow([
                method,
                f"{vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}",
                f"{vals['kendall'][0]:.4f}±{vals['kendall'][1]:.4f}",
                f"{vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}",
                f"{vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}",
            ])
            print(f"{method:<22}"
                  f" Spearman={vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}"
                  f" NDCG={vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}"
                  f" Top={vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}")
    print("\n文件：", detail_csv, " / ", summary_csv)


if __name__ == "__main__":
    main()
