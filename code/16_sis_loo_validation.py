# -*- coding: utf-8 -*-
"""
16_SIS 留一网络验证（与 SIR 主表流程完全对齐）
==============================================
数据：15_sis_labels.py 生成的 SIS 标签
  sis_labels_email-Eu-core_p0.025_r100_T50.csv
  sis_labels_Facebook_p0.012_r100_T50.csv
  sis_labels_US-power-grid_p0.35_r100_T50.csv
  sis_labels_OpenFlights_p0.03_r100_T50.csv

流程：与 SIR 主表一致——六特征、z-score、留一网络训练/测试；
方法：degree / k-core / PageRank / betweenness / XGBoost / TopL-200。

运行：python 16_sis_loo_validation.py
输出：results/sis_loo_detail.csv
      results/sis_loo_summary.csv
"""

import os
import sys
import csv
import statistics
import argparse
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(ROOT, "work", "figlibs4"),
           os.path.join(ROOT, "work", "figlibs3"),
           os.path.join(ROOT, "work", "figlibs")):
    if os.path.isdir(_p):
        sys.path.append(_p)

_spec = importlib.util.spec_from_file_location(
    "m06", os.path.join(ROOT, "outputs", "code", "06_leave_one_out_xgboost.py")
)
m06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m06)

import xgboost as xgb


SIS_PARAMS = {
    "email-Eu-core": dict(p=0.025, repeats=100, T=50),
    "Facebook": dict(p=0.012, repeats=100, T=50),
    "US-power-grid": dict(p=0.35, repeats=100, T=50),
    "OpenFlights": dict(p=0.03, repeats=100, T=50),
}


def load_sis(net):
    cfg = SIS_PARAMS[net]
    pstr = ("%g" % cfg["p"])
    path = os.path.join(m06.RESULT_DIR,
                        "sis_labels_%s_p%s_r%d_T%d.csv" %
                        (net, pstr, cfg["repeats"], cfg["T"]))
    if not os.path.exists(path):
        print(f"[!] 缺少 SIS 标签：{path}")
        raise SystemExit(1)
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["node"]] = float(row["mean_infected_time"])
    return out


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


def topL_order_scores(rrf_vals, kcore_vals, L):
    n = len(rrf_vals)
    rrf_order = sorted(range(n), key=lambda i: rrf_vals[i], reverse=True)
    kc_order = sorted(range(n), key=lambda i: kcore_vals[i], reverse=True)
    set_r, set_k = set(rrf_order[:L]), set(kc_order[:L])
    head = [i for i in rrf_order[:L] if i in set_k] + \
           [i for i in kc_order[:L] if i not in set_r] + \
           [i for i in rrf_order[:L] if i not in set_k]
    rest = [i for i in rrf_order if i not in set(head)]
    order = head + rest
    scores = [0.0] * n
    for pos, i in enumerate(order):
        scores[i] = n - pos
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    print("=" * 76)
    print("SIS 留一网络验证（与 SIR 主表流程对齐）")
    print("=" * 76)

    # 复用 SIR 脚本的数据结构（图、特征、基线），只替换标签
    data = {net: m06.prepare_network(net) for net in m06.NETWORKS}
    for net in m06.NETWORKS:
        sis = load_sis(net)
        d = data[net]
        y = [sis[u] for u in d["nodes"]]
        d["y"] = y
        d["y_rank"] = m06.rank_fraction(y)

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
        kcore = test["baselines"]["k-core"]
        rrf = rrf_scores([pred_xgb, kcore])
        topL = topL_order_scores(rrf, kcore, 200)

        print(f"\n测试网络 = {test_net}")
        methods = [
            ("degree", test["baselines"]["degree"]),
            ("k-core", kcore),
            ("PageRank", test["baselines"]["PageRank"]),
            ("betweenness", test["baselines"]["betweenness"]),
            ("XGBoost", pred_xgb),
            ("TopL-200", topL),
        ]
        for name, vals in methods:
            m = m06.evaluate(vals, test["y"], args.topk)
            detail.append([test_net, name, round(m["spearman"], 4),
                           round(m["kendall"], 4), round(m["ndcg"], 4),
                           m["top_hit"]])
            print(f"  {name:<12} Spearman={m['spearman']:.4f} "
                  f"Kendall={m['kendall']:.4f} NDCG={m['ndcg']:.4f} "
                  f"Top={m['top_hit']}")

    detail_csv = os.path.join(m06.RESULT_DIR, "sis_loo_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["test_network", "method", "spearman", "kendall",
                    "ndcg_top", "top_hit"])
        w.writerows(detail)

    methods = ["degree", "k-core", "PageRank", "betweenness",
               "XGBoost", "TopL-200"]
    summary_csv = os.path.join(m06.RESULT_DIR, "sis_loo_summary.csv")
    print("\nSIS 汇总（4 网络均值 ± 标准差）")
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
            print(f"{method:<12}"
                  f" Spearman={vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}"
                  f" NDCG={vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}"
                  f" Top={vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}")
    print("\n文件：", detail_csv, " / ", summary_csv)


if __name__ == "__main__":
    main()
