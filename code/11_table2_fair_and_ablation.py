# -*- coding: utf-8 -*-
"""
11_Table2 公平口径 + 消融表 + L 稳定性
=======================================
Part A（Table 2 公平口径）
  email-Eu-core、p=0.025、R=500。
  degree / k-core / PageRank / betweenness / XGBoost
  全部只在同一个 20% 测试节点子集上评价（修复“基线用全网络、
  XGBoost 用 198 个测试节点”的口径不一致）。

Part B（消融 + L 稳定性，留一网络交叉验证）
  每一轮用其余 3 个网络训练回归版 XGBoost，测试第 4 个网络；
  对比：
    k-core、XGBoost(regression)、RRF(XGB+k-core)、
    TopL-50、TopL-100、TopL-200

运行：python 11_table2_fair_and_ablation.py
输出：
  results/table2_fair_email.csv
  results/ablation_detail.csv
  results/ablation_summary.csv
"""

import os
import sys
import csv
import random
import importlib.util
import statistics
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_local_lib = os.path.join(ROOT, "work", "figlibs")
for _p in (os.path.join(ROOT, "work", "figlibs3"),
           os.path.join(ROOT, "work", "figlibs")):
    if os.path.isdir(_p):
        sys.path.append(_p)

_spec = importlib.util.spec_from_file_location(
    "m06", os.path.join(ROOT, "outputs", "code", "06_leave_one_out_xgboost.py")
)
m06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m06)

import networkx as nx
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


def topL_scores_from_order(rrf_vals, kcore_vals, L):
    n = len(rrf_vals)
    rrf_order = sorted(range(n), key=lambda i: rrf_vals[i], reverse=True)
    kc_order = sorted(range(n), key=lambda i: kcore_vals[i], reverse=True)
    set_rrf, set_kc = set(rrf_order[:L]), set(kc_order[:L])
    top_segment = [i for i in rrf_order[:L] if i in set_kc] + \
                  [i for i in kc_order[:L] if i not in set_rrf] + \
                  [i for i in rrf_order[:L] if i not in set_kc]
    rest = [i for i in rrf_order if i not in set(top_segment)]
    final_order = top_segment + rest
    scores = [0.0] * n
    for pos, i in enumerate(final_order):
        scores[i] = n - pos
    return scores


# ---------------- Part A ----------------
def part_table2_fair(seed=2026):
    print("=" * 76)
    print("Part A：email-Eu-core Table 2 公平口径")
    print("=" * 76)
    G = nx.read_edgelist(os.path.join(m06.CLEANED_DIR,
                                      "email-Eu-core_undirected.edges"))
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = list(G.nodes())

    label_path = os.path.join(
        m06.RESULT_DIR, "sir_labels_email-Eu-core_p0.025_r500.csv")
    labels = {}
    with open(label_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["node"]] = float(row["mean_size"])
    nodes = [u for u in nodes if u in labels]
    y = [labels[u] for u in nodes]
    n = len(nodes)

    deg = dict(G.degree())
    core = nx.core_number(G)
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)
    clu = nx.clustering(G)

    def twohop(u):
        seen = {u} | set(G[u])
        for v in G[u]:
            seen |= set(G[v])
        return len(seen) - 1

    X_all = [[deg[u], core[u], pr[u], bc[u], clu[u], twohop(u)] for u in nodes]
    baseline = {
        "degree": [deg[u] for u in nodes],
        "k-core": [core[u] for u in nodes],
        "PageRank": [pr[u] for u in nodes],
        "betweenness": [bc[u] for u in nodes],
    }

    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    cut = int(n * 0.8)
    train_idx, test_idx = perm[:cut], perm[cut:]
    print(f"训练节点 {len(train_idx)}，测试节点 {len(test_idx)}")

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
    )
    model.fit([X_all[i] for i in train_idx], [y[i] for i in train_idx])
    pred_xgb = [float(v) for v in model.predict([X_all[i] for i in test_idx])]

    rows = []
    test_y = [y[i] for i in test_idx]
    for name, vals in [("degree", baseline["degree"]),
                       ("k-core", baseline["k-core"]),
                       ("PageRank", baseline["PageRank"]),
                       ("betweenness", baseline["betweenness"])]:
        test_vals = [vals[i] for i in test_idx]
        m = m06.evaluate(test_vals, test_y, 100)
        rows.append([name, round(m["spearman"], 4), round(m["kendall"], 4),
                     round(m["ndcg"], 4), m["top_hit"], len(test_idx)])
    # XGBoost 的 pred_xgb 本身就是测试节点上的预测，直接评价
    m = m06.evaluate(pred_xgb, test_y, 100)
    rows.append(["XGBoost", round(m["spearman"], 4), round(m["kendall"], 4),
                 round(m["ndcg"], 4), m["top_hit"], len(test_idx)])

    print(f"\n{'方法':<14}{'Spearman':>10}{'Kendall':>9}{'NDCG@100':>10}"
          f"{'Top命中':>9}")
    for r in rows:
        print(f"{r[0]:<14}{r[1]:>10.4f}{r[2]:>9.4f}{r[3]:>10.4f}{r[4]:>9}")

    out_csv = os.path.join(m06.RESULT_DIR, "table2_fair_email.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "spearman", "kendall", "ndcg_top100",
                    "top100_hit", "test_nodes"])
        w.writerows(rows)
    print("保存：", out_csv)


# ---------------- Part B ----------------
def part_ablation(seed=2026):
    print("\n" + "=" * 76)
    print("Part B：留一网络消融 + L 稳定性")
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
            random_state=seed, n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        test = data[test_net]
        pred_xgb = [float(v) for v in model.predict(test["X"])]
        kcore = test["baselines"]["k-core"]
        rrf = rrf_scores([pred_xgb, kcore])

        print(f"\n测试网络 = {test_net}")
        candidates = [
            ("k-core", kcore),
            ("XGBoost", pred_xgb),
            ("RRF", rrf),
            ("TopL-50", topL_scores_from_order(rrf, kcore, 50)),
            ("TopL-100", topL_scores_from_order(rrf, kcore, 100)),
            ("TopL-200", topL_scores_from_order(rrf, kcore, 200)),
        ]
        for name, vals in candidates:
            m = m06.evaluate(vals, test["y"], 100)
            detail.append([test_net, name, round(m["spearman"], 4),
                           round(m["kendall"], 4), round(m["ndcg"], 4),
                           m["top_hit"]])
            print(f"  {name:<10} Spearman={m['spearman']:.4f} "
                  f"Kendall={m['kendall']:.4f} NDCG={m['ndcg']:.4f} "
                  f"Top={m['top_hit']}")

    detail_csv = os.path.join(m06.RESULT_DIR, "ablation_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["test_network", "method", "spearman", "kendall",
                    "ndcg_top", "top_hit"])
        w.writerows(detail)

    methods = ["k-core", "XGBoost", "RRF", "TopL-50", "TopL-100", "TopL-200"]
    summary_csv = os.path.join(m06.RESULT_DIR, "ablation_summary.csv")
    print("\n消融 + L 稳定性汇总（4 网络均值 ± 标准差）")
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
            print(f"{method:<10}"
                  f" Spearman={vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}"
                  f" NDCG={vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}"
                  f" Top={vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}")
    print("\n保存：", detail_csv, " / ", summary_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["table2", "ablation", "both"],
                        default="both")
    args = parser.parse_args()
    if args.part in ("table2", "both"):
        part_table2_fair()
    if args.part in ("ablation", "both"):
        part_ablation()
