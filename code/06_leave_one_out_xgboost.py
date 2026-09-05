# -*- coding: utf-8 -*-
"""
06_留一网络交叉验证（论文主表正式版）
====================================
目的：
  不再只做“email+Facebook → 电网+航线”这一组，而是依次把
  email-Eu-core、Facebook、US-power-grid、OpenFlights 各留出一次：

  第 1 轮：训练 Facebook + 电网 + 航线，测试 email；
  第 2 轮：训练 email + 电网 + 航线，测试 Facebook；
  第 3 轮：训练 email + Facebook + 航线，测试电网；
  第 4 轮：训练 email + Facebook + 电网，测试航线。

  每个测试网络都算 degree / k-core / PageRank / betweenness / XGBoost
  的 Spearman、Kendall、NDCG@100、Top-100 命中。
  最后输出 4 轮均值±标准差——这张就是论文主表。

运行前：4 个网络的 SIR 标签都必须存在：
  email-Eu-core      p=0.025 r=500
  Facebook           p=0.012 r=200
  US-power-grid      p=0.35  r=200
  OpenFlights        p=0.03  r=200

运行：
  python 06_leave_one_out_xgboost.py

输出：
  results/leave_one_out_detail.csv
  results/leave_one_out_summary.csv
"""

import os
import csv
import statistics
import argparse

import networkx as nx


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

NETWORKS = ["email-Eu-core", "Facebook", "US-power-grid", "OpenFlights"]
NET_PARAMS = {
    "email-Eu-core": dict(prob=0.025, repeats=500),
    "Facebook": dict(prob=0.012, repeats=200),
    "US-power-grid": dict(prob=0.35, repeats=200),
    "OpenFlights": dict(prob=0.03, repeats=200),
}
EDGE_FILES = {
    "email-Eu-core": "email-Eu-core_undirected.edges",
    "Facebook": "facebook_undirected.edges",
    "US-power-grid": "opsahl-powergrid_undirected.edges",
    "OpenFlights": "opsahl-openflights_undirected.edges",
}


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i, n = 0, len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = average_ranks(x), average_ranks(y)
    n = len(x)
    dx = [a - (n + 1) / 2.0 for a in rx]
    dy = [a - (n + 1) / 2.0 for a in ry]
    den = (sum(d * d for d in dx) * sum(d * d for d in dy)) ** 0.5
    return sum(a * b for a, b in zip(dx, dy)) / den if den else 0.0


def kendall_tau(x, y):
    n = len(x)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = (x[i] > x[j]) - (x[i] < x[j])
            sy = (y[i] > y[j]) - (y[i] < y[j])
            if sx * sy > 0:
                conc += 1
            elif sx * sy < 0:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2.0)


def dcg(order, rel, k):
    return sum(rel[order[i]] / (i + 2) ** 0.5 for i in range(min(k, len(order))))


def ndcg(pred_order, rel, k):
    ideal = sorted(range(len(rel)), key=lambda i: rel[i], reverse=True)
    idcg = dcg(ideal, rel, k)
    return dcg(pred_order, rel, k) / idcg if idcg else 0.0


def overlap(a, b, k):
    return len(set(a[:k]) & set(b[:k]))


def zscore_rows(X):
    nrow, ncol = len(X), len(X[0])
    cols = []
    for j in range(ncol):
        col = [X[i][j] for i in range(nrow)]
        mean = sum(col) / nrow
        std = (sum((v - mean) ** 2 for v in col) / nrow) ** 0.5
        if std == 0:
            std = 1.0
        cols.append([(X[i][j] - mean) / std for i in range(nrow)])
    return [[cols[j][i] for j in range(ncol)] for i in range(nrow)]


def rank_fraction(values):
    ranks = average_ranks(values)
    return [(r - 1.0) / max(len(values) - 1, 1) for r in ranks]


def two_hop_degree(G, node):
    seen = {node} | set(G[node])
    for v in G[node]:
        seen |= set(G[v])
    return len(seen) - 1


def load_label_path(network):
    p = NET_PARAMS[network]["prob"]
    r = NET_PARAMS[network]["repeats"]
    return os.path.join(RESULT_DIR, f"sir_labels_{network}_p{p}_r{r}.csv")


def prepare_network(network):
    """读取网络 + SIR 标签，一次性算好特征/基线，缓存供多轮使用。"""
    path = load_label_path(network)
    if not os.path.exists(path):
        p = NET_PARAMS[network]["prob"]
        r = NET_PARAMS[network]["repeats"]
        print(f"[!] 缺少 {path}")
        print(f"    请先运行：python 02_sir_mc_simulation.py "
              f"--network {network} --prob {p} --repeats {r}")
        raise SystemExit(1)

    labels = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["node"]] = float(row["mean_size"])

    G = nx.read_edgelist(os.path.join(CLEANED_DIR, EDGE_FILES[network]))
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = [u for u in G.nodes() if u in labels]
    y = [labels[u] for u in nodes]

    print(f"  预处理 {network} ...（含介数计算，可能稍慢）")
    deg = dict(G.degree())
    core = nx.core_number(G)
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)
    clu = nx.clustering(G)
    names = ["degree", "kcore", "pagerank", "betweenness",
             "clustering", "two_hop_degree"]
    X_raw = [[deg[u], core[u], pr[u], bc[u], clu[u], two_hop_degree(G, u)]
             for u in nodes]

    return {
        "nodes": nodes,
        "y": y,
        "X": zscore_rows(X_raw),
        "y_rank": rank_fraction(y),
        "y_order": [int(r) for r in average_ranks(y)],
        "baselines": {
            "degree": [deg[u] for u in nodes],
            "k-core": [core[u] for u in nodes],
            "PageRank": [pr[u] for u in nodes],
            "betweenness": [bc[u] for u in nodes],
        },
        "G": G,
        "features": names,
    }


def evaluate(pred, truth, topk):
    rho = spearman(pred, truth)
    tau = kendall_tau(pred, truth)
    porder = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)
    torder = sorted(range(len(pred)), key=lambda i: truth[i], reverse=True)
    return {
        "spearman": rho,
        "kendall": tau,
        "ndcg": ndcg(porder, truth, min(topk, len(pred))),
        "top_hit": overlap(porder, torder, min(topk, len(pred))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--loss",
        default="rank",
        choices=["rank", "reg"],
        help="rank=排序目标(默认，推荐论文主表)；reg=旧版回归目标",
    )
    args = parser.parse_args()

    print("=" * 76)
    print("留一网络交叉验证（论文主表）")
    print("网络顺序：", " -> ".join(NETWORKS))
    print("=" * 76)

    try:
        import xgboost as xgb
    except ImportError:
        print("[!] 未安装 xgboost：pip install xgboost")
        raise SystemExit(1)

    # 预处理 4 个网络（只做一次）
    data = {net: prepare_network(net) for net in NETWORKS}
    detail_rows = []

    for test_net in NETWORKS:
        train_nets = [n for n in NETWORKS if n != test_net]
        print("\n" + "=" * 76)
        fold_no = NETWORKS.index(test_net) + 1
        print(f"第 {fold_no}/4 轮：测试={test_net}，训练={train_nets}")

        Xtr_all, ytr_all, groups = [], [], []
        for net in train_nets:
            d = data[net]
            Xtr_all.extend(d["X"])
            if args.loss == "rank":
                ytr_all.extend(d["y_order"])
            else:
                ytr_all.extend(d["y_rank"])
            groups.append(len(d["X"]))

        common_params = dict(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=args.seed, n_jobs=-1,
        )
        if args.loss == "rank":
            model = xgb.XGBRanker(objective="rank:pairwise", **common_params)
            model.fit(Xtr_all, ytr_all, group=groups)
        else:
            model = xgb.XGBRegressor(**common_params)
            model.fit(Xtr_all, ytr_all)

        test = data[test_net]
        pred = [float(v) for v in model.predict(test["X"])]
        print(f"  XGBoost 预测 min/max/不同取值数: "
              f"{min(pred):.3f}/{max(pred):.3f}/"
              f"{len(set(round(v, 4) for v in pred))}")

        print(f"\n{'方法':<12}{'Spearman':>10}{'Kendall':>9}{'NDCG@':>9}{'Top命中':>9}")
        for name in ["degree", "k-core", "PageRank", "betweenness", "XGBoost"]:
            vals = pred if name == "XGBoost" else test["baselines"][name]
            m = evaluate(vals, test["y"], args.topk)
            detail_rows.append([
                test_net, name,
                round(m["spearman"], 4), round(m["kendall"], 4),
                round(m["ndcg"], 4), m["top_hit"],
            ])
            print(f"{name:<12}{m['spearman']:>10.4f}{m['kendall']:>9.4f}"
                  f"{m['ndcg']:>9.4f}{m['top_hit']:>9}")

    # 保存每轮明细
    detail_csv = os.path.join(
        RESULT_DIR, f"leave_one_out_detail_{args.loss}.csv"
    )
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["test_network", "method", "spearman", "kendall",
                    "ndcg_top", "top_hit"])
        w.writerows(detail_rows)

    # 汇总：每种方法在 4 个测试网络上的均值±标准差
    methods = ["degree", "k-core", "PageRank", "betweenness", "XGBoost"]
    summary_rows = []
    print("\n" + "=" * 76)
    print("论文主表：4 个留一网络的 均值 ± 标准差")
    print("=" * 76)
    print(f"{'方法':<12}{'Spearman':>16}{'Kendall':>14}{'NDCG@100':>14}{'Top命中':>14}")

    for method in methods:
        rows = [r for r in detail_rows if r[1] == method]
        vals = {}
        for metric_idx, metric_name in [(2, "spearman"), (3, "kendall"),
                                        (4, "ndcg"), (5, "top_hit")]:
            xs = [r[metric_idx] for r in rows]
            mean = statistics.mean(xs)
            std = statistics.stdev(xs) if len(xs) > 1 else 0.0
            vals[metric_name] = (mean, std)
        summary_rows.append([
            method,
            f"{vals['spearman'][0]:.4f}±{vals['spearman'][1]:.4f}",
            f"{vals['kendall'][0]:.4f}±{vals['kendall'][1]:.4f}",
            f"{vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}",
            f"{vals['top_hit'][0]:.1f}±{vals['top_hit'][1]:.1f}",
        ])
        print(
            f"{method:<12}{vals['spearman'][0]:>8.4f}±{vals['spearman'][1]:<6.4f}"
            f"{vals['kendall'][0]:>7.4f}±{vals['kendall'][1]:<6.4f}"
            f"{vals['ndcg'][0]:>7.4f}±{vals['ndcg'][1]:<6.4f}"
            f"{vals['top_hit'][0]:>6.1f}±{vals['top_hit'][1]:<5.1f}"
        )

    summary_csv = os.path.join(
        RESULT_DIR, f"leave_one_out_summary_{args.loss}.csv"
    )
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "spearman_mean±std", "kendall_mean±std",
                    "ndcg_mean±std", "top_hit_mean±std"])
        w.writerows(summary_rows)

    print("\n明细文件：", detail_csv)
    print("主表文件：", summary_csv)


if __name__ == "__main__":
    main()
