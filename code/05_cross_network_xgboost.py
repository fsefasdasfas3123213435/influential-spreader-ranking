# -*- coding: utf-8 -*-
"""
05_跨网络 XGBoost（论文正式版）
===============================
为什么需要这一份：
  04 是“同一个网络随机划分”，结果偏乐观，不能写进论文。
  这一份改成：在 email 和 Facebook 上训练，在没见过的
  US-power-grid、OpenFlights 上测试，才是论文能用的泛化实验。

跨网络最关键的一步：标签标准化
  每个网络的平均感染规模量纲不同，不能直接混在一起训练。
  我们把每个网络里的 SIR 标签换成“相对名次（0=最小，1=最大）”，
  模型学习的是“在这个网络里，结构特征如何决定相对重要性”，
  这样跨网络才是公平的。

运行前，请先把另外 3 个网络的 SIR 标签跑出来：

  python 02_sir_mc_simulation.py --network Facebook --prob 0.012 --repeats 200
  python 02_sir_mc_simulation.py --network US-power-grid --prob 0.35 --repeats 200
  python 02_sir_mc_simulation.py --network OpenFlights --prob 0.03 --repeats 200

然后运行本脚本：

  python 05_cross_network_xgboost.py

输出：
  results/cross_network_results.csv
"""

import os
import csv
import random
import argparse

import networkx as nx

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

NETWORKS = {
    "email-Eu-core": "email-Eu-core_undirected.edges",
    "Facebook": "facebook_undirected.edges",
    "ca-AstroPh": "ca-AstroPh_undirected.edges",
    "US-power-grid": "opsahl-powergrid_undirected.edges",
    "OpenFlights": "opsahl-openflights_undirected.edges",
}

# 每个网络生成 SIR 标签时建议使用的 p 和重复次数（与 02 脚本参数一致）
NET_PARAMS = {
    "email-Eu-core": dict(prob=0.025, repeats=500),
    "Facebook": dict(prob=0.012, repeats=200),
    "US-power-grid": dict(prob=0.35, repeats=200),
    "OpenFlights": dict(prob=0.03, repeats=200),
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


# ---------- 数据与特征 ----------
def label_path(network):
    p = NET_PARAMS[network]["prob"]
    r = NET_PARAMS[network]["repeats"]
    return os.path.join(RESULT_DIR, f"sir_labels_{network}_p{p}_r{r}.csv")


def load_labels(network):
    path = label_path(network)
    if not os.path.exists(path):
        p = NET_PARAMS[network]["prob"]
        r = NET_PARAMS[network]["repeats"]
        print(f"[!] 缺少标签：{path}")
        print(f"    请先运行：python 02_sir_mc_simulation.py "
              f"--network {network} --prob {p} --repeats {r}")
        raise SystemExit(1)
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["node"]] = float(row["mean_size"])
    return out


def load_graph(network):
    G = nx.read_edgelist(os.path.join(CLEANED_DIR, NETWORKS[network]))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def two_hop_degree(G, node):
    seen = {node} | set(G[node])
    for v in G[node]:
        seen |= set(G[v])
    return len(seen) - 1


def build_features(G, nodes):
    deg = dict(G.degree())
    core = nx.core_number(G)
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)
    clu = nx.clustering(G)
    names = ["degree", "kcore", "pagerank", "betweenness",
             "clustering", "two_hop_degree"]
    X = [[deg[u], core[u], pr[u], bc[u], clu[u], two_hop_degree(G, u)]
         for u in nodes]
    return names, X


def zscore_rows(X):
    """z-score standardization per column using this network's own stats."""
    nrow = len(X)
    ncol = len(X[0])
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
    """把 SIR 标签转成 0~1 的相对名次，用于跨网络训练。"""
    ranks = average_ranks(values)
    return [(r - 1.0) / max(len(values) - 1, 1) for r in ranks]


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
    parser.add_argument("--train", nargs="+",
                        default=["email-Eu-core", "Facebook"])
    parser.add_argument("--test", nargs="+",
                        default=["US-power-grid", "OpenFlights"])
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    print("=" * 74)
    print("跨网络 XGBoost（论文正式版）")
    print("训练：", ", ".join(args.train))
    print("测试（没见过的网络）：", ", ".join(args.test))
    print("=" * 74)

    # 组合训练数据
    Xtr_all, ytr_all = [], []
    for net in args.train:
        labels = load_labels(net)
        G = load_graph(net)
        nodes = [u for u in G.nodes() if u in labels]
        y = [labels[u] for u in nodes]
        names, X = build_features(G, nodes)
        X = zscore_rows(X)
        # 用相对名次作为训练目标
        yrank = rank_fraction(y)
        Xtr_all.extend(X)
        ytr_all.extend(yrank)
        print(f"训练网络 {net}: 节点 {len(nodes)}，目标=网络内相对名次")

    try:
        import xgboost as xgb
    except ImportError:
        print("[!] 未安装 xgboost：pip install xgboost")
        raise SystemExit(1)

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=args.seed, n_jobs=-1,
    )
    model.fit(Xtr_all, ytr_all)

    rows = []
    for net in args.test:
        print("-" * 74)
        print(f"测试网络：{net}")
        labels = load_labels(net)
        G = load_graph(net)
        nodes = [u for u in G.nodes() if u in labels]
        y = [labels[u] for u in nodes]
        _, X = build_features(G, nodes)
        X = zscore_rows(X)
        pred = [float(v) for v in model.predict(X)]
        uniq_pred = len(set(round(v, 4) for v in pred))
        print("  XGBoost 预测 min/max/不同取值数: "
              f"{min(pred):.3f}/{max(pred):.3f}/{uniq_pred}")

        pred_csv = os.path.join(RESULT_DIR, "cross_pred_" + net + ".csv")
        with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["node", "sir_label", "xgb_pred"])
            for i, u in enumerate(nodes):
                w.writerow([u, round(y[i], 4), round(pred[i], 4)])

        # 传统基线
        deg = dict(G.degree())
        core = nx.core_number(G)
        pr = nx.pagerank(G)
        baselines = {
            "degree": [deg[u] for u in nodes],
            "k-core": [core[u] for u in nodes],
            "PageRank": [pr[u] for u in nodes],
        }
        print(f"{'方法':<12}{'Spearman':>10}{'Kendall':>9}{'NDCG@':>9}{'Top命中':>9}")
        for name in ["degree", "k-core", "PageRank", "XGBoost"]:
            vals = pred if name == "XGBoost" else baselines[name]
            m = evaluate(vals, y, args.topk)
            rows.append([net, name, round(m["spearman"], 4),
                         round(m["kendall"], 4), round(m["ndcg"], 4),
                         m["top_hit"], len(nodes)])
            print(f"{name:<12}{m['spearman']:>10.4f}{m['kendall']:>9.4f}"
                  f"{m['ndcg']:>9.4f}{m['top_hit']:>9}")

    out_csv = os.path.join(RESULT_DIR, "cross_network_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["network", "method", "spearman", "kendall", "ndcg_top",
                    "top_hit", "nodes"])
        w.writerows(rows)

    imp = sorted(zip(names, model.feature_importances_),
                 key=lambda t: t[1], reverse=True)
    print("\n特征重要性（跨网络训练）：")
    for name, v in imp:
        print(f"  {name:<14}: {v:.3f}")
    print("\n结果文件：", out_csv)
    print("这张表就是论文主表雏形：XGBoost 与 degree/k-core/PageRank 对比。")


if __name__ == "__main__":
    main()
