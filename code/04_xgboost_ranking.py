# -*- coding: utf-8 -*-
"""
04_XGBoost 训练与评估（练手版）
===============================
做什么：
  用“网络结构特征”预测“SIR 影响力标签”，再拿预测排序和真实 SIR 排序比较。
  这一步跑通后，你就有“自己的方法”的第一版结果了。

特征（先给节点做“画像”）：
  degree            度
  kcore             k-core
  pagerank          PageRank
  betweenness       介数
  clustering        局部聚类系数
  two_hop_degree    二阶邻居数（邻居的邻居数量，表示节点周边的“扩散潜力”）

模型：
  XGBoost 回归，预测每个节点的 SIR 平均感染规模，再用预测值排序。

⚠️ 重要提醒：
  本版本把 email-Eu-core 的节点随机分成训练 80% / 测试 20%，
  只是用来“跑通流程”。论文里不能这样下结论，
  因为同一网络里的节点结构相似，结果会偏乐观。
  论文正式实验要改成“网络 A 训练、网络 B 测试”（跑通本脚本后我再给扩展版）。

安装（第 1 次）：
  pip install xgboost

运行：
  python 04_xgboost_ranking.py

输出：
  results/xgb_metrics_email-Eu-core_p0.025_r500.csv
  results/xgb_predictions_email-Eu-core_p0.025_r500.csv
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


# ---------- 与 03 相同的评价指标（便于直接对比） ----------
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


def overlap(pred_order, true_order, k):
    return len(set(pred_order[:k]) & set(true_order[:k]))


# ---------- 特征 ----------
def two_hop_degree(G, node):
    """二阶邻居数：距离不超过 2 的节点数（不含自身）。"""
    seen = {node}
    frontier = set(G[node])
    seen |= frontier
    for v in frontier:
        seen |= set(G[v])
    return len(seen) - 1


def build_features(G, nodes):
    """返回：特征名列表、每个节点的特征向量列表。"""
    deg = dict(G.degree())
    core = nx.core_number(G)
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)
    clu = nx.clustering(G)

    features = []
    for u in nodes:
        features.append([
            deg[u],
            core[u],
            pr[u],
            bc[u],
            clu[u],
            two_hop_degree(G, u),
        ])
    return ["degree", "kcore", "pagerank", "betweenness",
            "clustering", "two_hop_degree"], features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="email-Eu-core",
                        choices=sorted(NETWORKS.keys()))
    parser.add_argument("--prob", type=float, default=0.025)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    print("=" * 70)
    print(f"XGBoost：{args.network}（SIR p={args.prob}, R={args.repeats}）")
    print("=" * 70)

    # 载入图和 SIR 标签
    G = nx.read_edgelist(os.path.join(CLEANED_DIR, NETWORKS[args.network]))
    G.remove_edges_from(nx.selfloop_edges(G))
    label_path = os.path.join(
        RESULT_DIR,
        f"sir_labels_{args.network}_p{args.prob}_r{args.repeats}.csv",
    )
    if not os.path.exists(label_path):
        print(f"[!] 找不到 {label_path}，请先运行 02 脚本。")
        raise SystemExit(1)
    node2y = {}
    with open(label_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            node2y[row["node"]] = float(row["mean_size"])

    nodes = [u for u in G.nodes() if u in node2y]
    y = [node2y[u] for u in nodes]
    n = len(nodes)
    print(f"节点数 N = {n}")

    print("正在构造特征（email 网络 1 分钟内；不要直接跑 ca-AstroPh）...")
    feature_names, X = build_features(G, nodes)
    print("特征：", feature_names)

    # 随机划分训练/测试（练手版）
    rng = random.Random(args.seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(n * (1 - args.test_ratio))
    train_idx, test_idx = idx[:cut], idx[cut:]
    print(f"训练节点 {len(train_idx)}，测试节点 {len(test_idx)}")

    Xtr = [X[i] for i in train_idx]
    ytr = [y[i] for i in train_idx]
    Xte = [X[i] for i in test_idx]
    yte = [y[i] for i in test_idx]

    # 训练 XGBoost
    try:
        import xgboost as xgb
    except ImportError:
        print("[!] 未安装 xgboost，请先运行：pip install xgboost")
        raise SystemExit(1)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    pred = [float(v) for v in model.predict(Xte)]

    # 指标：只用测试节点的相对排序评价
    rho = spearman(pred, yte)
    tau = kendall_tau(pred, yte)
    pred_order = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)
    true_order = sorted(range(len(pred)), key=lambda i: yte[i], reverse=True)
    ndcg_val = ndcg(pred_order, yte, min(args.topk, len(pred)))
    hit = overlap(pred_order, true_order, min(args.topk, len(pred)))

    print("\n" + "-" * 56)
    print(f"{'方法':<14}{'Spearman':>10}{'Kendall':>9}{'NDCG@':>9}{'Top命中':>9}")
    print(f"{'XGBoost':<14}{rho:>10.4f}{tau:>9.4f}{ndcg_val:>9.4f}{hit:>9}")
    print("-" * 56)

    # 保存预测与特征重要性
    pred_csv = os.path.join(
        RESULT_DIR,
        f"xgb_predictions_{args.network}_p{args.prob}_r{args.repeats}.csv",
    )
    with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["node", "sir_label", "xgb_pred"])
        for original_i, i in enumerate(test_idx):
            w.writerow([nodes[i], round(y[i], 4), round(pred[original_i], 4)])

    metric_csv = os.path.join(
        RESULT_DIR,
        f"xgb_metrics_{args.network}_p{args.prob}_r{args.repeats}.csv",
    )
    with open(metric_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "spearman", "kendall", "ndcg_top", "top_hit",
                    "test_nodes"])
        w.writerow(["XGBoost", round(rho, 4), round(tau, 4), round(ndcg_val, 4),
                    hit, len(test_idx)])

    imp = sorted(zip(feature_names, model.feature_importances_),
                 key=lambda t: t[1], reverse=True)
    print("\n特征重要性（越大越有用）：")
    for name, v in imp:
        print(f"  {name:<14}: {v:.3f}")

    print("\n结果文件：")
    print("  ", metric_csv)
    print("  ", pred_csv)
    print("\n注意：本结果是单网络随机划分的“练手结果”。")
    print("论文正式版需要改成跨网络训练/测试，下一份脚本再做。")


if __name__ == "__main__":
    main()
