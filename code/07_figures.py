# -*- coding: utf-8 -*-
"""
07_Figure1 & Figure2 生成脚本
==============================
Figure 1：email-Eu-core 上，不同传播概率 p 下各方法的 Spearman / NDCG@100。
          XGBoost 使用“同一网络内 80/20 节点划分”（与论文 Table 2 同一协议）；
          为使对比公平，图里所有方法都只在同一个 20% 测试节点子集上计算指标。

Figure 2：email-Eu-core，p=0.025，R=500 的标签上比较免疫预算 L
          （度/k-core/PageRank/介数/XGBoost/随机）对“残余平均感染规模”的影响。
          XGBoost 排名来自“其余 3 个网络训练→email 预测”的跨网络模型。

运行：python 07_figures.py
输出：outputs/figures/Figure1_spearman_ndcg_vs_p.png
      outputs/figures/Figure2_immunization_budget.png
      以及结果 CSV 到 code/results/。
"""

import os
import csv
import sys
import random
import argparse
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _cand in ("work/figlibs4", "work/figlibs3", "work/figlibs"):
    _p = os.path.join(ROOT, _cand)
    if os.path.isdir(_p):
        sys.path.append(_p)

import networkx as nx
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLEANED = os.path.join(ROOT, "outputs", "datasets", "cleaned")
RESULT_DIR = os.path.join(ROOT, "outputs", "code", "results")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EMAIL_FILE = os.path.join(CLEANED, "email-Eu-core_undirected.edges")


# ---------------- 指标 ----------------
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


def dcg(order, rel, k):
    return sum(rel[order[i]] / (i + 2) ** 0.5 for i in range(min(k, len(order))))


def ndcg_at_k(pred_order, rel, k):
    ideal = sorted(range(len(rel)), key=lambda i: rel[i], reverse=True)
    idcg = dcg(ideal, rel, k)
    return dcg(pred_order, rel, k) / idcg if idcg else 0.0


def eval_metrics(pred, truth, topk=100):
    rho = spearman(pred, truth)
    porder = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)
    return rho, ndcg_at_k(porder, truth, min(topk, len(pred)))


# ---------------- SIR percolation ----------------
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def load_email():
    G = nx.read_edgelist(EMAIL_FILE)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def make_labels(G, p, repeats, seed=2026):
    """返回 dict node->平均连通分量规模（SIR 终态等价）。"""
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    n = len(nodes)
    acc = [0.0] * n
    rng = random.Random(seed)
    for _ in range(repeats):
        dsu = DSU(n)
        for u, v in edges:
            if rng.random() < p:
                dsu.union(u, v)
        for i in range(n):
            acc[i] += dsu.size[dsu.find(i)]
    return {u: acc[i] / repeats for i, u in enumerate(nodes)}


def build_features(G, nodes):
    deg = dict(G.degree())
    core = nx.core_number(G)
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)
    clu = nx.clustering(G)
    features = []
    for u in nodes:
        seen = {u} | set(G[u])
        for v in G[u]:
            seen |= set(G[v])
        features.append([deg[u], core[u], pr[u], bc[u], clu[u], len(seen) - 1])
    return features


def zscore(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).tolist()


def rank_fraction(values):
    ranks = average_ranks(values)
    return [(r - 1.0) / max(len(values) - 1, 1) for r in ranks]


def load_label(network, p, repeats):
    pstr = ("%g" % p)
    path = os.path.join(
        RESULT_DIR,
        "sir_labels_%s_p%s_r%d.csv" % (network, pstr, repeats)
    )
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["node"]] = float(row["mean_size"])
    return out


# ---------------- Figure 1 ----------------
def figure1():
    print("== Figure 1: email-Eu-core 多 p 扫描 ==")
    G = load_email()
    nodes = list(G.nodes())
    X_all = build_features(G, nodes)
    ps = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06]
    R = 500
    seed = 2026
    rows = []

    rng = random.Random(seed)
    perm = list(range(len(nodes)))
    rng.shuffle(perm)
    cut = int(len(perm) * 0.8)
    train_idx = perm[:cut]
    test_idx = perm[cut:]

    deg = dict(G.degree())
    core = nx.core_number(G)
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)
    baselines = {
        "degree": [deg[u] for u in nodes],
        "k-core": [core[u] for u in nodes],
        "PageRank": [pr[u] for u in nodes],
        "betweenness": [bc[u] for u in nodes],
    }

    for p in ps:
        print(f"  p = {p} ...")
        labels = make_labels(G, p, R, seed=seed)
        y = [labels[u] for u in nodes]
        Xtr = [X_all[i] for i in train_idx]
        ytr = [y[i] for i in train_idx]
        Xte = [X_all[i] for i in test_idx]
        yte = [y[i] for i in test_idx]

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        pred_xgb = list(model.predict(Xte))

        preds = {"XGBoost": pred_xgb}
        for name, vals in baselines.items():
            preds[name] = [vals[i] for i in test_idx]
        true_te = [y[i] for i in test_idx]

        for name, pred in preds.items():
            rho, nd = eval_metrics(pred, true_te, 100)
            rows.append([p, name, round(rho, 4), round(nd, 4)])
        xgb_row = next(r for r in rows
                       if r[0] == p and r[1] == "XGBoost")
        print(f"    XGBoost: rho={xgb_row[2]:.4f}, NDCG={xgb_row[3]:.4f}")

    with open(os.path.join(RESULT_DIR, "fig1_p_sweep.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["p", "method", "spearman", "ndcg@100"])
        w.writerows(rows)

    colors = {
        "degree": "#1f77b4", "k-core": "#ff7f0e", "PageRank": "#2ca02c",
        "betweenness": "#d62728", "XGBoost": "#9467bd",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name in ["degree", "k-core", "PageRank", "betweenness", "XGBoost"]:
        pts = [r for r in rows if r[1] == name]
        xs = [r[0] for r in pts]
        for ax, metric in zip(axes, [2, 3]):
            ax.plot(xs, [r[metric] for r in pts], "-o", ms=3.5,
                    color=colors[name], label=name)
    axes[0].set_xlabel("p = β/(β+μ)")
    axes[0].set_ylabel("Spearman correlation")
    axes[1].set_xlabel("p = β/(β+μ)")
    axes[1].set_ylabel("NDCG@100")
    axes[0].set_title("(a) Spearman vs p")
    axes[1].set_title("(b) NDCG@100 vs p")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, "Figure1_spearman_ndcg_vs_p.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print("  saved", out_png)


# ---------------- Figure 2 ----------------
def figure2():
    print("== Figure 2: email-Eu-core 免疫预算实验 ==")
    G = load_email()
    nodes = list(G.nodes())
    labels = load_label("email-Eu-core", 0.025, 500)
    y = [labels[u] for u in nodes]

    # 跨网络 XGBoost 排名：在其余 3 个网络训练，预测 email
    train_nets = ["Facebook", "US-power-grid", "OpenFlights"]
    params = {"Facebook": (0.012, 200), "US-power-grid": (0.35, 200),
              "OpenFlights": (0.03, 200)}
    edge_files = {
        "Facebook": "facebook_undirected.edges",
        "US-power-grid": "opsahl-powergrid_undirected.edges",
        "OpenFlights": "opsahl-openflights_undirected.edges",
    }
    Xtr_all, ytr_all = [], []
    for net in train_nets:
        netG = nx.read_edgelist(os.path.join(CLEANED, edge_files[net]))
        netG.remove_edges_from(nx.selfloop_edges(netG))
        net_nodes = list(netG.nodes())
        pl, rl = params[net]
        net_labels = load_label(net, pl, rl)
        use_nodes = [u for u in net_nodes if u in net_labels]
        X = build_features(netG, use_nodes)
        yv = [net_labels[u] for u in use_nodes]
        Xtr_all.extend(zscore(X))
        ytr_all.extend(rank_fraction(yv))
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=2026, n_jobs=-1,
    )
    model.fit(Xtr_all, ytr_all)
    X_email = zscore(build_features(G, nodes))
    pred = list(model.predict(X_email))

    deg = dict(G.degree())
    core = nx.core_number(G)
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)

    # 最终方法：RRF(XGBoost, k-core) 给出整体顺序，
    # 再用 k-core 保护前 L=100 名（与 10_topL_corrected.py 相同）。
    n_all = len(nodes)
    kcore_vals = [core[u] for u in nodes]

    def rrf_rank(score_lists, k=60):
        tot = [0.0] * n_all
        for scores in score_lists:
            order = sorted(range(n_all), key=lambda i: scores[i], reverse=True)
            rank = [0] * n_all
            for pos, i in enumerate(order, start=1):
                rank[i] = pos
            for i in range(n_all):
                tot[i] += 1.0 / (k + rank[i])
        return tot

    rrf_vals = rrf_rank([pred, kcore_vals])
    L = 200
    rrf_order = sorted(range(n_all), key=lambda i: rrf_vals[i], reverse=True)
    kc_order = sorted(range(n_all), key=lambda i: kcore_vals[i], reverse=True)
    top_rrf, top_kc = rrf_order[:L], kc_order[:L]
    set_rrf, set_kc = set(top_rrf), set(top_kc)
    top_segment = [i for i in top_rrf if i in set_kc] + \
                  [i for i in top_kc if i not in set_rrf] + \
                  [i for i in top_rrf if i not in set_kc]
    rest = [i for i in rrf_order if i not in set(top_segment)]
    final_order = top_segment + rest
    topL_scores = [0.0] * n_all
    for pos, i in enumerate(final_order):
        topL_scores[i] = n_all - pos

    rankings = {
        "degree": [deg[u] for u in nodes],
        "k-core": [core[u] for u in nodes],
        "PageRank": [pr[u] for u in nodes],
        "betweenness": [bc[u] for u in nodes],
        "TopL-corrected": topL_scores,
        "Random": [random.random() for _ in nodes],
    }

    budgets = [0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]
    repeats = 40
    n = len(nodes)
    idx = list(range(n))
    p = 0.025
    rng = random.Random(2026)

    # 边表换成“位置下标”，和 keep/remap 的整数下标一致
    # （G.edges() 里是字符串节点 ID，直接 remap.get 会全部落空）
    node_pos = {u: i for i, u in enumerate(nodes)}
    edge_pos = [(node_pos[u], node_pos[v]) for u, v in G.edges()]

    rows = []
    for name, scores in rankings.items():
        order = sorted(idx, key=lambda i: scores[i], reverse=True)
        print(f"  免疫曲线: {name}")
        for L in budgets:
            total = 0.0
            for _ in range(repeats):
                removed = set(order[:L])
                keep = [i for i in idx if i not in removed]
                remap = {i: j for j, i in enumerate(keep)}
                dsu = DSU(len(keep))
                for u, v in edge_pos:
                    a = remap.get(u, -1)
                    b = remap.get(v, -1)
                    if a >= 0 and b >= 0 and rng.random() < p:
                        dsu.union(a, b)
                s = 0.0
                for j in range(len(keep)):
                    s += dsu.size[dsu.find(j)]
                if keep:
                    total += s / len(keep)
            rows.append([name, L, round(total / repeats, 4)])

    with open(os.path.join(RESULT_DIR, "fig2_immunization.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget_L", "residual_mean_outbreak"])
        w.writerows(rows)

    colors = {
        "degree": "#1f77b4", "k-core": "#ff7f0e", "PageRank": "#2ca02c",
        "betweenness": "#d62728", "TopL-corrected": "#9467bd",
        "Random": "#7f7f7f",
    }
    plt.figure(figsize=(7, 5))
    for name in rankings:
        pts = [r for r in rows if r[0] == name]
        xs = [r[1] for r in pts]
        ys = [r[2] for r in pts]
        plt.plot(xs, ys, "-o", ms=3.5, color=colors[name], label=name)
    plt.xlabel("Immunization budget L")
    plt.ylabel("Mean residual outbreak size")
    plt.title("Targeted immunization on email-Eu-core (p = 0.025)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "Figure2_immunization_budget.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print("  saved", out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", choices=["1", "2", "both"], default="both")
    args = parser.parse_args()
    if args.fig in ("1", "both"):
        figure1()
    if args.fig in ("2", "both"):
        figure2()
