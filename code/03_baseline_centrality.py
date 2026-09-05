# -*- coding: utf-8 -*-
"""
03_基线中心性对比
=================
作用：
  把传统中心性方法当作“参赛选手”，看它们的排序和 SIR 标准答案有多接近：
    - 度中心性 degree
    - k-core
    - PageRank
    - 介数中心性 betweenness

评价指标（都自己实现，不需要额外安装 scipy）：
  Spearman  ：排序相关性，范围 [-1,1]，越大越接近 SIR 排序；
  Kendall   ：另一种排序相关性；
  NDCG@L    ：只看前 L 名排序质量（论文最常用）；
  Top-L 命中：前 L 名里有多少个真的是 SIR 前 L 名。

运行前：必须先跑过 02，生成了 SIR 标签 CSV。

运行（自动读取 02 生成的标签文件）：
    python 03_baseline_centrality.py

如果你的标签文件是别的参数：
    python 03_baseline_centrality.py --prob 0.025 --repeats 500

输出：
  results/baseline_comparison_<网络>_p<概率>_r<次数>.csv
"""

import os
import csv
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


# ---------- 统计工具（不依赖 scipy） ----------
def average_ranks(values):
    """把数值转成名次，并列的取平均名次（统计里的标准做法）。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
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
    """斯皮尔曼等级相关。"""
    rx = average_ranks(x)
    ry = average_ranks(y)
    n = len(x)
    dx = [a - (n + 1) / 2.0 for a in rx]
    dy = [a - (n + 1) / 2.0 for a in ry]
    denom = (sum(d * d for d in dx) * sum(d * d for d in dy)) ** 0.5
    if denom == 0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denom


def kendall_tau(x, y):
    """Kendall tau-b（处理并列的简化版）。"""
    n = len(x)
    conc, disc = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = (x[i] > x[j]) - (x[i] < x[j])
            sy = (y[i] > y[j]) - (y[i] < y[j])
            if sx * sy > 0:
                conc += 1
            elif sx * sy < 0:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2.0)


def dcg_at_k(order, relevance, k):
    return sum(relevance[order[i]] / (i + 2) ** 0.5 for i in range(min(k, len(order))))


def ndcg_at_k(pred_order, relevance, k):
    """预测排序 pred_order 的 NDCG@k（relevance 越大越重要）。"""
    dcg = dcg_at_k(pred_order, relevance, k)
    ideal = sorted(range(len(relevance)), key=lambda i: relevance[i], reverse=True)
    idcg = dcg_at_k(ideal, relevance, k)
    return dcg / idcg if idcg > 0 else 0.0


def top_overlap(pred_order, true_order, k):
    return len(set(pred_order[:k]) & set(true_order[:k]))


# ---------- 数据读取 ----------
def load_network(filename):
    G = nx.read_edgelist(os.path.join(CLEANED_DIR, filename))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def load_sir_labels(network, prob, repeats):
    path = os.path.join(
        RESULT_DIR,
        f"sir_labels_{network}_p{prob}_r{repeats}.csv",
    )
    if not os.path.exists(path):
        print(f"[!] 找不到 {path}")
        print("    请先运行 02_sir_mc_simulation.py 生成标签，再运行本脚本。")
        raise SystemExit(1)
    node2size = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node2size[row["node"]] = float(row["mean_size"])
    return node2size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="email-Eu-core",
                        choices=sorted(NETWORKS.keys()))
    parser.add_argument("--prob", type=float, default=0.025)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--topk", type=int, default=100,
                        help="NDCG 和命中率只看前多少名")
    args = parser.parse_args()

    print("=" * 70)
    print(f"基线对比：{args.network}（p={args.prob}，SIR 重复 {args.repeats} 次）")
    print("=" * 70)

    G = load_network(NETWORKS[args.network])
    labels = load_sir_labels(args.network, args.prob, args.repeats)
    nodes = list(G.nodes())
    n = len(nodes)
    print(f"图节点数 N = {n}，标签数 = {len(labels)}")

    # 只对“有 SIR 标签的节点”比较
    nodes = [u for u in nodes if u in labels]
    relevance = [labels[u] for u in nodes]

    # 计算各基线
    methods = {}
    deg = nx.degree(G)
    methods["degree"] = [deg[u] for u in nodes]

    core = nx.core_number(G)
    methods["k-core"] = [core[u] for u in nodes]

    pr = nx.pagerank(G)
    methods["PageRank"] = [pr[u] for u in nodes]

    try:
        bc = nx.betweenness_centrality(G)
        methods["betweenness"] = [bc[u] for u in nodes]
    except Exception as exc:
        print("介数中心性计算失败，跳过：", exc)

    # 真实 SIR 排序（标准答案）
    true_order = sorted(range(n), key=lambda i: relevance[i], reverse=True)

    rows = []
    print(
        f"\n{'方法':<14}{'Spearman':>10}{'Kendall':>9}{'NDCG@':>9}"
        f"{'Top命中':>9}"
    )
    print("-" * 56)
    for name, vals in methods.items():
        # 该方法的预测排序：中心性从大到小
        pred_order = sorted(range(n), key=lambda i: vals[i], reverse=True)
        rho = spearman(vals, relevance)
        tau = kendall_tau(vals, relevance)
        ndcg = ndcg_at_k(pred_order, relevance, args.topk)
        hit = top_overlap(pred_order, true_order, args.topk)
        rows.append([name, rho, tau, ndcg, hit])
        print(
            f"{name:<14}{rho:>10.4f}{tau:>9.4f}{ndcg:>9.4f}{hit:>9}"
        )

    out_csv = os.path.join(
        RESULT_DIR,
        f"baseline_comparison_{args.network}_p{args.prob}_r{args.repeats}.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "spearman", "kendall", "ndcg_top%d" % args.topk,
                         "top%d_hit" % args.topk])
        writer.writerows(rows)

    # 打印每个基线自己的前 10 名，方便直观检查
    print("\n各基线自己认为最重要的前 10 个节点：")
    for name, vals in methods.items():
        order = sorted(range(n), key=lambda i: vals[i], reverse=True)
        top_ids = [nodes[i] for i in order[:10]]
        print(f"  {name:<14}: {', '.join(map(str, top_ids))}")

    print("\n结果文件：", out_csv)
    print("下一步：把这个表发我，确认后用同样的表格式加入你的 XGBoost 方法。")


if __name__ == "__main__":
    main()
