# -*- coding: utf-8 -*-
"""
02_SIR 蒙特卡洛仿真（节点影响力标签）
=====================================
作用：
  对给定网络，从每个节点分别作为 SIR 初始感染者，重复多次，
  得到“每个节点的平均最终感染规模”，这就是后面训练模型的标签。

速度技巧（非常重要）：
  SIR 的最终感染规模有一个精确等价关系——
  每条边以概率 p = beta / (beta + mu) 保留（“开路”），
  从某个节点出发的最终感染规模 = 该节点在“开路网络”中的连通分量大小。
  所以我们不需要真的“一天一天地传播”，
  每次随机生成一个开路网络，一次就能得到所有节点的最终规模，
  重复 R 次取平均即可。

运行前安装（如果第 1 份已装过 networkx，只装一次）：
    pip install networkx

运行示例（默认用小网络 email-Eu-core，速度快）：
    python 02_sir_mc_simulation.py

换网络/换参数：
    python 02_sir_mc_simulation.py --network Facebook --prob 0.05 --repeats 50

参数含义：
  --prob     ：上面公式里的 p。p 越大，疫情越容易扩散，越接近“全网感染”；
               建议从 0.03–0.10 试几个值。
  --repeats  ：蒙特卡洛重复次数，论文里常见 100–1000；
               第一次跑建议 30–50，确认没问题再加。

输出：
  results/sir_labels_<网络名>_p<概率>_r<次数>.csv
  每一行：node（节点编号）, mean_size（该节点平均最终感染规模）
"""

import os
import csv
import random
import argparse

import networkx as nx


# ---------- 0. 路径 ----------
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


class UnionFind:
    """并查集：用来找“开路网络”里每个节点属于哪个连通分量。"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        # 路径压缩：让下次查找更快
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def load_adjacency(filename):
    """读边表，返回：
       nodes   ：连续整数编号 0..N-1（论文里重新编号没影响）
       edge_list：边列表（(u, v)）
    """
    path = os.path.join(CLEANED_DIR, filename)
    G = nx.read_edgelist(path)
    # 去掉自环（若有），保留简单无向图
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    edge_list = [(idx[u], idx[v]) for u, v in G.edges()]
    print(f"节点数 N = {len(nodes)}，边数 M = {len(edge_list)}")
    return nodes, edge_list


def one_percolation(n, edge_list, prob, rng):
    """一次随机试验：每条边以概率 prob 保留，返回每个节点的分量大小。"""
    uf = UnionFind(n)
    for u, v in edge_list:
        if rng.random() < prob:      # 这条边“通了”
            uf.union(u, v)
    sizes = [0] * n
    for i in range(n):
        sizes[i] = uf.size[uf.find(i)]
    return sizes


def main():
    parser = argparse.ArgumentParser(description="SIR final-size Monte Carlo")
    parser.add_argument("--network", default="email-Eu-core",
                        choices=sorted(NETWORKS.keys()))
    parser.add_argument("--prob", type=float, default=0.025,
                        help="edge-open probability p = beta/(beta+mu); email 网络建议 0.02-0.03")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    print("=" * 60)
    print(f"网络：{args.network}")
    print(f"p = {args.prob}，重复次数 = {args.repeats}")
    print("=" * 60)

    nodes, edge_list = load_adjacency(NETWORKS[args.network])
    n = len(nodes)
    total = [0.0] * n
    rng = random.Random(args.seed)

    for rep in range(1, args.repeats + 1):
        sizes = one_percolation(n, edge_list, args.prob, rng)
        for i in range(n):
            total[i] += sizes[i]
        if rep % 10 == 0 or rep == args.repeats:
            print(f"  已完成 {rep}/{args.repeats}")

    mean_size = [total[i] / args.repeats for i in range(n)]

    # 保存结果
    out_csv = os.path.join(
        RESULT_DIR,
        f"sir_labels_{args.network}_p{args.prob}_r{args.repeats}.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "mean_size"])
        for original_id, m in zip(nodes, mean_size):
            writer.writerow([original_id, round(m, 4)])

    # 简单看一下结果
    order = sorted(range(n), key=lambda i: mean_size[i], reverse=True)
    print("\n影响力最大的 5 个节点（原始编号, 平均感染规模）：")
    for rank, i in enumerate(order[:5], 1):
        print(f"  {rank}. 节点 {nodes[i]} -> {mean_size[i]:.1f}")
    print("\n结果文件：", out_csv)
    print("\n下一步：把输出里的 5 个节点编号发我，确认没问题后给基线对比脚本。")


if __name__ == "__main__":
    main()
