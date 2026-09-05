# -*- coding: utf-8 -*-
"""
13_M-RCNN 独立学习型基线（留一网络交叉验证适配版）
=================================================
方法出处：
  Ou Y, Guo Q, Xing J L, Liu J G. Identification of spreading influence nodes
  via multi-level structural attributes based on the graph convolutional
  network. Expert Systems with Applications, 203:117515, 2022.

官方代码（Apache 2.0）已放入 M-RCNN_official/，本脚本只做适配：
  1) 沿用官方的 Embeddings.main1 构造三通道 28×28 输入：
     通道1 = 邻居度之和，通道2 = 节点连接社团数，通道3 = k-core；
  2) 沿用官方 Models.CNN1 三通道卷积回归网络；
  3) 训练/测试改为我们的留一网络协议：
     训练网络 SIR 标签转为网络内相对名次，避免不同网络量纲不同。

依赖：
  pip install torch python-louvain

运行（四网络）：
  python 13_mrcnn_loo.py

包含 ca-AstroPh（时间较长，可选）：
  python 13_mrcnn_loo.py --with_caastro

输出：results/mrcnn_baseline_detail.csv
      results/mrcnn_baseline_summary.csv
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

import networkx as nx
import numpy as np


def ensure_deps():
    try:
        import torch  # noqa
    except ImportError:
        print("[!] 请先安装 torch：pip install torch")
        raise SystemExit(1)
    try:
        import community.community_louvain  # noqa
    except ImportError:
        print("[!] 请先安装 python-louvain：pip install python-louvain")
        raise SystemExit(1)


# ---------- 让官方 Embeddings.py 能 import 的最小 Utils 桩 ----------
import types

_utils_stub = types.ModuleType("Utils")


def neighbor_degree(G):
    degree = dict(G.degree())
    out = {}
    for node in G.nodes():
        out[node] = sum(degree[nb] for nb in G.adj[node])
    return out


def generate_subgraph(G, node_list):
    L = len(node_list)
    encode = dict(zip(node_list, list(range(L))))
    subgraph = nx.subgraph(G, node_list)
    G_sub = nx.Graph()
    G_sub.add_edges_from((encode[i], encode[j]) for i, j in subgraph.edges())
    A = np.zeros([L, L])
    for i in range(L):
        for j in range(L):
            if G_sub.has_edge(i, j) and i != j:
                A[i, j] = 1
    return G_sub, A


def transform1(A, degree_list, com_list, shell_list):
    # 注意：官方 Embeddings.py 的 padding 分支会把结果赋进 torch.zeros，
    # torch 2.x 不再允许 numpy -> torch 切片赋值，所以这里直接返回 torch 张量。
    import torch
    B1, B2, B3 = A.copy(), A.copy(), A.copy()
    for (B, values) in ((B1, degree_list), (B2, com_list), (B3, shell_list)):
        B[0, 1:] = A[0, 1:] * np.array(values)[1:]
        B[1:, 0] = A[1:, 0] * np.array(values)[1:]
    for i in range(len(degree_list)):
        B1[i, i], B2[i, i], B3[i, i] = degree_list[i], com_list[i], shell_list[i]
    B = np.stack([B1, B2, B3], axis=0).astype(np.float32)
    return torch.from_numpy(B)


_utils_stub.neighbor_degree = neighbor_degree
_utils_stub.generate_subgraph = generate_subgraph
_utils_stub.transform1 = transform1
sys.modules["Utils"] = _utils_stub


# ---------- 载入官方 Embeddings 与 Models ----------
OFFICIAL_DIR = os.path.join(ROOT, "outputs", "code", "M-RCNN_official")
sys.path.insert(0, OFFICIAL_DIR)

_emb_spec = importlib.util.spec_from_file_location(
    "mrcnn_embeddings", os.path.join(OFFICIAL_DIR, "Embeddings.py"))
_emb = importlib.util.module_from_spec(_emb_spec)
_emb_spec.loader.exec_module(_emb)

import Models as MRC_Models


def community_number(G):
    import community.community_louvain as community
    partition = community.best_partition(G)
    com_num = {}
    for node in G.nodes():
        ids = {partition[node]}
        for nb in G.adj[node]:
            ids.add(partition[nb])
        com_num[node] = len(ids)
    return com_num


PARAMS = {
    "email-Eu-core": dict(prob=0.025, repeats=500),
    "Facebook": dict(prob=0.012, repeats=200),
    "US-power-grid": dict(prob=0.35, repeats=200),
    "OpenFlights": dict(prob=0.03, repeats=200),
    "ca-AstroPh": dict(prob=0.015, repeats=200),
}
EDGE_FILES = {
    "email-Eu-core": "email-Eu-core_undirected.edges",
    "Facebook": "facebook_undirected.edges",
    "US-power-grid": "opsahl-powergrid_undirected.edges",
    "OpenFlights": "opsahl-openflights_undirected.edges",
    "ca-AstroPh": "ca-AstroPh_undirected.edges",
}


def load_label(net):
    p = PARAMS[net]["prob"]
    r = PARAMS[net]["repeats"]
    pstr = ("%g" % p)
    path = os.path.join(
        m06.RESULT_DIR, "sir_labels_%s_p%s_r%d.csv" % (net, pstr, r))
    if not os.path.exists(path):
        print(f"[!] 缺少 SIR 标签：{path}")
        print("    请先用 02 脚本生成对应标签。")
        raise SystemExit(1)
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["node"]] = float(row["mean_size"])
    return out


def prepare(net, L):
    G = nx.read_edgelist(os.path.join(m06.CLEANED_DIR, EDGE_FILES[net]))
    G.remove_edges_from(nx.selfloop_edges(G))
    labels = load_label(net)
    nodes = [u for u in G.nodes() if u in labels]
    y = [labels[u] for u in nodes]
    ranks = m06.average_ranks(y)
    y_rank = [(r - 1.0) / max(len(y) - 1, 1) for r in ranks]
    print(f"  构造 M-RCNN 输入：{net}（Louvain + 28×28 邻域）...")
    com_num = community_number(G)
    data_dict = _emb.main1(G, L, com_num)
    return {"G": G, "nodes": nodes, "y": y,
            "y_rank": y_rank, "data": data_dict}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--with_caastro", action="store_true")
    args = parser.parse_args()

    ensure_deps()
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    nets = ["email-Eu-core", "Facebook", "US-power-grid", "OpenFlights"]
    if args.with_caastro:
        nets.append("ca-AstroPh")

    print("=" * 76)
    print(f"M-RCNN 留一交叉验证：{len(nets)} 个网络")
    print("=" * 76)

    data = {net: prepare(net, args.L) for net in nets}
    detail = []

    for test_net in nets:
        train_nets = [n for n in nets if n != test_net]
        print(f"\n测试网络 = {test_net}，训练 = {train_nets}")

        Xs, ys = [], []
        for net in train_nets:
            d = data[net]
            Xs.extend([torch.as_tensor(d["data"][u], dtype=torch.float32)
                       for u in d["nodes"] if u in d["data"]])
            ys.extend([d["y_rank"][i] for i, u in enumerate(d["nodes"])
                       if u in d["data"]])

        X = torch.stack(Xs)
        Y = torch.tensor(ys, dtype=torch.float32).reshape(-1, 1)
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True)

        model = MRC_Models.CNN1(args.L)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
            if epoch % 25 == 0:
                print(f"  epoch {epoch}: loss={total_loss/len(loader):.5f}")

        test = data[test_net]
        model.eval()
        Xte = torch.stack([torch.as_tensor(test["data"][u],
                                           dtype=torch.float32)
                           for u in test["nodes"] if u in test["data"]])
        pred_all = []
        with torch.no_grad():
            for i in range(0, len(Xte), args.batch_size):
                pred_all.extend(model(Xte[i:i + args.batch_size]).squeeze(-1).tolist())
        m = m06.evaluate(pred_all, test["y"], 100)
        detail.append([test_net, "M-RCNN", round(m["spearman"], 4),
                       round(m["kendall"], 4), round(m["ndcg"], 4),
                       m["top_hit"]])
        print(f"  M-RCNN: Spearman={m['spearman']:.4f} Kendall={m['kendall']:.4f} "
              f"NDCG={m['ndcg']:.4f} Top={m['top_hit']}")

    detail_csv = os.path.join(m06.RESULT_DIR, "mrcnn_baseline_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["test_network", "method", "spearman", "kendall",
                    "ndcg_top", "top_hit"])
        w.writerows(detail)

    summary_csv = os.path.join(m06.RESULT_DIR, "mrcnn_baseline_summary.csv")
    sp = [r[2] for r in detail]
    ken = [r[3] for r in detail]
    nd = [r[4] for r in detail]
    hit = [r[5] for r in detail]
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "spearman", "kendall", "ndcg", "top_hit"])
        w.writerow(["M-RCNN",
                    f"{statistics.mean(sp):.4f}±{statistics.stdev(sp):.4f}",
                    f"{statistics.mean(ken):.4f}±{statistics.stdev(ken):.4f}",
                    f"{statistics.mean(nd):.4f}±{statistics.stdev(nd):.4f}",
                    f"{statistics.mean(hit):.1f}±{statistics.stdev(hit):.1f}"])
    print("\n文件：", detail_csv, " / ", summary_csv)


if __name__ == "__main__":
    main()
