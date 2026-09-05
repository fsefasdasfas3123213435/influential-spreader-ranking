# -*- coding: utf-8 -*-
"""
12_同框架 GCN 回归基线（留一网络交叉验证）
==========================================
目的：
  用与 XGBoost 完全相同的输入（网络结构特征、SIR 相对名次标签、
  z-score 标准化、留一网络划分），把“学习器”换成两层 GCN，
  证明 TopL-200 的增益来自排名融合与 k-core 保护，
  而不是仅仅因为选择了 XGBoost。

GCN 定义（基线，不用 RRF/TopL）：
  2 层图卷积回归，对称归一化邻接矩阵 A~ = D^-1/2 (A+I) D^-1/2，
  输出每个节点的相对影响力名次预测。

运行前安装 PyTorch（若已装 torch 则跳过）：
  pip install torch

运行：python 12_gcn_same_framework_loo.py
输出：results/learning_baseline_gcn.csv
"""

import os
import sys
import csv
import importlib.util
import argparse

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


def preprocess_adj(G, nodes):
    """返回归一化邻接矩阵的稀疏表示（自环 + 对称归一化）。"""
    n = len(nodes)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="coo").astype(np.float32)
    A = A.toarray()  # 稀疏 + 稠密相加会退化成 ndarray，这里统一转稠密
    A = A + np.eye(n, dtype=np.float32)  # 加自环
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.power(deg, -0.5)
    dinv[np.isinf(dinv)] = 0.0
    D = np.diag(dinv)
    A_norm = D @ A @ D
    return A_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("[!] 需要安装 PyTorch：pip install torch")
        raise SystemExit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class GCNRegressor(nn.Module):
        def __init__(self, in_dim, hidden, out_dim=1):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hidden)
            self.lin2 = nn.Linear(hidden, out_dim)

        def forward(self, x, A):
            h = F.relu(A @ self.lin1(x))
            return (A @ self.lin2(h)).reshape(-1)

    data = {net: m06.prepare_network(net) for net in m06.NETWORKS}
    # 把特征/numpy 数组与邻接一次性准备好
    for net, d in data.items():
        d["X"] = np.asarray(d["X"], dtype=np.float32)
        d["y_rank"] = np.asarray(d["y_rank"], dtype=np.float32)
        d["A"] = preprocess_adj(d["G"], d["nodes"])

    rows = []
    for test_net in m06.NETWORKS:
        train_nets = [n for n in m06.NETWORKS if n != test_net]
        print(f"\n测试网络 = {test_net}，训练 = {train_nets}")

        in_dim = data[test_net]["X"].shape[1]
        model = GCNRegressor(in_dim, args.hidden)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        loss_fn = nn.MSELoss()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for net in train_nets:
                d = data[net]
                x = torch.from_numpy(d["X"])
                A = torch.from_numpy(d["A"])
                y = torch.from_numpy(d["y_rank"])
                optimizer.zero_grad()
                pred = model(x, A)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
            if epoch % 50 == 0:
                print(f"  epoch {epoch}: train_loss={total_loss/len(train_nets):.5f}")

        model.eval()
        test = data[test_net]
        with torch.no_grad():
            pred = model(
                torch.from_numpy(test["X"]),
                torch.from_numpy(test["A"]),
            ).numpy()
        m = m06.evaluate(pred.tolist(), test["y"], 100)
        rows.append([test_net, "GCN(same-framework)",
                     round(m["spearman"], 4), round(m["kendall"], 4),
                     round(m["ndcg"], 4), m["top_hit"]])
        print(f"  GCN: Spearman={m['spearman']:.4f} Kendall={m['kendall']:.4f} "
              f"NDCG={m['ndcg']:.4f} Top={m['top_hit']}")

    out_csv = os.path.join(m06.RESULT_DIR, "learning_baseline_gcn.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["test_network", "method", "spearman", "kendall",
                    "ndcg_top", "top_hit"])
        w.writerows(rows)
    print("\n保存：", out_csv)


if __name__ == "__main__":
    main()
