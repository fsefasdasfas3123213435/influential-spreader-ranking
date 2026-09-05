# -*- coding: utf-8 -*-
"""
17_static_baselines_ci_domirank.py
===================================
在 SIR 四网络留一主表上补充两个现代静态基线：
  * Collective Influence (CI_l, l=2)  Morone & Makse, Nature 2015
      CI_l(i) = (k_i - 1) * sum_{j in d-Ball(i,l)} (k_j - 1)
  * DomiRank  Engsig et al., Nature Communications 2024
      Gamma = sigma * (sigma*A + I)^{-1} * A * 1,  theta = 1
      收敛区间 sigma in (0, 1/(-lambda_min(A)))
      本文统一取 sigma = 0.5 * (-1/lambda_min(A))，不按网络额外调参。

这两个指标只依赖网络拓扑，不参与训练，因此可以直接在四个留一测试网络
的 SIR 标签上评估，口径与主表完全一致。

运行：
  python 17_static_baselines_ci_domirank.py

输出：
  results/static_baselines_sir_detail.csv   （逐网络明细）
  results/static_baselines_sir_summary.csv  （4 网络均值 ± 标准差）
"""

import os
import csv
import sys
import statistics
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")

# 复用 06 号脚本的加载、标签与评价函数，保证口径与论文主表完全一致。
_spec = importlib.util.spec_from_file_location(
    "m06", os.path.join(ROOT, "outputs", "code", "06_leave_one_out_xgboost.py")
)
m06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m06)

import networkx as nx
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import eigsh, spsolve


def ci_scores(G, nodes, radius=2):
    """CI_l(i) = (k_i - 1) * sum_{j in boundary ball(i,l)} (k_j - 1), l = 2."""
    deg = dict(G.degree())
    scores = {}
    for u in nodes:
        frontier = {u}
        visited = {u}
        for _ in range(radius):
            nxt = set()
            for x in frontier:
                nxt.update(G[x])
            nxt.difference_update(visited)
            visited.update(nxt)
            frontier = nxt
            if not frontier:
                break
        scores[u] = (deg[u] - 1.0) * sum(deg[v] - 1.0 for v in frontier)
    return scores


def domirank_scores(G, nodes):
    """DomiRank: Gamma = sigma*(sigma*A+I)^{-1} A 1, sigma = 0.5*(-1/lambda_min)."""
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float).tocsr()
    n = A.shape[0]
    if n == 0:
        return {}
    try:
        lam_min = eigsh(A, k=1, which="SA",
                        return_eigenvectors=False)[0]
    except Exception as exc:  # 极小概率收敛失败时用最小模特征值兜底
        print(f"    [warn] eigsh(SA) failed ({exc}); fallback to SM")
        lam_min = eigsh(A, k=1, which="SM",
                        return_eigenvectors=False)[0]
    if lam_min >= 0:
        # 正常无向网络（无自环、至少一条边）不会进入该分支，这里仅作保护。
        lam_min = -1.0
    sigma = 0.5 * (-1.0 / lam_min)
    M = eye(n, format="csr") + sigma * A
    rhs = A.dot(np.ones(n))
    gamma = sigma * spsolve(M.tocsc(), rhs)
    return {u: float(gamma[i]) for i, u in enumerate(nodes)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 76)
    print("CI(l=2) + DomiRank(sigma=0.5*sigma_c) 静态基线（SIR 四网络）")
    print("=" * 76)

    data = {}
    for net in m06.NETWORKS:
        print(f"  预处理 {net} ...")
        d = m06.prepare_network(net)
        print(f"    计算 CI / DomiRank（N={len(d['nodes'])}）...")
        ci = ci_scores(d["G"], d["nodes"], radius=2)
        dr = domirank_scores(d["G"], d["nodes"])
        d["baselines"]["CI"] = [ci[u] for u in d["nodes"]]
        d["baselines"]["DomiRank"] = [dr[u] for u in d["nodes"]]
        data[net] = d

    detail_rows = []
    for test_net in m06.NETWORKS:
        test = data[test_net]
        print(f"\n测试网络 = {test_net}")
        for name in ["CI", "DomiRank"]:
            vals = test["baselines"][name]
            m = m06.evaluate(vals, test["y"], 100)
            detail_rows.append([
                test_net, name,
                round(m["spearman"], 4), round(m["kendall"], 4),
                round(m["ndcg"], 4), m["top_hit"],
            ])
            print(f"  {name:<10} Spearman={m['spearman']:.4f} "
                  f"Kendall={m['kendall']:.4f} NDCG@100={m['ndcg']:.4f} "
                  f"Top100 hit={m['top_hit']}")

    detail_csv = os.path.join(RESULT_DIR, "static_baselines_sir_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["test_network", "method", "spearman", "kendall",
                    "ndcg_top", "top_hit"])
        w.writerows(detail_rows)

    print("\n" + "=" * 76)
    print("汇总：4 网络均值 ± 标准差")
    print("=" * 76)
    summary_rows = []
    for method in ["CI", "DomiRank"]:
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
        print(f"  {method:<10} Spearman={vals['spearman'][0]:.4f}±"
              f"{vals['spearman'][1]:.4f}  NDCG@100={vals['ndcg'][0]:.4f}±"
              f"{vals['ndcg'][1]:.4f}  Top100 hit={vals['top_hit'][0]:.1f}±"
              f"{vals['top_hit'][1]:.1f}")

    summary_csv = os.path.join(RESULT_DIR, "static_baselines_sir_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "spearman", "kendall", "ndcg", "top_hit"])
        w.writerows(summary_rows)

    print("\n明细文件：", detail_csv)
    print("汇总文件：", summary_csv)


if __name__ == "__main__":
    main()
