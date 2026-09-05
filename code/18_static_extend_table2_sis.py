# -*- coding: utf-8 -*-
"""
18_static_extend_table2_sis.py
===============================
把 CI(l=2) 与 DomiRank(sigma=0.5*sigma_c) 两个静态基线补进：
  * Table 2（email-Eu-core 内部公平口径，198 个测试节点）
  * SIS 表 7（四网络 SIS 留一，与现有 sis_loo_detail/summary 合并）

CI/DomiRank 只依赖网络拓扑，不需要重新训练模型。
脚本会先复算并校验原有行，数字完全一致后才更新结果文件；
任一校验不一致则中止写入，避免破坏已有结果。

运行：python 18_static_extend_table2_sis.py
"""

import os
import csv
import sys
import random
import statistics
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

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
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float).tocsr()
    n = A.shape[0]
    if n == 0:
        return {}
    lam_min = eigsh(A, k=1, which="SA", return_eigenvectors=False)[0]
    if lam_min >= 0:
        lam_min = -1.0
    sigma = 0.5 * (-1.0 / lam_min)
    M = eye(n, format="csr") + sigma * A
    rhs = A.dot(np.ones(n))
    gamma = sigma * spsolve(M.tocsc(), rhs)
    return {u: float(gamma[i]) for i, u in enumerate(nodes)}


def read_existing(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print("  写入：", path)


# ---------------- Table 2（email-Eu-core 公平口径） ----------------
def extend_table2():
    print("=" * 76)
    print("Part A：Table 2 公平口径追加 CI / DomiRank")
    print("=" * 76)

    G = nx.read_edgelist(os.path.join(
        m06.CLEANED_DIR, "email-Eu-core_undirected.edges"))
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

    def twohop(u):
        seen = {u} | set(G[u])
        for v in G[u]:
            seen |= set(G[v])
        return len(seen) - 1

    clu = nx.clustering(G)
    X_all = [[deg[u], core[u], pr[u], bc[u], clu[u],
              twohop(u)] for u in nodes]
    baseline = {
        "degree": [deg[u] for u in nodes],
        "k-core": [core[u] for u in nodes],
        "PageRank": [pr[u] for u in nodes],
        "betweenness": [bc[u] for u in nodes],
    }

    rng = random.Random(2026)
    perm = list(range(n))
    rng.shuffle(perm)
    cut = int(n * 0.8)
    train_idx, test_idx = perm[:cut], perm[cut:]
    test_y = [y[i] for i in test_idx]

    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=2026, n_jobs=-1,
    )
    model.fit([X_all[i] for i in train_idx], [y[i] for i in train_idx])
    pred_xgb = [float(v) for v in model.predict([X_all[i] for i in test_idx])]

    # 计算 CI / DomiRank（全网络拓扑分，再取测试节点）
    ci = ci_scores(G, nodes, radius=2)
    dr = domirank_scores(G, nodes)

    def rounded(m):
        return [round(m["spearman"], 4), round(m["kendall"], 4),
                round(m["ndcg"], 4), m["top_hit"]]

    computed = {}
    for name, vals in [("degree", baseline["degree"]),
                       ("k-core", baseline["k-core"]),
                       ("PageRank", baseline["PageRank"]),
                       ("betweenness", baseline["betweenness"])]:
        test_vals = [vals[i] for i in test_idx]
        computed[name] = rounded(m06.evaluate(test_vals, test_y, 100))
    computed["XGBoost"] = rounded(m06.evaluate(pred_xgb, test_y, 100))

    existing = read_existing(os.path.join(m06.RESULT_DIR,
                                          "table2_fair_email.csv"))
    old_map = {}
    for r in existing:
        old_map[r["method"]] = [
            float(r["spearman"]), float(r["kendall"]),
            float(r["ndcg_top100"]), int(r["top100_hit"])]

    for name in ["degree", "k-core", "PageRank", "betweenness", "XGBoost"]:
        c = [float(x) for x in computed[name]]
        o = old_map[name]
        if any(abs(a - b) > 1e-9 for a, b in zip(c, o)):
            print(f"  [check fail] {name}: computed {c} != existing {o}")
            raise SystemExit(2)
    print("  原有 5 行复算一致，继续写入。")

    def row(name, m):
        return {"method": name, "spearman": f"{m['spearman']:.4f}",
                "kendall": f"{m['kendall']:.4f}",
                "ndcg_top100": f"{m['ndcg']:.4f}",
                "top100_hit": m["top_hit"], "test_nodes": len(test_idx)}

    out = []
    order = ["degree", "k-core", "PageRank", "betweenness",
             "CI", "DomiRank", "XGBoost"]
    for name in order:
        if name in ("CI", "DomiRank"):
            vals_full = ci if name == "CI" else dr
            test_vals = [vals_full[nodes[i]] for i in test_idx]
            m = m06.evaluate(test_vals, test_y, 100)
        else:
            vals = baseline.get(name)
            if name == "XGBoost":
                m = m06.evaluate(pred_xgb, test_y, 100)
            else:
                test_vals = [vals[i] for i in test_idx]
                m = m06.evaluate(test_vals, test_y, 100)
        out.append(row(name, m))
        print(f"  {name:<12} Spearman={m['spearman']:.4f} "
              f"Kendall={m['kendall']:.4f} NDCG@100={m['ndcg']:.4f} "
              f"Top100={m['top_hit']}")

    header = ["method", "spearman", "kendall", "ndcg_top100",
              "top100_hit", "test_nodes"]
    write_csv(os.path.join(m06.RESULT_DIR, "table2_fair_email.csv"),
              header, out)


# ---------------- SIS 表 7（静态基线，不重训模型） ----------------
SIS_PARAMS = {
    "email-Eu-core": dict(p=0.025, repeats=100, T=50),
    "Facebook": dict(p=0.012, repeats=100, T=50),
    "US-power-grid": dict(p=0.35, repeats=100, T=50),
    "OpenFlights": dict(p=0.03, repeats=100, T=50),
}


def load_sis(net):
    cfg = SIS_PARAMS[net]
    pstr = "%g" % cfg["p"]
    path = os.path.join(m06.RESULT_DIR,
                        "sis_labels_%s_p%s_r%d_T%d.csv" %
                        (net, pstr, cfg["repeats"], cfg["T"]))
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["node"]] = float(row["mean_infected_time"])
    return out


def net_graph_nodes(net):
    G = nx.read_edgelist(os.path.join(
        m06.CLEANED_DIR, m06.EDGE_FILES[net]))
    G.remove_edges_from(nx.selfloop_edges(G))
    sir_path = m06.load_label_path(net)
    sir_keys = set()
    with open(sir_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sir_keys.add(row["node"])
    nodes = [u for u in G.nodes() if u in sir_keys]
    return G, nodes


def extend_sis():
    print("\n" + "=" * 76)
    print("Part B：SIS 表 7 追加 CI / DomiRank")
    print("=" * 76)

    new_rows = []
    for net in m06.NETWORKS:
        sis = load_sis(net)
        G, nodes = net_graph_nodes(net)
        y = [sis[u] for u in nodes]
        print(f"  {net}: N={len(nodes)}")
        ci = ci_scores(G, nodes, radius=2)
        dr = domirank_scores(G, nodes)
        for name, scores in [("CI", ci), ("DomiRank", dr)]:
            vals = [scores[u] for u in nodes]
            m = m06.evaluate(vals, y, 100)
            new_rows.append({
                "test_network": net, "method": name,
                "spearman": round(m["spearman"], 4),
                "kendall": round(m["kendall"], 4),
                "ndcg_top": round(m["ndcg"], 4),
                "top_hit": m["top_hit"],
            })
            print(f"    {name:<10} Spearman={m['spearman']:.4f} "
                  f"Kendall={m['kendall']:.4f} NDCG@100={m['ndcg']:.4f} "
                  f"Top100={m['top_hit']}")

    detail_path = os.path.join(m06.RESULT_DIR, "sis_loo_detail.csv")
    old_detail = read_existing(detail_path)
    method_order = ["degree", "k-core", "PageRank", "betweenness",
                    "CI", "DomiRank", "XGBoost", "TopL-200"]
    detail_out = []
    for net in m06.NETWORKS:
        net_rows = {r["method"]: r for r in old_detail
                    if r["test_network"] == net}
        if len(net_rows) != 6:
            print(f"  [check fail] {net}: expected 6 rows, got "
                  f"{len(net_rows)}")
            raise SystemExit(2)
        for method in method_order:
            if method in ("CI", "DomiRank"):
                src = [r for r in new_rows
                       if r["test_network"] == net and r["method"] == method][0]
                detail_out.append(src)
            else:
                r = dict(net_rows[method])
                r["spearman"] = float(r["spearman"])
                r["kendall"] = float(r["kendall"])
                r["ndcg_top"] = float(r["ndcg_top"])
                r["top_hit"] = int(r["top_hit"])
                detail_out.append(r)

    header = ["test_network", "method", "spearman", "kendall",
              "ndcg_top", "top_hit"]
    write_csv(detail_path, header, detail_out)

    summary_path = os.path.join(m06.RESULT_DIR, "sis_loo_summary.csv")
    summary_out = []
    print("\nSIS 汇总（4 网络均值 ± 标准差）")
    for method in method_order:
        rows = [r for r in detail_out if r["method"] == method]
        vals = {}
        for key, idx in [("spearman", "spearman"), ("kendall", "kendall"),
                         ("ndcg", "ndcg_top"), ("top_hit", "top_hit")]:
            xs = [r[idx] for r in rows]
            mean = statistics.mean(xs)
            std = statistics.stdev(xs) if len(xs) > 1 else 0.0
            vals[key] = (mean, std)
        summary_out.append({
            "method": method,
            "spearman": f"{vals['spearman'][0]:.4f}±"
                        f"{vals['spearman'][1]:.4f}",
            "kendall": f"{vals['kendall'][0]:.4f}±"
                       f"{vals['kendall'][1]:.4f}",
            "ndcg": f"{vals['ndcg'][0]:.4f}±{vals['ndcg'][1]:.4f}",
            "top_hit": f"{vals['top_hit'][0]:.1f}±"
                       f"{vals['top_hit'][1]:.1f}",
        })
        print(f"  {method:<12} Spearman={vals['spearman'][0]:.4f}±"
              f"{vals['spearman'][1]:.4f}  NDCG@100={vals['ndcg'][0]:.4f}±"
              f"{vals['ndcg'][1]:.4f}  Top100={vals['top_hit'][0]:.1f}±"
              f"{vals['top_hit'][1]:.1f}")
    write_csv(summary_path,
              ["method", "spearman", "kendall", "ndcg", "top_hit"],
              summary_out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    extend_table2()
    extend_sis()
