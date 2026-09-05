# -*- coding: utf-8 -*-
"""
15_SIS 传播标签生成（累计感染时间，多进程优化版）
================================================
模型：离散时间 SIS，节点影响力 = 从该节点初始感染后，
      在固定窗口 T 步内累计“感染人数-时间”的蒙特卡洛平均。

与 SIR 对齐：p = beta/(beta+mu) 使用相同取值，mu=0.2。

运行：
  A 规模试跑：python 15_sis_labels.py --network email-Eu-core --repeats 30
  B 规模正式：python 15_sis_labels.py --network Facebook --repeats 100

输出：results/sis_labels_<network>_p<p>_r<R>_T<T>.csv
"""

import os
import csv
import random
import time
import argparse
import multiprocessing as mp

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CLEANED = os.path.join(ROOT, "outputs", "datasets", "cleaned")
RESULT_DIR = os.path.join(ROOT, "outputs", "code", "results")

PARAMS = {
    "email-Eu-core": dict(p=0.025, file="email-Eu-core_undirected.edges"),
    "Facebook": dict(p=0.012, file="facebook_undirected.edges"),
    "US-power-grid": dict(p=0.35, file="opsahl-powergrid_undirected.edges"),
    "OpenFlights": dict(p=0.03, file="opsahl-openflights_undirected.edges"),
    "ca-AstroPh": dict(p=0.015, file="ca-AstroPh_undirected.edges"),
}


def load_int_graph(net):
    path = os.path.join(CLEANED, PARAMS[net]["file"])
    edges = []
    names = {}
    with open(path) as f:
        for ln in f:
            a, b = ln.split()
            if a not in names:
                names[a] = len(names)
            if b not in names:
                names[b] = len(names)
            edges.append((names[a], names[b]))
    n = len(names)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return list(names.keys()), adj


def simulate_one(node_args):
    adj, n, source, beta, mu, T, repeats, base_seed = node_args
    rng = random.Random(base_seed + source)
    state = bytearray(n)
    touched = []
    total = 0.0
    for _ in range(repeats):
        for x in touched:
            state[x] = 0
        touched.clear()
        state[source] = 1
        infected = [source]
        touched.append(source)
        area = 0.0
        for _ in range(T):
            area += len(infected)
            if not infected:
                break
            newly = []
            for i in infected:
                for j in adj[i]:
                    if state[j] == 0 and rng.random() < beta:
                        state[j] = 1
                        newly.append(j)
                        touched.append(j)
            survivors = []
            for i in infected:
                if rng.random() < mu:
                    state[i] = 0
                else:
                    survivors.append(i)
            infected = survivors + newly
        total += area
    return source, total / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="email-Eu-core")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    p = PARAMS[args.network]["p"]
    beta = args.mu * p / (1.0 - p)
    pstr = ("%g" % p)
    out_csv = os.path.join(
        RESULT_DIR,
        f"sis_labels_{args.network}_p{pstr}_r{args.repeats}_T{args.T}.csv",
    )
    if os.path.exists(out_csv):
        print("已存在，跳过：", out_csv)
        return

    names, adj = load_int_graph(args.network)
    n = len(names)
    workers = args.workers or max(1, mp.cpu_count() - 1)
    print("=" * 70)
    print(f"SIS: {args.network}, N={n}, p={p}, beta={beta:.5f}, mu={args.mu}, "
          f"R={args.repeats}, T={args.T}, workers={workers}")
    print("=" * 70)

    tasks = [
        (adj, n, source, beta, args.mu, args.T, args.repeats, args.seed)
        for source in range(n)
    ]
    t0 = time.time()
    labels = [0.0] * n
    done = 0
    with mp.Pool(workers) as pool:
        for source, val in pool.imap_unordered(simulate_one, tasks, chunksize=8):
            labels[source] = val
            done += 1
            if done % 200 == 0 or done == n:
                print(f"  {done}/{n}，用时 {time.time()-t0:.1f}s")

    vals = labels
    top = sorted(vals, reverse=True)[:10]
    print(f"\nmean={sum(vals)/n:.3f}, max={max(vals):.3f}, "
          f"不同取值={len(set(round(v,3) for v in vals))}")
    print("top10:", [round(x, 3) for x in top])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "mean_infected_time"])
        for u, v in zip(names, labels):
            w.writerow([u, round(v, 4)])
    print("保存：", out_csv, f"总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
