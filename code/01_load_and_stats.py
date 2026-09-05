# -*- coding: utf-8 -*-
"""
01_数据加载 + 网络统计
======================
目的：
  1) 读取 datasets/cleaned/ 下的 5 个真实网络；
  2) 对每个网络统计：节点数、边数、平均度、密度、
     最大连通分量、平均聚类系数、度分布摘要等；
  3) 把结果保存成 CSV，方便以后写论文用。

第一次使用前安装（在命令行运行一次即可）：
    pip install networkx

运行方式（在命令行，先进入本文件所在目录）：
    python 01_load_and_stats.py

如果提示“No module named 'networkx'”，说明还没安装成功，
先执行上面那行 pip install networkx。
"""

import os
import csv

import networkx as nx


# ---------- 0. 路径设置 ----------
# 数据目录：改成你自己的绝对路径也行
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")

# 结果保存目录：和本脚本放在同一个文件夹下的 results/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------- 1. 要读入的网络 ----------
# 键：论文里显示的名字；值：cleaned/ 里的文件名
NETWORKS = {
    "email-Eu-core": "email-Eu-core_undirected.edges",
    "Facebook": "facebook_undirected.edges",
    "ca-AstroPh": "ca-AstroPh_undirected.edges",
    "US-power-grid": "opsahl-powergrid_undirected.edges",
    "OpenFlights": "opsahl-openflights_undirected.edges",
}


def load_graph(filename):
    """读取一行一个“节点A 节点B”的无向边文件，返回 networkx 图。

    如果节点编号都能转成整数就转成整数，否则保留字符串。
    """
    path = os.path.join(CLEANED_DIR, filename)
    print("正在读取:", path)

    # 先按整数节点读；一旦失败，就按字符串节点读
    try:
        G = nx.read_edgelist(path, nodetype=int)
    except Exception:
        G = nx.read_edgelist(path)
    return G


def largest_component(G):
    """返回最大连通分量（论文实验里通常只使用最大连通分量）。"""
    comps = list(nx.connected_components(G))
    giant = max(comps, key=len)
    return G.subgraph(giant).copy(), len(giant) / G.number_of_nodes()


def describe_degree_distribution(G):
    """返回度分布的简单摘要，方便判断网络是均匀还是异质。"""
    degs = sorted((d for _, d in G.degree()), reverse=True)
    n = len(degs)
    mean = sum(degs) / n
    variance = sum((d - mean) ** 2 for d in degs) / n
    return {
        "max_degree": degs[0],
        "mean_degree": round(mean, 4),
        "median_degree": degs[n // 2],
        "std_degree": round(variance ** 0.5, 4),
        "top10_degree": degs[:10],
    }


def analyze_one(name, filename):
    """对单个网络做全部统计，返回一行记录（字典）。"""
    G = load_graph(filename)
    G_giant, giant_ratio = largest_component(G)
    deg_info = describe_degree_distribution(G)

    # 聚类系数比较慢：ca-AstroPh 可能需要几十秒，先打印提示
    print(f"  正在计算聚类系数（{name}，稍等）...")
    avg_clustering = nx.average_clustering(G_giant)

    record = {
        "network": name,
        "N_nodes": G.number_of_nodes(),
        "M_edges": G.number_of_edges(),
        "avg_degree": round(2 * G.number_of_edges() / G.number_of_nodes(), 4),
        "density": round(nx.density(G), 8),
        "isolated_nodes": sum(1 for _, d in G.degree() if d == 0),
        "giant_size": G_giant.number_of_nodes(),
        "giant_edges": G_giant.number_of_edges(),
        "giant_ratio": round(giant_ratio, 4),
        "avg_clustering_giant": round(avg_clustering, 6),
        "max_degree": deg_info["max_degree"],
        "median_degree": deg_info["median_degree"],
        "mean_degree": deg_info["mean_degree"],
        "std_degree": deg_info["std_degree"],
        "degree_assortativity": round(nx.degree_assortativity_coefficient(G), 6),
    }
    return record


def main():
    print("=" * 70)
    print("开始分析 5 个真实网络")
    print("=" * 70)

    rows = []
    for name, fname in NETWORKS.items():
        print("-" * 70)
        try:
            rec = analyze_one(name, fname)
            rows.append(rec)
            print(f"  完成：{name}，节点 {rec['N_nodes']}，边 {rec['M_edges']}")
        except Exception as exc:
            print(f"  [!] {name} 分析失败：{exc}")

    # 保存总表
    out_csv = os.path.join(RESULT_DIR, "network_stats.csv")
    fieldnames = [
        "network", "N_nodes", "M_edges", "avg_degree", "density",
        "isolated_nodes", "giant_size", "giant_edges", "giant_ratio",
        "avg_clustering_giant", "max_degree", "median_degree",
        "mean_degree", "std_degree", "degree_assortativity",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 控制台简单打印
    print("\n" + "=" * 70)
    print("统计结果（完整版见 results/network_stats.csv）")
    print("=" * 70)
    header = f"{'network':<16}{'N':>7}{'M':>8}{'<k>':>8}{'giant%':>8}{'C':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['network']:<16}{r['N_nodes']:>7}{r['M_edges']:>8}"
            f"{r['avg_degree']:>8}{r['giant_ratio']:>8}{r['avg_clustering_giant']:>9}"
        )

    print("\n结果文件：", out_csv)
    print("完成。下一步可以让我给你 SIR 蒙特卡洛仿真脚本。")


if __name__ == "__main__":
    main()
