# -*- coding: utf-8 -*-
"""
19_complementarity_schematic.py
===============================
生成“全局排序精度—头部稳定性”权衡示意图（插入 6.2 节，Figure 1）。

横轴：4 个留一网络的均值 Spearman（全局排序质量）
纵轴：均值 NDCG@100（头部排序精度）
点坐标读取 Table 3 对应的结果文件：
  degree/k-core                    : leave_one_out_summary_rank.csv
  XGBoost/TopL-200                 : ablation_summary.csv
  CI                               : static_baselines_sir_summary.csv

运行：python 19_complementarity_schematic.py
输出：outputs/figures/Figure_global_head_tradeoff.png
"""

import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RESULT_DIR = os.path.join(ROOT, "outputs", "code", "results")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def first_value(text):
    return float(text.split("±")[0])


def main():
    loo = {}
    with open(os.path.join(RESULT_DIR,
                           "leave_one_out_summary_rank.csv"),
              newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            loo[row["method"]] = (first_value(row["spearman_mean±std"]),
                                  first_value(row["ndcg_mean±std"]))
    abl = {}
    with open(os.path.join(RESULT_DIR, "ablation_summary.csv"),
              newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            abl[row["method"]] = (first_value(row["spearman"]),
                                  first_value(row["ndcg"]))
    st = {}
    with open(os.path.join(RESULT_DIR,
                           "static_baselines_sir_summary.csv"),
              newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            st[row["method"]] = (first_value(row["spearman"]),
                                 first_value(row["ndcg"]))

    points = {
        "degree": loo["degree"],
        "k-core": loo["k-core"],
        "XGBoost": abl["XGBoost"],
        "CI (l=2)": st["CI"],
        "TopL-200": abl["TopL-200"],
    }
    for k, v in points.items():
        print(f"{k:<10} Spearman={v[0]:.4f}, NDCG@100={v[1]:.4f}")

    colors = {
        "degree": "#7f7f7f",
        "k-core": "#1f77b4",
        "XGBoost": "#d62728",
        "CI (l=2)": "#ff7f0e",
        "TopL-200": "#2ca02c",
    }
    sizes = {
        "degree": 140,
        "k-core": 200,
        "XGBoost": 200,
        "CI (l=2)": 200,
        "TopL-200": 360,
    }
    markers = {
        "degree": "o",
        "k-core": "o",
        "XGBoost": "o",
        "CI (l=2)": "o",
        "TopL-200": "*",
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=300)

    # 互补融合：k-core（头部稳定）与学习型（全局强）共同指向 TopL
    kc = points["k-core"]
    top = points["TopL-200"]
    xgb = points["XGBoost"]
    ci = points["CI (l=2)"]
    for src, c, lab in [(kc, "#1f77b4", "head protection"),
                        (xgb, "#d62728", "learned order below L"),
                        (ci, "#ff7f0e", "learned order below L")]:
        ax.annotate("", xy=(top[0] - 0.003, top[1] - 0.004),
                    xytext=(src[0] + 0.004, src[1] + 0.004),
                    arrowprops=dict(arrowstyle="->", color=c,
                                    lw=1.4, alpha=0.75,
                                    linestyle="--"))

    for name, (x, y) in points.items():
        ax.scatter(x, y, s=sizes[name], marker=markers[name],
                   color=colors[name], edgecolor="white", linewidth=1.2,
                   zorder=5, label=name)

    ax.set_xlim(0.740, 0.930)
    ax.set_ylim(0.740, 0.970)
    ax.set_xlabel("Overall ranking quality\n(mean Spearman over 4 folds)",
                  fontsize=11)
    ax.set_ylabel("Head-segment precision (mean NDCG@100)", fontsize=11)
    ax.set_title("Complementarity of global-oriented and head-stable rankings",
                 fontsize=12)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9)
    ax.text(0.002, 0.002,
            "Marker positions are means of Table 3 "
            "(4 leave-one-network-out folds).",
            transform=ax.transAxes, fontsize=7, color="#666666")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_png = os.path.join(FIG_DIR, "Figure_global_head_tradeoff.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print("saved", out_png)


if __name__ == "__main__":
    main()
