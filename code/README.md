# 论文实验代码（按步骤使用）

代码按顺序推进：

1. `01_load_and_stats.py`：数据加载 + 网络统计（已完成）
2. `02_sir_mc_simulation.py`：SIR 蒙特卡洛仿真（已完成）
3. `03_baseline_centrality.py`：基线中心性对比（已完成）
4. `04_xgboost_ranking.py`：XGBoost 训练与评估（练手版，已完成）
5. `05_cross_network_xgboost.py`：跨网络 XGBoost（论文正式版，已完成）
6. `06_leave_one_out_xgboost.py`：留一网络交叉验证（论文主表，现在这一份）
7. `07_figures.py`：Figure2/Figure3（论文编号）绘图脚本（已完成）
8. `08_fusion_leave_one_out.py`：XGBoost+k-core 排名融合（应对 Facebook top-100 问题）
9. `09_rrf_leave_one_out.py`：倒数排名融合 RRF（更稳的融合版本）
10. `10_topL_corrected.py`：Top-L 分段修正（前 100 由 k-core 保护）
11. `11_table2_fair_and_ablation.py`：Table 2 公平口径 + 消融 + L 稳定性
12. `12_gcn_same_framework_loo.py`：同框架 GCN 学习型基线
13. `13_mrcnn_loo.py`：M-RCNN（ESWA 2022）独立学习型基线适配
15. `15_sis_labels.py`：SIS 标签生成（辅助验证实验）
16. `16_sis_loo_validation.py`：SIS 留一网络交叉验证
17. `17_static_baselines_ci_domirank.py`：CI(l=2) + DomiRank 静态基线（Table 3 新增行）
18. `18_static_extend_table2_sis.py`：CI/DomiRank 补齐 Table 2 与 SIS Table 7（先校验原行一致再写入）
19. `19_complementarity_schematic.py`：Figure 1 全局精度-头部稳定性互补示意图

## 安装环境

打开命令行（Windows 按 Win+R，输入 `cmd`，回车），执行：

```bash
pip install networkx
```

## 运行第 1 份代码

```bash
cd C:\Users\DELL\Documents\Codex\2026-09-03\zha\outputs\code
python 01_load_and_stats.py
```

正常情况下会输出 5 个网络的统计表，并把完整结果保存到
`results/network_stats.csv`。

## 常见问题

- `No module named 'networkx'`：环境没装好，重跑 `pip install networkx`；
- `python 不是内部或外部命令`：安装 Python 时勾选“Add Python to PATH”，
  或改用 `py 01_load_and_stats.py`；
- 计算聚类系数稍慢属正常（最大网络 ca-AstroPh 可能需要 1 分钟左右）。

## 跑通后请把控制台输出发我

我会根据你看到的节点数/边数是否与下面一致，确认没问题后再给 SIR 仿真脚本：

| 网络 | 预期节点 | 预期无向边（清洗后） |
|---|---:|---:|
| email-Eu-core | 986 | 16064 |
| Facebook | 4039 | 88234 |
| ca-AstroPh | 18771 | 198050 |
| US-power-grid | 4941 | 6594 |
| OpenFlights | 2939 | 15677 |

> 为什么 email-Eu-core 不是官方写的 1005、ca-AstroPh 不是 18772？
> 官方统计包含 19（email）/1（ca-AstroPh）个“只有自环、没有普通连边”的孤立节点。
> 我们的清洗规则删除了自环，所以这些节点在清洗后的边表里不再出现，读出来自然少 19/1 个。
> 论文里请写“去掉自环后的可用网络”：email 986 节点 / 16064 边，ca-AstroPh 18771 节点 / 198050 边，
> 并在数据描述中注明原始数据集官方节点数为 1005 / 18772。

## 运行第 2 份代码（SIR 蒙特卡洛）

先确认第 1 份已经跑通，然后执行：

```bash
python 02_sir_mc_simulation.py
```

默认参数：email-Eu-core，p = 0.025，重复 50 次。
结果会保存到 `results/sir_labels_email-Eu-core_p0.025_r50.csv`。

如果第一次用 p=0.05 跑出“前 5 名都是同一个数”（例如全部 537.4），
说明 p 太大，网络几乎总是整体连通，节点没有区分度。请改用小 p：

```bash
python 02_sir_mc_simulation.py --prob 0.025 --repeats 100
```

email 网络建议 p 在 0.02–0.03；如果前几名仍然几乎相同，再往 0.015 方向调。
合理的结果是：前 5 名的数值不同且递减，而不是 5 个并列。

想换网络：

```bash
python 02_sir_mc_simulation.py --network Facebook --repeats 100
```

可用网络名：email-Eu-core、Facebook、ca-AstroPh、US-power-grid、OpenFlights。

说明：
- p 越大传播越强；建议每个网络试 2–3 个 p（例如 0.03、0.05、0.08）；
- 这个脚本得到的是“从每个节点出发的 SIR 平均最终感染规模”，正是后面 XGBoost 要学的标签；
- 跑完后把控制台显示的影响力前 5 节点发我，确认标签可用再进入第 3 份。

## 运行第 3 份代码（基线对比）

先确认第 2 份已生成标签文件，例如
`results/sir_labels_email-Eu-core_p0.025_r500.csv`，然后执行：

```bash
python 03_baseline_centrality.py
```

脚本会自动读取该标签文件，并计算：

- degree（度）
- k-core
- PageRank
- betweenness（介数）

每种方法输出 4 个指标：Spearman、Kendall、NDCG@100、Top-100 命中数。
数值越大，说明该传统中心性越接近 SIR 标准答案。

结果保存到 `results/baseline_comparison_email-Eu-core_p0.025_r500.csv`。
把控制台表格发我后，用同一张表加入你的 XGBoost 方法即可形成论文主表。

## 运行第 4 份代码（XGBoost，练手版）

先安装 xgboost：

```bash
pip install xgboost
```

然后运行：

```bash
python 04_xgboost_ranking.py
```

脚本会：

1. 用 6 个网络结构特征给节点“画像”（度、k-core、PageRank、介数、聚类系数、二阶邻居数）；
2. 用 SIR 标签训练 XGBoost；
3. 在测试节点上计算与 03 相同的 Spearman / Kendall / NDCG / Top 命中；
4. 保存预测结果和特征重要性。

⚠️ 练手版是“同一个网络随机划分训练/测试”，结果偏乐观，不能直接写进论文。
跑通后请把控制台“XGBoost”那一行发我；我会给跨网络训练/测试的正式版脚本。

## 运行第 5 份代码（跨网络 XGBoost，论文正式版）

先把其他 3 个网络的 SIR 标签生成出来（标定好的 p 已避免“全并列”）：

```bash
python 02_sir_mc_simulation.py --network Facebook --prob 0.012 --repeats 200
python 02_sir_mc_simulation.py --network US-power-grid --prob 0.35 --repeats 200
python 02_sir_mc_simulation.py --network OpenFlights --prob 0.03 --repeats 200
```

然后运行：

```bash
python 05_cross_network_xgboost.py
```

- 默认设计：

- 训练：email-Eu-core + Facebook；
- 测试（模型没见过的网络）：US-power-grid + OpenFlights；
- 特征与 04 相同，但先按每个网络自身做 z-score 标准化，标签改成“网络内相对名次”，避免不同网络量纲差异导致迁移失败；
- 输出每个测试网络上的 degree / k-core / PageRank / XGBoost 对比表。

结果保存到 `results/cross_network_results.csv`，这就是论文主表的雏形。

## 运行第 6 份代码（留一网络交叉验证）

4 个网络标签都齐全后运行：

```bash
python 06_leave_one_out_xgboost.py
```

默认使用 XGBoost 的排序目标 `rank:pairwise`（直接优化“关键节点排前面”），
这比回归目标更符合关键传播者识别任务。旧版回归目标可用：

```bash
python 06_leave_one_out_xgboost.py --loss reg
```

脚本依次把 email-Eu-core、Facebook、US-power-grid、OpenFlights 各留出一次，
每次用其余 3 个网络训练，输出该轮对比表。最后生成：

- `results/leave_one_out_detail_rank.csv`：每一轮的完整结果；
- `results/leave_one_out_summary_rank.csv`：论文主表（4 轮均值±标准差）。

运行耗时主要在介数中心性，第一次完整运行可能需要几分钟。

## 运行第 8 份代码（XGBoost + k-core 融合）

```bash
python 08_fusion_leave_one_out.py
```

输出：
- `results/leave_one_out_fusion_detail.csv`
- `results/leave_one_out_fusion_summary.csv`

如需要调整权重（默认 XGBoost 与 k-core 各 0.5）：

```bash
python 08_fusion_leave_one_out.py --w_xgb 0.7
```

## 运行第 9 份代码（RRF 倒数排名融合）

```bash
python 09_rrf_leave_one_out.py
```

输出：
- `results/leave_one_out_rrf_detail.csv`
- `results/leave_one_out_rrf_summary.csv`

如需把 degree 也加入融合：

```bash
python 09_rrf_leave_one_out.py --add_degree
```

## 运行第 10 份代码（Top-L 分段修正）

```bash
python 10_topL_corrected.py
```

前 L 名（默认 100）由 k-core 保护，其余节点按 RRF 排序。
可调整 L：

```bash
python 10_topL_corrected.py --L 100
```

输出：
- `results/leave_one_out_topL_detail.csv`
- `results/leave_one_out_topL_summary.csv`

## 运行第 11 份代码（Table 2 公平口径 + 消融 + L 稳定性）

```bash
python 11_table2_fair_and_ablation.py
```

只跑其中一部分：

```bash
python 11_table2_fair_and_ablation.py --part table2
python 11_table2_fair_and_ablation.py --part ablation
```

输出：
- `results/table2_fair_email.csv`：所有方法都在同一 20% 测试子集上评价；
- `results/ablation_detail.csv`、`results/ablation_summary.csv`：
  k-core / XGBoost / RRF / TopL-50 / TopL-100 / TopL-200 的留一结果。

完整运行需几分钟（含介数中心性与四轮训练）。

## 运行第 12 份代码（同框架 GCN）

先安装 PyTorch：

```bash
pip install torch
```

然后：

```bash
python 12_gcn_same_framework_loo.py
```

输出 `results/learning_baseline_gcn.csv`。

## 运行第 13 份代码（M-RCNN 独立基线）

M-RCNN 官方核心代码位于 `M-RCNN_official/`（Apache 2.0，Ou et al., ESWA 2022）。

```bash
pip install torch python-louvain
python 13_mrcnn_loo.py
```

如需加入 ca-AstroPh（很慢，可选）：

```bash
python 13_mrcnn_loo.py --with_caastro
```

输出：
- `results/mrcnn_baseline_detail.csv`
- `results/mrcnn_baseline_summary.csv`
