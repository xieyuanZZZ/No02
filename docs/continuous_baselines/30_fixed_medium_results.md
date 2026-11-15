# 30 credible_v3 固定场景中等规模实验

**结果**：`results/Continuous_Baseline_Comparison/credible_v3/fixed_medium/`

## 设置

- 12 个固定场景：4 开阔 + 8 静态障碍；
- 不再做逐次位置/障碍随机扰动；
- p=0/0.1/0.2；
- 每条件 2 seeds；
- 3 方法；
- 共 **216 条记录**。

## 总体结果

| 方法 | n | 安全完成 | 机器人碰撞 | 静态碰撞 | 完成均值 |
|---|---:|---:|---:|---:|---:|
| Ours | 72 | 0.542 | 0.292 | 0.000 | 31.57 s |
| ORCA | 72 | **0.806** | 0.153 | 0.000 | 38.40 s* |
| MPC-CBF | 72 | 0.569 | 0.139 | 0.229 | 29.60 s |

\* ORCA 完成均值含失败/超时样本。

## 分类结果

| 方法 | 开阔安全完成 | 静态障碍安全完成 |
|---|---:|---:|
| Ours | 0.375 | 0.625 |
| ORCA | 0.667 | **0.875** |
| MPC-CBF | 0.625 | 0.542 |

## 数据可视化

![固定12场景汇总](D:/No02_external_baseline_v1/results/Continuous_Baseline_Comparison/credible_v3/figures/v3_fixed_medium_summary.png)

## 结论

- 固定 12 场景下，Ours 仍只有 0.542 安全完成；
- ORCA 0.806 最高，MPC-CBF 0.569；
- Ours 在开阔场景只有 0.375，是主要短板；
- Ours 静态障碍 0.625，高于 MPC-CBF 0.542，但低于 ORCA 0.875；
- 这次去掉了随机路径变化，说明 Ours 的问题不只是“路径变化”，在固定复杂场景上也存在等待点不足。

## 动态演示

- `figures/anim_v3_fixed_A3.gif`
- `figures/anim_v3_fixed_B4.gif`
- `figures/anim_v3_fixed_B6.gif`
- `figures/anim_v3_fixed_B8.gif`

每组三栏为 Ours / ORCA / MPC-CBF，p=0.1。

## 限制

- 每条件 2 seeds，仍不是最终 20-seed 正式统计；
- ORCA 完成均值包含超时，不能当成功 Makespan；
- 这组数据说明固定场景下 Ours 仍落后于 ORCA，不能靠随机化解释。
