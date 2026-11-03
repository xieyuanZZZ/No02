# S5 完整对照实验报告

- 阶段：`S5_evaluation`
- 运行目录：`D:\paper2\results\S5_evaluation\20261103_030006_s5_benchmark_afed08`
- 生成时间：2026-11-03T03:00:18+08:00
- 耗时：0.7 s

## 实验设计

- 任务：4 个（4 类地图 × [2] 台），每格 1 个；
- 方法：B0, A3（共享同一 Diffusion 权重、同一噪声、20 步 DDIM、相同引导预算）；
- 执行：每份规划在 p∈[0.0, 0.1] 下各做 30 次独立随机执行；
- 成功口径：规划有效 + 执行中全部到达 + 执行全程无碰撞（失败规划计 0）；
- 置信区间：以任务为簇的 bootstrap（2000 次重采样）。

## 结果

| 方法 | 规划成功率 | 平均耗时 [s] | 平均等待 | 平均硬风险 |
|---|---|---|---|---|
| B0 独立 Diffusion | 0.250 | 0.62 | 0.00 | 0.2085 |
| A3 首次碰撞 + 联合 | 0.250 | 1.45 | 0.00 | 0.2085 |

## 执行结果（p=0.1）

| 方法 | 安全完成率 | 95% 区间 | 碰撞率 | makespan [s] |
|---|---|---|---|---|
| B0 | 0.250 | [0.181, 0.334] | 0.217 | 14.20 |
| A3 | 0.250 | [0.181, 0.334] | 0.217 | 14.20 |

## 关键检验（G2：首次碰撞吸收是否不可替代）

- A3 vs B3（逐时刻期望风险）：安全完成率差 +nan，95% 区间 [+nan, +nan]，不显著；
- A3 安全完成率 0.250，B3 nan（p=0.1）。

## 结论与局限

- 本报告为**预实验规模**（配置 `benchmark.scale`），不可当作完整规模的最终结论；
- 若 A3 与 B3 差异不显著，则“首次碰撞吸收”的独立贡献尚未被证明（G2 未达成）；
- 所有失败均按 §10.2 口径计入分母，未做后验剔除。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T03:00:17+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | 9cc9e0661d9d4e27d119acc8f1c159f172230f4a |  |
| environment | `git_dirty` | True |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| plan | `B0/n_plans` | 4 |  |
| plan | `B0/plan_valid_rate` | 0.25 |  |
| plan | `B0/plan_time_mean_s` | 0.624488 |  |
| plan | `B0/wait_total_mean` | 0 |  |
| plan | `B0/hard_risk_mean` | 0.208538 |  |
| plan | `A3/n_plans` | 4 |  |
| plan | `A3/plan_valid_rate` | 0.25 |  |
| plan | `A3/plan_time_mean_s` | 1.45092 |  |
| plan | `A3/wait_total_mean` | 0 |  |
| plan | `A3/hard_risk_mean` | 0.208538 |  |
| execution/p0.0 | `B0/p0.0/n` | 120 |  |
| execution/p0.0 | `B0/p0.0/successes` | 30 |  |
| execution/p0.0 | `B0/p0.0/safe_completion_rate` | 0.25 |  |
| execution/p0.0 | `B0/p0.0/wilson_lo` | 0.181097 |  |
| execution/p0.0 | `B0/p0.0/wilson_hi` | 0.334413 |  |
| execution/p0.0 | `B0/p0.0/collision_rate` | 0.25 |  |
| execution/p0.0 | `B0/p0.0/makespan_mean_s` | 12.4 |  |
| execution/p0.0 | `B0/p0.0/makespan_p90_s` | 12.4 |  |
| execution/p0.1 | `B0/p0.1/n` | 120 |  |
| execution/p0.1 | `B0/p0.1/successes` | 30 |  |
| execution/p0.1 | `B0/p0.1/safe_completion_rate` | 0.25 |  |
| execution/p0.1 | `B0/p0.1/wilson_lo` | 0.181097 |  |
| execution/p0.1 | `B0/p0.1/wilson_hi` | 0.334413 |  |
| execution/p0.1 | `B0/p0.1/collision_rate` | 0.216667 |  |
| execution/p0.1 | `B0/p0.1/makespan_mean_s` | 14.1967 |  |
| execution/p0.1 | `B0/p0.1/makespan_p90_s` | 15.01 |  |
| execution/p0.0 | `A3/p0.0/n` | 120 |  |
| execution/p0.0 | `A3/p0.0/successes` | 30 |  |
| execution/p0.0 | `A3/p0.0/safe_completion_rate` | 0.25 |  |
| execution/p0.0 | `A3/p0.0/wilson_lo` | 0.181097 |  |
| execution/p0.0 | `A3/p0.0/wilson_hi` | 0.334413 |  |
| execution/p0.0 | `A3/p0.0/collision_rate` | 0.25 |  |
| execution/p0.0 | `A3/p0.0/makespan_mean_s` | 12.4 |  |
| execution/p0.0 | `A3/p0.0/makespan_p90_s` | 12.4 |  |
| execution/p0.1 | `A3/p0.1/n` | 120 |  |
| execution/p0.1 | `A3/p0.1/successes` | 30 |  |
| execution/p0.1 | `A3/p0.1/safe_completion_rate` | 0.25 |  |
| execution/p0.1 | `A3/p0.1/wilson_lo` | 0.181097 |  |
| execution/p0.1 | `A3/p0.1/wilson_hi` | 0.334413 |  |
| execution/p0.1 | `A3/p0.1/collision_rate` | 0.216667 |  |
| execution/p0.1 | `A3/p0.1/makespan_mean_s` | 14.1967 |  |
| execution/p0.1 | `A3/p0.1/makespan_p90_s` | 15.01 |  |

## 产物

- `tables\plans.csv`
- `tables\executions.csv`
- `plots\safe_completion_vs_pause.png`
- `plots\plan_time_and_makespan.png`
- `plots\safety_efficiency_scatter.png`
- `plots\failure_attribution.png`
