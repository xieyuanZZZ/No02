# S5 完整对照实验报告

- 阶段：`S5_evaluation`
- 运行目录：`D:\paper2\results\S5_evaluation\20261103_033410_s5_benchmark_a98f98`
- 生成时间：2026-11-03T03:46:41+08:00
- 耗时：0.9 s

## 实验设计

- 任务：36 个（4 类地图 × [2, 4, 6] 台），每格 3 个；
- 方法：B0, B1, B2, B3, A1, A2, A3（共享同一 Diffusion 权重、同一噪声、20 步 DDIM、相同引导预算）；
- 执行：每份规划在 p∈[0.0, 0.05, 0.1, 0.2] 下各做 30 次独立随机执行；
- 成功口径：规划有效 + 执行中全部到达 + 执行全程无碰撞（失败规划计 0）；
- 置信区间：以任务为簇的 bootstrap（2000 次重采样）。

## 结果

| 方法 | 规划成功率 | 平均耗时 [s] | 平均等待 | 平均硬风险 |
|---|---|---|---|---|
| B0 独立 Diffusion | 0.306 | 0.81 | 0.00 | 1.1070 |
| B1 同步距离引导 | 0.306 | 0.81 | 0.00 | 0.9924 |
| B2 固定时间窗引导 | 0.306 | 1.09 | 0.00 | 0.8783 |
| B3 逐时刻期望碰撞风险 | 0.306 | 5.08 | 6.32 | 0.4208 |
| A1 首次碰撞 + 只改路径 | 0.306 | 2.60 | 0.00 | 0.9203 |
| A2 首次碰撞 + 只改等待 | 0.306 | 2.59 | 3.87 | 0.6765 |
| A3 首次碰撞 + 联合 | 0.306 | 2.50 | 3.83 | 0.5611 |

## 执行结果（p=0.2）

| 方法 | 安全完成率 | 95% 区间 | 碰撞率 | makespan [s] |
|---|---|---|---|---|
| B0 | 0.186 | [0.164, 0.210] | 0.568 | 16.76 |
| B1 | 0.192 | [0.169, 0.216] | 0.556 | 16.76 |
| B2 | 0.192 | [0.169, 0.216] | 0.525 | 16.76 |
| B3 | 0.237 | [0.213, 0.263] | 0.383 | 17.36 |
| A1 | 0.192 | [0.169, 0.216] | 0.538 | 16.76 |
| A2 | 0.211 | [0.188, 0.236] | 0.469 | 17.13 |
| A3 | 0.216 | [0.192, 0.241] | 0.432 | 17.13 |

## 关键检验（G2：首次碰撞吸收是否不可替代）

- A3 vs B3（逐时刻期望风险）：安全完成率差 -0.0366，95% 区间 [-0.0963, -0.0002]，显著；
- A3 安全完成率 0.216，B3 0.237（p=0.2）。

## 结论与局限

- 本报告为**预实验规模**（配置 `benchmark.scale`），不可当作完整规模的最终结论；
- 若 A3 与 B3 差异不显著，则“首次碰撞吸收”的独立贡献尚未被证明（G2 未达成）；
- 所有失败均按 §10.2 口径计入分母，未做后验剔除。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T03:46:40+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | 41d9a81a58961779f75c107425a63a82c3171eae |  |
| environment | `git_dirty` | True |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| plan | `B0/n_plans` | 36 |  |
| plan | `B0/plan_valid_rate` | 0.305556 |  |
| plan | `B0/plan_time_mean_s` | 0.813873 |  |
| plan | `B0/wait_total_mean` | 0 |  |
| plan | `B0/hard_risk_mean` | 1.10697 |  |
| plan | `B1/n_plans` | 36 |  |
| plan | `B1/plan_valid_rate` | 0.305556 |  |
| plan | `B1/plan_time_mean_s` | 0.809737 |  |
| plan | `B1/wait_total_mean` | 0 |  |
| plan | `B1/hard_risk_mean` | 0.992405 |  |
| plan | `B2/n_plans` | 36 |  |
| plan | `B2/plan_valid_rate` | 0.305556 |  |
| plan | `B2/plan_time_mean_s` | 1.08568 |  |
| plan | `B2/wait_total_mean` | 0 |  |
| plan | `B2/hard_risk_mean` | 0.878279 |  |
| plan | `B3/n_plans` | 36 |  |
| plan | `B3/plan_valid_rate` | 0.305556 |  |
| plan | `B3/plan_time_mean_s` | 5.08363 |  |
| plan | `B3/wait_total_mean` | 6.31794 |  |
| plan | `B3/hard_risk_mean` | 0.420832 |  |
| plan | `A1/n_plans` | 36 |  |
| plan | `A1/plan_valid_rate` | 0.305556 |  |
| plan | `A1/plan_time_mean_s` | 2.6031 |  |
| plan | `A1/wait_total_mean` | 0 |  |
| plan | `A1/hard_risk_mean` | 0.920274 |  |
| plan | `A2/n_plans` | 36 |  |
| plan | `A2/plan_valid_rate` | 0.305556 |  |
| plan | `A2/plan_time_mean_s` | 2.59365 |  |
| plan | `A2/wait_total_mean` | 3.86585 |  |
| plan | `A2/hard_risk_mean` | 0.676472 |  |
| plan | `A3/n_plans` | 36 |  |
| plan | `A3/plan_valid_rate` | 0.305556 |  |
| plan | `A3/plan_time_mean_s` | 2.49608 |  |
| plan | `A3/wait_total_mean` | 3.83168 |  |
| plan | `A3/hard_risk_mean` | 0.561063 |  |
| execution/p0.0 | `B0/p0.0/n` | 1080 |  |
| execution/p0.0 | `B0/p0.0/successes` | 180 |  |
| execution/p0.0 | `B0/p0.0/safe_completion_rate` | 0.166667 |  |
| execution/p0.0 | `B0/p0.0/wilson_lo` | 0.145629 |  |
| execution/p0.0 | `B0/p0.0/wilson_hi` | 0.190067 |  |
| execution/p0.0 | `B0/p0.0/collision_rate` | 0.583333 |  |
| execution/p0.0 | `B0/p0.0/makespan_mean_s` | 12.4 |  |
| execution/p0.0 | `B0/p0.0/makespan_p90_s` | 12.4 |  |
| execution/p0.05 | `B0/p0.05/n` | 1080 |  |
| execution/p0.05 | `B0/p0.05/successes` | 194 |  |
| execution/p0.05 | `B0/p0.05/safe_completion_rate` | 0.17963 |  |
| execution/p0.05 | `B0/p0.05/wilson_lo` | 0.157883 |  |
| execution/p0.05 | `B0/p0.05/wilson_hi` | 0.203648 |  |
| execution/p0.05 | `B0/p0.05/collision_rate` | 0.566667 |  |
| execution/p0.05 | `B0/p0.05/makespan_mean_s` | 13.5796 |  |
| execution/p0.05 | `B0/p0.05/makespan_p90_s` | 14.1889 |  |
| execution/p0.1 | `B0/p0.1/n` | 1080 |  |
| execution/p0.1 | `B0/p0.1/successes` | 197 |  |
| execution/p0.1 | `B0/p0.1/safe_completion_rate` | 0.182407 |  |
| execution/p0.1 | `B0/p0.1/wilson_lo` | 0.160514 |  |
| execution/p0.1 | `B0/p0.1/wilson_hi` | 0.206552 |  |
| execution/p0.1 | `B0/p0.1/collision_rate` | 0.560185 |  |
| execution/p0.1 | `B0/p0.1/makespan_mean_s` | 14.5274 |  |
| execution/p0.1 | `B0/p0.1/makespan_p90_s` | 15.3478 |  |
| execution/p0.2 | `B0/p0.2/n` | 1080 |  |
| execution/p0.2 | `B0/p0.2/successes` | 201 |  |
| execution/p0.2 | `B0/p0.2/safe_completion_rate` | 0.186111 |  |
| execution/p0.2 | `B0/p0.2/wilson_lo` | 0.164026 |  |
| execution/p0.2 | `B0/p0.2/wilson_hi` | 0.210421 |  |
| execution/p0.2 | `B0/p0.2/collision_rate` | 0.567593 |  |
| execution/p0.2 | `B0/p0.2/makespan_mean_s` | 16.757 |  |
| execution/p0.2 | `B0/p0.2/makespan_p90_s` | 17.9944 |  |
| execution/p0.0 | `B1/p0.0/n` | 1080 |  |
| execution/p0.0 | `B1/p0.0/successes` | 210 |  |
| execution/p0.0 | `B1/p0.0/safe_completion_rate` | 0.194444 |  |
| execution/p0.0 | `B1/p0.0/wilson_lo` | 0.17194 |  |
| execution/p0.0 | `B1/p0.0/wilson_hi` | 0.219115 |  |
| execution/p0.0 | `B1/p0.0/collision_rate` | 0.555556 |  |
| execution/p0.0 | `B1/p0.0/makespan_mean_s` | 12.4 |  |
| execution/p0.0 | `B1/p0.0/makespan_p90_s` | 12.4 |  |
| execution/p0.05 | `B1/p0.05/n` | 1080 |  |
| execution/p0.05 | `B1/p0.05/successes` | 204 |  |
| execution/p0.05 | `B1/p0.05/safe_completion_rate` | 0.188889 |  |
| execution/p0.05 | `B1/p0.05/wilson_lo` | 0.166662 |  |
| execution/p0.05 | `B1/p0.05/wilson_hi` | 0.213321 |  |
| execution/p0.05 | `B1/p0.05/collision_rate` | 0.549074 |  |
| execution/p0.05 | `B1/p0.05/makespan_mean_s` | 13.5796 |  |
| execution/p0.05 | `B1/p0.05/makespan_p90_s` | 14.1889 |  |
| execution/p0.1 | `B1/p0.1/n` | 1080 |  |
| execution/p0.1 | `B1/p0.1/successes` | 208 |  |
| execution/p0.1 | `B1/p0.1/safe_completion_rate` | 0.192593 |  |
| execution/p0.1 | `B1/p0.1/wilson_lo` | 0.17018 |  |
| execution/p0.1 | `B1/p0.1/wilson_hi` | 0.217184 |  |
| execution/p0.1 | `B1/p0.1/collision_rate` | 0.543519 |  |
| execution/p0.1 | `B1/p0.1/makespan_mean_s` | 14.5274 |  |
| execution/p0.1 | `B1/p0.1/makespan_p90_s` | 15.3478 |  |
| execution/p0.2 | `B1/p0.2/n` | 1080 |  |
| execution/p0.2 | `B1/p0.2/successes` | 207 |  |
| execution/p0.2 | `B1/p0.2/safe_completion_rate` | 0.191667 |  |
| execution/p0.2 | `B1/p0.2/wilson_lo` | 0.1693 |  |
| execution/p0.2 | `B1/p0.2/wilson_hi` | 0.216219 |  |
| execution/p0.2 | `B1/p0.2/collision_rate` | 0.555556 |  |
| execution/p0.2 | `B1/p0.2/makespan_mean_s` | 16.757 |  |
| execution/p0.2 | `B1/p0.2/makespan_p90_s` | 17.9944 |  |
| execution/p0.0 | `B2/p0.0/n` | 1080 |  |
| execution/p0.0 | `B2/p0.0/successes` | 210 |  |
| execution/p0.0 | `B2/p0.0/safe_completion_rate` | 0.194444 |  |
| execution/p0.0 | `B2/p0.0/wilson_lo` | 0.17194 |  |
| execution/p0.0 | `B2/p0.0/wilson_hi` | 0.219115 |  |
| execution/p0.0 | `B2/p0.0/collision_rate` | 0.527778 |  |
| execution/p0.0 | `B2/p0.0/makespan_mean_s` | 12.4 |  |
| execution/p0.0 | `B2/p0.0/makespan_p90_s` | 12.4 |  |
| execution/p0.05 | `B2/p0.05/n` | 1080 |  |
| execution/p0.05 | `B2/p0.05/successes` | 204 |  |
| execution/p0.05 | `B2/p0.05/safe_completion_rate` | 0.188889 |  |
| execution/p0.05 | `B2/p0.05/wilson_lo` | 0.166662 |  |
| execution/p0.05 | `B2/p0.05/wilson_hi` | 0.213321 |  |
| execution/p0.05 | `B2/p0.05/collision_rate` | 0.513889 |  |
| execution/p0.05 | `B2/p0.05/makespan_mean_s` | 13.5796 |  |
| execution/p0.05 | `B2/p0.05/makespan_p90_s` | 14.1889 |  |
| execution/p0.1 | `B2/p0.1/n` | 1080 |  |
| execution/p0.1 | `B2/p0.1/successes` | 210 |  |
| execution/p0.1 | `B2/p0.1/safe_completion_rate` | 0.194444 |  |
| execution/p0.1 | `B2/p0.1/wilson_lo` | 0.17194 |  |
| execution/p0.1 | `B2/p0.1/wilson_hi` | 0.219115 |  |
| execution/p0.1 | `B2/p0.1/collision_rate` | 0.512963 |  |
| execution/p0.1 | `B2/p0.1/makespan_mean_s` | 14.5274 |  |
| execution/p0.1 | `B2/p0.1/makespan_p90_s` | 15.3478 |  |
| execution/p0.2 | `B2/p0.2/n` | 1080 |  |
| execution/p0.2 | `B2/p0.2/successes` | 207 |  |
| execution/p0.2 | `B2/p0.2/safe_completion_rate` | 0.191667 |  |
| execution/p0.2 | `B2/p0.2/wilson_lo` | 0.1693 |  |
| execution/p0.2 | `B2/p0.2/wilson_hi` | 0.216219 |  |
| execution/p0.2 | `B2/p0.2/collision_rate` | 0.525 |  |
| execution/p0.2 | `B2/p0.2/makespan_mean_s` | 16.757 |  |
| execution/p0.2 | `B2/p0.2/makespan_p90_s` | 17.9944 |  |
| execution/p0.0 | `B3/p0.0/n` | 1080 |  |
| execution/p0.0 | `B3/p0.0/successes` | 300 |  |
| execution/p0.0 | `B3/p0.0/safe_completion_rate` | 0.277778 |  |
| execution/p0.0 | `B3/p0.0/wilson_lo` | 0.251888 |  |
| execution/p0.0 | `B3/p0.0/wilson_hi` | 0.305243 |  |
| execution/p0.0 | `B3/p0.0/collision_rate` | 0.25 |  |
| execution/p0.0 | `B3/p0.0/makespan_mean_s` | 14.1444 |  |
| execution/p0.0 | `B3/p0.0/makespan_p90_s` | 14.1444 |  |
| execution/p0.05 | `B3/p0.05/n` | 1080 |  |
| execution/p0.05 | `B3/p0.05/successes` | 282 |  |
| execution/p0.05 | `B3/p0.05/safe_completion_rate` | 0.261111 |  |
| execution/p0.05 | `B3/p0.05/wilson_lo` | 0.235794 |  |
| execution/p0.05 | `B3/p0.05/wilson_hi` | 0.288122 |  |
| execution/p0.05 | `B3/p0.05/collision_rate` | 0.281481 |  |
| execution/p0.05 | `B3/p0.05/makespan_mean_s` | 14.8707 |  |
| execution/p0.05 | `B3/p0.05/makespan_p90_s` | 15.5067 |  |
| execution/p0.1 | `B3/p0.1/n` | 1080 |  |
| execution/p0.1 | `B3/p0.1/successes` | 277 |  |
| execution/p0.1 | `B3/p0.1/safe_completion_rate` | 0.256481 |  |
| execution/p0.1 | `B3/p0.1/wilson_lo` | 0.231332 |  |
| execution/p0.1 | `B3/p0.1/wilson_hi` | 0.283357 |  |
| execution/p0.1 | `B3/p0.1/collision_rate` | 0.302778 |  |
| execution/p0.1 | `B3/p0.1/makespan_mean_s` | 15.5841 |  |
| execution/p0.1 | `B3/p0.1/makespan_p90_s` | 16.3833 |  |
| execution/p0.2 | `B3/p0.2/n` | 1080 |  |
| execution/p0.2 | `B3/p0.2/successes` | 256 |  |
| execution/p0.2 | `B3/p0.2/safe_completion_rate` | 0.237037 |  |
| execution/p0.2 | `B3/p0.2/wilson_lo` | 0.212634 |  |
| execution/p0.2 | `B3/p0.2/wilson_hi` | 0.263304 |  |
| execution/p0.2 | `B3/p0.2/collision_rate` | 0.383333 |  |
| execution/p0.2 | `B3/p0.2/makespan_mean_s` | 17.3615 |  |
| execution/p0.2 | `B3/p0.2/makespan_p90_s` | 18.5967 |  |
| execution/p0.0 | `A1/p0.0/n` | 1080 |  |
| execution/p0.0 | `A1/p0.0/successes` | 210 |  |
| execution/p0.0 | `A1/p0.0/safe_completion_rate` | 0.194444 |  |
| execution/p0.0 | `A1/p0.0/wilson_lo` | 0.17194 |  |
| execution/p0.0 | `A1/p0.0/wilson_hi` | 0.219115 |  |
| execution/p0.0 | `A1/p0.0/collision_rate` | 0.527778 |  |
| execution/p0.0 | `A1/p0.0/makespan_mean_s` | 12.4 |  |
| execution/p0.0 | `A1/p0.0/makespan_p90_s` | 12.4 |  |
| execution/p0.05 | `A1/p0.05/n` | 1080 |  |
| execution/p0.05 | `A1/p0.05/successes` | 204 |  |
| execution/p0.05 | `A1/p0.05/safe_completion_rate` | 0.188889 |  |
| execution/p0.05 | `A1/p0.05/wilson_lo` | 0.166662 |  |
| execution/p0.05 | `A1/p0.05/wilson_hi` | 0.213321 |  |
| execution/p0.05 | `A1/p0.05/collision_rate` | 0.525926 |  |
| execution/p0.05 | `A1/p0.05/makespan_mean_s` | 13.5796 |  |
| execution/p0.05 | `A1/p0.05/makespan_p90_s` | 14.1889 |  |
| execution/p0.1 | `A1/p0.1/n` | 1080 |  |
| execution/p0.1 | `A1/p0.1/successes` | 210 |  |
| execution/p0.1 | `A1/p0.1/safe_completion_rate` | 0.194444 |  |
| execution/p0.1 | `A1/p0.1/wilson_lo` | 0.17194 |  |
| execution/p0.1 | `A1/p0.1/wilson_hi` | 0.219115 |  |
| execution/p0.1 | `A1/p0.1/collision_rate` | 0.522222 |  |
| execution/p0.1 | `A1/p0.1/makespan_mean_s` | 14.5274 |  |
| execution/p0.1 | `A1/p0.1/makespan_p90_s` | 15.3478 |  |
| execution/p0.2 | `A1/p0.2/n` | 1080 |  |
| execution/p0.2 | `A1/p0.2/successes` | 207 |  |
| execution/p0.2 | `A1/p0.2/safe_completion_rate` | 0.191667 |  |
| execution/p0.2 | `A1/p0.2/wilson_lo` | 0.1693 |  |
| execution/p0.2 | `A1/p0.2/wilson_hi` | 0.216219 |  |
| execution/p0.2 | `A1/p0.2/collision_rate` | 0.537963 |  |
| execution/p0.2 | `A1/p0.2/makespan_mean_s` | 16.757 |  |
| execution/p0.2 | `A1/p0.2/makespan_p90_s` | 17.9944 |  |
| execution/p0.0 | `A2/p0.0/n` | 1080 |  |
| execution/p0.0 | `A2/p0.0/successes` | 240 |  |
| execution/p0.0 | `A2/p0.0/safe_completion_rate` | 0.222222 |  |
| execution/p0.0 | `A2/p0.0/wilson_lo` | 0.198436 |  |
| execution/p0.0 | `A2/p0.0/wilson_hi` | 0.247977 |  |
| execution/p0.0 | `A2/p0.0/collision_rate` | 0.444444 |  |
| execution/p0.0 | `A2/p0.0/makespan_mean_s` | 13.7 |  |
| execution/p0.0 | `A2/p0.0/makespan_p90_s` | 13.7 |  |
| execution/p0.05 | `A2/p0.05/n` | 1080 |  |
| execution/p0.05 | `A2/p0.05/successes` | 240 |  |
| execution/p0.05 | `A2/p0.05/safe_completion_rate` | 0.222222 |  |
| execution/p0.05 | `A2/p0.05/wilson_lo` | 0.198436 |  |
| execution/p0.05 | `A2/p0.05/wilson_hi` | 0.247977 |  |
| execution/p0.05 | `A2/p0.05/collision_rate` | 0.439815 |  |
| execution/p0.05 | `A2/p0.05/makespan_mean_s` | 14.4711 |  |
| execution/p0.05 | `A2/p0.05/makespan_p90_s` | 15.0744 |  |
| execution/p0.1 | `A2/p0.1/n` | 1080 |  |
| execution/p0.1 | `A2/p0.1/successes` | 241 |  |
| execution/p0.1 | `A2/p0.1/safe_completion_rate` | 0.223148 |  |
| execution/p0.1 | `A2/p0.1/wilson_lo` | 0.199322 |  |
| execution/p0.1 | `A2/p0.1/wilson_hi` | 0.248937 |  |
| execution/p0.1 | `A2/p0.1/collision_rate` | 0.436111 |  |
| execution/p0.1 | `A2/p0.1/makespan_mean_s` | 15.2615 |  |
| execution/p0.1 | `A2/p0.1/makespan_p90_s` | 16.0822 |  |
| execution/p0.2 | `A2/p0.2/n` | 1080 |  |
| execution/p0.2 | `A2/p0.2/successes` | 228 |  |
| execution/p0.2 | `A2/p0.2/safe_completion_rate` | 0.211111 |  |
| execution/p0.2 | `A2/p0.2/wilson_lo` | 0.187817 |  |
| execution/p0.2 | `A2/p0.2/wilson_hi` | 0.236453 |  |
| execution/p0.2 | `A2/p0.2/collision_rate` | 0.469444 |  |
| execution/p0.2 | `A2/p0.2/makespan_mean_s` | 17.1341 |  |
| execution/p0.2 | `A2/p0.2/makespan_p90_s` | 18.3567 |  |
| execution/p0.0 | `A3/p0.0/n` | 1080 |  |
| execution/p0.0 | `A3/p0.0/successes` | 240 |  |
| execution/p0.0 | `A3/p0.0/safe_completion_rate` | 0.222222 |  |
| execution/p0.0 | `A3/p0.0/wilson_lo` | 0.198436 |  |
| execution/p0.0 | `A3/p0.0/wilson_hi` | 0.247977 |  |
| execution/p0.0 | `A3/p0.0/collision_rate` | 0.361111 |  |
| execution/p0.0 | `A3/p0.0/makespan_mean_s` | 13.6667 |  |
| execution/p0.0 | `A3/p0.0/makespan_p90_s` | 13.6667 |  |
| execution/p0.05 | `A3/p0.05/n` | 1080 |  |
| execution/p0.05 | `A3/p0.05/successes` | 240 |  |
| execution/p0.05 | `A3/p0.05/safe_completion_rate` | 0.222222 |  |
| execution/p0.05 | `A3/p0.05/wilson_lo` | 0.198436 |  |
| execution/p0.05 | `A3/p0.05/wilson_hi` | 0.247977 |  |
| execution/p0.05 | `A3/p0.05/collision_rate` | 0.376852 |  |
| execution/p0.05 | `A3/p0.05/makespan_mean_s` | 14.4478 |  |
| execution/p0.05 | `A3/p0.05/makespan_p90_s` | 15.0511 |  |
| execution/p0.1 | `A3/p0.1/n` | 1080 |  |
| execution/p0.1 | `A3/p0.1/successes` | 244 |  |
| execution/p0.1 | `A3/p0.1/safe_completion_rate` | 0.225926 |  |
| execution/p0.1 | `A3/p0.1/wilson_lo` | 0.201981 |  |
| execution/p0.1 | `A3/p0.1/wilson_hi` | 0.251813 |  |
| execution/p0.1 | `A3/p0.1/collision_rate` | 0.384259 |  |
| execution/p0.1 | `A3/p0.1/makespan_mean_s` | 15.2404 |  |
| execution/p0.1 | `A3/p0.1/makespan_p90_s` | 16.0622 |  |
| execution/p0.2 | `A3/p0.2/n` | 1080 |  |
| execution/p0.2 | `A3/p0.2/successes` | 233 |  |
| execution/p0.2 | `A3/p0.2/safe_completion_rate` | 0.215741 |  |
| execution/p0.2 | `A3/p0.2/wilson_lo` | 0.192239 |  |
| execution/p0.2 | `A3/p0.2/wilson_hi` | 0.241258 |  |
| execution/p0.2 | `A3/p0.2/collision_rate` | 0.432407 |  |
| execution/p0.2 | `A3/p0.2/makespan_mean_s` | 17.1285 |  |
| execution/p0.2 | `A3/p0.2/makespan_p90_s` | 18.3356 |  |

## 产物

- `tables\plans.csv`
- `tables\executions.csv`
- `plots\safe_completion_vs_pause.png`
- `plots\plan_time_and_makespan.png`
- `plots\safety_efficiency_scatter.png`
- `plots\failure_attribution.png`
