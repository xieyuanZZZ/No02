# S5 完整对照实验报告

- 阶段：`S5_evaluation`
- 运行目录：`D:\paper2\results\S5_evaluation\20261103_032138_s5_benchmark_984312`
- 生成时间：2026-11-03T03:32:03+08:00
- 耗时：0.8 s

## 实验设计

- 任务：36 个（4 类地图 × [2, 4, 6] 台），每格 3 个；
- 方法：B0, B1, B2, B3, A1, A2, A3（共享同一 Diffusion 权重、同一噪声、20 步 DDIM、相同引导预算）；
- 执行：每份规划在 p∈[0.0, 0.05, 0.1, 0.2] 下各做 30 次独立随机执行；
- 成功口径：规划有效 + 执行中全部到达 + 执行全程无碰撞（失败规划计 0）；
- 置信区间：以任务为簇的 bootstrap（2000 次重采样）。

## 结果

| 方法 | 规划成功率 | 平均耗时 [s] | 平均等待 | 平均硬风险 |
|---|---|---|---|---|
| B0 独立 Diffusion | 0.306 | 0.84 | 0.00 | 1.1070 |
| B1 同步距离引导 | 0.306 | 0.84 | 0.00 | 0.9902 |
| B2 固定时间窗引导 | 0.306 | 1.07 | 0.00 | 0.8805 |
| B3 逐时刻期望碰撞风险 | 0.306 | 5.06 | 5.16 | 0.4691 |
| A1 首次碰撞 + 只改路径 | 0.306 | 2.54 | 0.00 | 0.9426 |
| A2 首次碰撞 + 只改等待 | 0.306 | 2.50 | 2.52 | 0.7459 |
| A3 首次碰撞 + 联合 | 0.306 | 2.61 | 2.50 | 0.6198 |

## 执行结果（p=0.2）

| 方法 | 安全完成率 | 95% 区间 | 碰撞率 | makespan [s] |
|---|---|---|---|---|
| B0 | 0.186 | [0.164, 0.210] | 0.568 | 16.76 |
| B1 | 0.192 | [0.169, 0.216] | 0.556 | 16.76 |
| B2 | 0.192 | [0.169, 0.216] | 0.525 | 16.76 |
| B3 | 0.233 | [0.209, 0.259] | 0.410 | 17.25 |
| A1 | 0.192 | [0.169, 0.216] | 0.553 | 16.76 |
| A2 | 0.195 | [0.173, 0.220] | 0.509 | 16.98 |
| A3 | 0.205 | [0.182, 0.230] | 0.480 | 16.98 |

## 关键检验（G2：首次碰撞吸收是否不可替代）

- A3 vs B3（逐时刻期望风险）：安全完成率差 -0.0299，95% 区间 [-0.0820, -0.0002]，显著；
- A3 安全完成率 0.205，B3 0.233（p=0.2）。

## 结论与局限

- 本报告为**预实验规模**（配置 `benchmark.scale`），不可当作完整规模的最终结论；
- 若 A3 与 B3 差异不显著，则“首次碰撞吸收”的独立贡献尚未被证明（G2 未达成）；
- 所有失败均按 §10.2 口径计入分母，未做后验剔除。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T03:32:02+08:00 |  |
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
| plan | `B0/plan_time_mean_s` | 0.836673 |  |
| plan | `B0/wait_total_mean` | 0 |  |
| plan | `B0/hard_risk_mean` | 1.10697 |  |
| plan | `B1/n_plans` | 36 |  |
| plan | `B1/plan_valid_rate` | 0.305556 |  |
| plan | `B1/plan_time_mean_s` | 0.839805 |  |
| plan | `B1/wait_total_mean` | 0 |  |
| plan | `B1/hard_risk_mean` | 0.990162 |  |
| plan | `B2/n_plans` | 36 |  |
| plan | `B2/plan_valid_rate` | 0.305556 |  |
| plan | `B2/plan_time_mean_s` | 1.07089 |  |
| plan | `B2/wait_total_mean` | 0 |  |
| plan | `B2/hard_risk_mean` | 0.880481 |  |
| plan | `B3/n_plans` | 36 |  |
| plan | `B3/plan_valid_rate` | 0.305556 |  |
| plan | `B3/plan_time_mean_s` | 5.06029 |  |
| plan | `B3/wait_total_mean` | 5.15728 |  |
| plan | `B3/hard_risk_mean` | 0.469109 |  |
| plan | `A1/n_plans` | 36 |  |
| plan | `A1/plan_valid_rate` | 0.305556 |  |
| plan | `A1/plan_time_mean_s` | 2.54263 |  |
| plan | `A1/wait_total_mean` | 0 |  |
| plan | `A1/hard_risk_mean` | 0.942618 |  |
| plan | `A2/n_plans` | 36 |  |
| plan | `A2/plan_valid_rate` | 0.305556 |  |
| plan | `A2/plan_time_mean_s` | 2.49519 |  |
| plan | `A2/wait_total_mean` | 2.51906 |  |
| plan | `A2/hard_risk_mean` | 0.745878 |  |
| plan | `A3/n_plans` | 36 |  |
| plan | `A3/plan_valid_rate` | 0.305556 |  |
| plan | `A3/plan_time_mean_s` | 2.60778 |  |
| plan | `A3/wait_total_mean` | 2.50147 |  |
| plan | `A3/hard_risk_mean` | 0.619847 |  |
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
| execution/p0.2 | `B1/p0.2/collision_rate` | 0.556481 |  |
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
| execution/p0.0 | `B3/p0.0/successes` | 270 |  |
| execution/p0.0 | `B3/p0.0/safe_completion_rate` | 0.25 |  |
| execution/p0.0 | `B3/p0.0/wilson_lo` | 0.225091 |  |
| execution/p0.0 | `B3/p0.0/wilson_hi` | 0.276681 |  |
| execution/p0.0 | `B3/p0.0/collision_rate` | 0.305556 |  |
| execution/p0.0 | `B3/p0.0/makespan_mean_s` | 13.9222 |  |
| execution/p0.0 | `B3/p0.0/makespan_p90_s` | 13.9222 |  |
| execution/p0.05 | `B3/p0.05/n` | 1080 |  |
| execution/p0.05 | `B3/p0.05/successes` | 270 |  |
| execution/p0.05 | `B3/p0.05/safe_completion_rate` | 0.25 |  |
| execution/p0.05 | `B3/p0.05/wilson_lo` | 0.225091 |  |
| execution/p0.05 | `B3/p0.05/wilson_hi` | 0.276681 |  |
| execution/p0.05 | `B3/p0.05/collision_rate` | 0.322222 |  |
| execution/p0.05 | `B3/p0.05/makespan_mean_s` | 14.6681 |  |
| execution/p0.05 | `B3/p0.05/makespan_p90_s` | 15.3178 |  |
| execution/p0.1 | `B3/p0.1/n` | 1080 |  |
| execution/p0.1 | `B3/p0.1/successes` | 271 |  |
| execution/p0.1 | `B3/p0.1/safe_completion_rate` | 0.250926 |  |
| execution/p0.1 | `B3/p0.1/wilson_lo` | 0.225982 |  |
| execution/p0.1 | `B3/p0.1/wilson_hi` | 0.277635 |  |
| execution/p0.1 | `B3/p0.1/collision_rate` | 0.333333 |  |
| execution/p0.1 | `B3/p0.1/makespan_mean_s` | 15.4074 |  |
| execution/p0.1 | `B3/p0.1/makespan_p90_s` | 16.2256 |  |
| execution/p0.2 | `B3/p0.2/n` | 1080 |  |
| execution/p0.2 | `B3/p0.2/successes` | 252 |  |
| execution/p0.2 | `B3/p0.2/safe_completion_rate` | 0.233333 |  |
| execution/p0.2 | `B3/p0.2/wilson_lo` | 0.20908 |  |
| execution/p0.2 | `B3/p0.2/wilson_hi` | 0.259477 |  |
| execution/p0.2 | `B3/p0.2/collision_rate` | 0.410185 |  |
| execution/p0.2 | `B3/p0.2/makespan_mean_s` | 17.2493 |  |
| execution/p0.2 | `B3/p0.2/makespan_p90_s` | 18.4544 |  |
| execution/p0.0 | `A1/p0.0/n` | 1080 |  |
| execution/p0.0 | `A1/p0.0/successes` | 210 |  |
| execution/p0.0 | `A1/p0.0/safe_completion_rate` | 0.194444 |  |
| execution/p0.0 | `A1/p0.0/wilson_lo` | 0.17194 |  |
| execution/p0.0 | `A1/p0.0/wilson_hi` | 0.219115 |  |
| execution/p0.0 | `A1/p0.0/collision_rate` | 0.555556 |  |
| execution/p0.0 | `A1/p0.0/makespan_mean_s` | 12.4 |  |
| execution/p0.0 | `A1/p0.0/makespan_p90_s` | 12.4 |  |
| execution/p0.05 | `A1/p0.05/n` | 1080 |  |
| execution/p0.05 | `A1/p0.05/successes` | 204 |  |
| execution/p0.05 | `A1/p0.05/safe_completion_rate` | 0.188889 |  |
| execution/p0.05 | `A1/p0.05/wilson_lo` | 0.166662 |  |
| execution/p0.05 | `A1/p0.05/wilson_hi` | 0.213321 |  |
| execution/p0.05 | `A1/p0.05/collision_rate` | 0.547222 |  |
| execution/p0.05 | `A1/p0.05/makespan_mean_s` | 13.5796 |  |
| execution/p0.05 | `A1/p0.05/makespan_p90_s` | 14.1889 |  |
| execution/p0.1 | `A1/p0.1/n` | 1080 |  |
| execution/p0.1 | `A1/p0.1/successes` | 210 |  |
| execution/p0.1 | `A1/p0.1/safe_completion_rate` | 0.194444 |  |
| execution/p0.1 | `A1/p0.1/wilson_lo` | 0.17194 |  |
| execution/p0.1 | `A1/p0.1/wilson_hi` | 0.219115 |  |
| execution/p0.1 | `A1/p0.1/collision_rate` | 0.537963 |  |
| execution/p0.1 | `A1/p0.1/makespan_mean_s` | 14.5274 |  |
| execution/p0.1 | `A1/p0.1/makespan_p90_s` | 15.3478 |  |
| execution/p0.2 | `A1/p0.2/n` | 1080 |  |
| execution/p0.2 | `A1/p0.2/successes` | 207 |  |
| execution/p0.2 | `A1/p0.2/safe_completion_rate` | 0.191667 |  |
| execution/p0.2 | `A1/p0.2/wilson_lo` | 0.1693 |  |
| execution/p0.2 | `A1/p0.2/wilson_hi` | 0.216219 |  |
| execution/p0.2 | `A1/p0.2/collision_rate` | 0.552778 |  |
| execution/p0.2 | `A1/p0.2/makespan_mean_s` | 16.757 |  |
| execution/p0.2 | `A1/p0.2/makespan_p90_s` | 17.9944 |  |
| execution/p0.0 | `A2/p0.0/n` | 1080 |  |
| execution/p0.0 | `A2/p0.0/successes` | 240 |  |
| execution/p0.0 | `A2/p0.0/safe_completion_rate` | 0.222222 |  |
| execution/p0.0 | `A2/p0.0/wilson_lo` | 0.198436 |  |
| execution/p0.0 | `A2/p0.0/wilson_hi` | 0.247977 |  |
| execution/p0.0 | `A2/p0.0/collision_rate` | 0.444444 |  |
| execution/p0.0 | `A2/p0.0/makespan_mean_s` | 13.3333 |  |
| execution/p0.0 | `A2/p0.0/makespan_p90_s` | 13.3333 |  |
| execution/p0.05 | `A2/p0.05/n` | 1080 |  |
| execution/p0.05 | `A2/p0.05/successes` | 235 |  |
| execution/p0.05 | `A2/p0.05/safe_completion_rate` | 0.217593 |  |
| execution/p0.05 | `A2/p0.05/wilson_lo` | 0.194008 |  |
| execution/p0.05 | `A2/p0.05/wilson_hi` | 0.243179 |  |
| execution/p0.05 | `A2/p0.05/collision_rate` | 0.459259 |  |
| execution/p0.05 | `A2/p0.05/makespan_mean_s` | 14.1493 |  |
| execution/p0.05 | `A2/p0.05/makespan_p90_s` | 14.7722 |  |
| execution/p0.1 | `A2/p0.1/n` | 1080 |  |
| execution/p0.1 | `A2/p0.1/successes` | 225 |  |
| execution/p0.1 | `A2/p0.1/safe_completion_rate` | 0.208333 |  |
| execution/p0.1 | `A2/p0.1/wilson_lo` | 0.185167 |  |
| execution/p0.1 | `A2/p0.1/wilson_hi` | 0.233567 |  |
| execution/p0.1 | `A2/p0.1/collision_rate` | 0.472222 |  |
| execution/p0.1 | `A2/p0.1/makespan_mean_s` | 14.9781 |  |
| execution/p0.1 | `A2/p0.1/makespan_p90_s` | 15.7844 |  |
| execution/p0.2 | `A2/p0.2/n` | 1080 |  |
| execution/p0.2 | `A2/p0.2/successes` | 211 |  |
| execution/p0.2 | `A2/p0.2/safe_completion_rate` | 0.19537 |  |
| execution/p0.2 | `A2/p0.2/wilson_lo` | 0.172821 |  |
| execution/p0.2 | `A2/p0.2/wilson_hi` | 0.22008 |  |
| execution/p0.2 | `A2/p0.2/collision_rate` | 0.509259 |  |
| execution/p0.2 | `A2/p0.2/makespan_mean_s` | 16.9815 |  |
| execution/p0.2 | `A2/p0.2/makespan_p90_s` | 18.2267 |  |
| execution/p0.0 | `A3/p0.0/n` | 1080 |  |
| execution/p0.0 | `A3/p0.0/successes` | 240 |  |
| execution/p0.0 | `A3/p0.0/safe_completion_rate` | 0.222222 |  |
| execution/p0.0 | `A3/p0.0/wilson_lo` | 0.198436 |  |
| execution/p0.0 | `A3/p0.0/wilson_hi` | 0.247977 |  |
| execution/p0.0 | `A3/p0.0/collision_rate` | 0.388889 |  |
| execution/p0.0 | `A3/p0.0/makespan_mean_s` | 13.3111 |  |
| execution/p0.0 | `A3/p0.0/makespan_p90_s` | 13.3111 |  |
| execution/p0.05 | `A3/p0.05/n` | 1080 |  |
| execution/p0.05 | `A3/p0.05/successes` | 237 |  |
| execution/p0.05 | `A3/p0.05/safe_completion_rate` | 0.219444 |  |
| execution/p0.05 | `A3/p0.05/wilson_lo` | 0.195779 |  |
| execution/p0.05 | `A3/p0.05/wilson_hi` | 0.245099 |  |
| execution/p0.05 | `A3/p0.05/collision_rate` | 0.402778 |  |
| execution/p0.05 | `A3/p0.05/makespan_mean_s` | 14.1304 |  |
| execution/p0.05 | `A3/p0.05/makespan_p90_s` | 14.7511 |  |
| execution/p0.1 | `A3/p0.1/n` | 1080 |  |
| execution/p0.1 | `A3/p0.1/successes` | 236 |  |
| execution/p0.1 | `A3/p0.1/safe_completion_rate` | 0.218519 |  |
| execution/p0.1 | `A3/p0.1/wilson_lo` | 0.194894 |  |
| execution/p0.1 | `A3/p0.1/wilson_hi` | 0.244139 |  |
| execution/p0.1 | `A3/p0.1/collision_rate` | 0.432407 |  |
| execution/p0.1 | `A3/p0.1/makespan_mean_s` | 14.9689 |  |
| execution/p0.1 | `A3/p0.1/makespan_p90_s` | 15.7844 |  |
| execution/p0.2 | `A3/p0.2/n` | 1080 |  |
| execution/p0.2 | `A3/p0.2/successes` | 221 |  |
| execution/p0.2 | `A3/p0.2/safe_completion_rate` | 0.20463 |  |
| execution/p0.2 | `A3/p0.2/wilson_lo` | 0.181635 |  |
| execution/p0.2 | `A3/p0.2/wilson_hi` | 0.229718 |  |
| execution/p0.2 | `A3/p0.2/collision_rate` | 0.47963 |  |
| execution/p0.2 | `A3/p0.2/makespan_mean_s` | 16.9778 |  |
| execution/p0.2 | `A3/p0.2/makespan_p90_s` | 18.2267 |  |

## 产物

- `tables\plans.csv`
- `tables\executions.csv`
- `plots\safe_completion_vs_pause.png`
- `plots\plan_time_and_makespan.png`
- `plots\safety_efficiency_scatter.png`
- `plots\failure_attribution.png`
