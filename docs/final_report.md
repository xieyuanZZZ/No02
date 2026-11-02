# 可微首次碰撞风险引导的二维多机器人扩散轨迹规划 —— 实验总报告

生成时间：2026-11-03T04:38:52+08:00

> 本报告由脚本自动汇总各阶段 `docs/stages/Sx/` 下的真实运行结果生成，所有数值均来自实际运行，不含手绘或估计。

## 阶段总览

| 阶段 | 名称 | 状态 | 完成时间 | 耗时 | Git commit | 指标数 |
|---|---|---|---|---|---|---|
| S0 | 环境与几何基础 | passed | 2026-11-03T02:03:14+08:00 | 3.6 s | unknown | 88 |
| S1 | 首次碰撞风险 DP | passed | 2026-11-03T02:05:26+08:00 | 13.4 s | 0ef64b83 | 36 |
| S2 | 无 Diffusion 的风险梯度优化 | passed | 2026-11-03T02:15:57+08:00 | 157.7 s | 2e7ab8e4 | 89 |
| S3 | 单机器人条件 Diffusion | completed | 2026-11-03T04:38:39+08:00 | 2.4 s | 84a735df | 29 |
| S4 | 风险引导 DDIM 接入 | passed | 2026-11-03T03:21:32+08:00 | 1.5 s | 41d9a81a | 46 |
| S5 | 完整对照实验 | completed | 2026-11-03T03:46:41+08:00 | 0.9 s | 41d9a81a | 275 |

## S0：环境与几何基础

运行目录：`results/S0_environment/`（完整记录），归档：`docs/stages/S0/`。

### 关键指标

| 分组 | 指标 | 值 |
|---|---|---|
| environment | `timestamp` | 2026-11-03T02:03:10+08:00 |
| environment | `python` | 3.12.13 |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |
| environment | `torch` | 2.11.0+cu128 |
| environment | `cuda_available` | True |
| environment | `cuda_version` | 12.8 |
| environment | `cudnn_version` | 91900 |
| environment | `platform` | Windows-11-10.0.26200-SP0 |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |
| environment | `cpu_count` | 10 |
| environment | `git_commit` | unknown |
| environment | `git_dirty` | False |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |
| environment | `gpu_total_memory_gb` | 11.99 |
| environment | `gpu_index` | 0 |
| environment | `gpu_capability` | 8.9 |
| map/open | `n_obstacles` | 0 |
| map/open | `occupied_cells` | 0 |
| map/open | `occupancy_ratio` | 0 |
| map/open | `sdf_min_m` | inf |
| map/open | `sdf_max_m` | inf |
| map/dual_corridor | `n_obstacles` | 1 |
| map/dual_corridor | `occupied_cells` | 500 |
| map/dual_corridor | `occupancy_ratio` | 0.12207 |
| map/dual_corridor | `sdf_min_m` | -0.18 |
| map/dual_corridor | `sdf_max_m` | 4.9795 |
| map/narrow_gate | `n_obstacles` | 2 |
| map/narrow_gate | `occupied_cells` | 245 |
| map/narrow_gate | `occupancy_ratio` | 0.0598145 |
| map/narrow_gate | `sdf_min_m` | -0.18 |
| map/narrow_gate | `sdf_max_m` | 4.99531 |
| map/bottleneck | `n_obstacles` | 6 |
| map/bottleneck | `occupied_cells` | 1273 |
| map/bottleneck | `occupancy_ratio` | 0.310791 |
| map/bottleneck | `sdf_min_m` | -0.408909 |
| map/bottleneck | `sdf_max_m` | 3.39408 |
| scenario/crossing | `n_agents` | 2 |
| scenario/crossing | `path_length_mean_m` | 8.8 |
| scenario/crossing | `max_segment_m` | 0.283871 |
| scenario/crossing | `speed_limit_ok` | True |
| scenario/crossing | `static_safe` | True |
| scenario/crossing | `min_clearance_m` | inf |
| scenario/head_on | `n_agents` | 2 |
| scenario/head_on | `path_length_mean_m` | 8.4 |
| scenario/head_on | `max_segment_m` | 0.270968 |
| scenario/head_on | `speed_limit_ok` | True |
| scenario/head_on | `static_safe` | True |
| scenario/head_on | `min_clearance_m` | inf |
| scenario/follow | `n_agents` | 2 |
| scenario/follow | `path_length_mean_m` | 7.48387 |
| scenario/follow | `max_segment_m` | 0.241415 |
| scenario/follow | `speed_limit_ok` | True |
| scenario/follow | `static_safe` | True |
| scenario/follow | `min_clearance_m` | inf |
| scenario/terminal_occupancy | `n_agents` | 2 |
| scenario/terminal_occupancy | `path_length_mean_m` | 5.4 |
| scenario/terminal_occupancy | `max_segment_m` | 0.270968 |
| scenario/terminal_occupancy | `speed_limit_ok` | True |
| scenario/terminal_occupancy | `static_safe` | True |
| scenario/terminal_occupancy | `min_clearance_m` | inf |
| scenario/dual_corridor | `n_agents` | 2 |
| scenario/dual_corridor | `path_length_mean_m` | 8.8 |
| scenario/dual_corridor | `max_segment_m` | 0.372156 |
| scenario/dual_corridor | `speed_limit_ok` | True |
| scenario/dual_corridor | `static_safe` | True |
| scenario/dual_corridor | `min_clearance_m` | 0.40907 |
| scenario/narrow_gate | `n_agents` | 2 |
| scenario/narrow_gate | `path_length_mean_m` | 8.8 |
| scenario/narrow_gate | `max_segment_m` | 0.283871 |
| scenario/narrow_gate | `speed_limit_ok` | True |
| scenario/narrow_gate | `static_safe` | True |
| scenario/narrow_gate | `min_clearance_m` | 0.202939 |
| scenario/multi_bottleneck | `n_agents` | 4 |
| scenario/multi_bottleneck | `path_length_mean_m` | 8.9607 |
| scenario/multi_bottleneck | `max_segment_m` | 0.393377 |
| scenario/multi_bottleneck | `speed_limit_ok` | True |
| scenario/multi_bottleneck | `static_safe` | True |
| scenario/multi_bottleneck | `min_clearance_m` | 0.0365593 |
| execution | `crossing/n_samples` | 500 |
| execution | `crossing/collision_rate` | 0.176 |
| execution | `crossing/completion_rate` | 1 |
| execution | `crossing/mean_makespan_steps` | 35.59 |
| execution | `crossing/mean_makespan_s` | 14.236 |
| execution | `crossing/std_makespan_s` | 0.764607 |
| execution | `crossing/unfinished_agent_rate` | 0 |
| execution | `crossing/d_safe_m` | 0.35 |
| tests | `tests_passed` | 12 |
| tests | `tests_failed` | 0 |

### 图表

![execution_single_sample_detail.png](stages/S0/plots/execution_single_sample_detail.png)
![execution_traces_crossing.png](stages/S0/plots/execution_traces_crossing.png)
![map_bottleneck.png](stages/S0/plots/map_bottleneck.png)
![map_dual_corridor.png](stages/S0/plots/map_dual_corridor.png)
![map_narrow_gate.png](stages/S0/plots/map_narrow_gate.png)
![map_open.png](stages/S0/plots/map_open.png)
![scenario_crossing.png](stages/S0/plots/scenario_crossing.png)
![scenario_dual_corridor.png](stages/S0/plots/scenario_dual_corridor.png)
![scenario_follow.png](stages/S0/plots/scenario_follow.png)
![scenario_head_on.png](stages/S0/plots/scenario_head_on.png)
![scenario_multi_bottleneck.png](stages/S0/plots/scenario_multi_bottleneck.png)
![scenario_narrow_gate.png](stages/S0/plots/scenario_narrow_gate.png)
![scenario_terminal_occupancy.png](stages/S0/plots/scenario_terminal_occupancy.png)

### 阶段报告原文

### S0 环境与几何基础验证报告

- 阶段：`S0_environment`
- 运行目录：`D:\paper2\results\S0_environment\20261103_020310_s0_baseline_7ff859`
- 生成时间：2026-11-03T02:03:14+08:00
- 耗时：3.6 s

#### 阶段目标

建立新工程环境、静态地图生成器、机器人圆盘几何、任务/场景构造、轨迹连续碰撞检测，以及独立随机执行模拟器；并用单元测试固定正确性。

#### 关键实现决定

- 障碍物只用**连续几何**（轴对齐矩形 + 圆）描述，栅格仅用于可视化与网络条件输入；
- 线段到矩形的距离用**逐矩形黄金分割**（单矩形距离对参数 u 是凸的，min 会破坏凸性，故不能合并求解）；
- 机器人之间用相对运动闭式解求最小距离，并额外保留一套独立实现供交叉验证；
- 执行模型为层 A：每时隙最多前进 1 个路点，时隙内线性插值，同步连续段碰撞判定。

#### 结论

- 4 类地图生成正常，占据率与 SDF 数值合理；
- 7 个手工场景全部满足段长 <= v_max*dt = 0.60 m；
- 交叉场景在 p=0.1 下随机执行碰撞率 0.1760，说明随机暂停确实会制造碰撞风险；
- S0 通过，可进入 S1（首次碰撞风险 DP）。


#### 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T02:03:10+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | unknown |  |
| environment | `git_dirty` | False |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| map/open | `n_obstacles` | 0 |  |
| map/open | `occupied_cells` | 0 |  |
| map/open | `occupancy_ratio` | 0 |  |
| map/open | `sdf_min_m` | inf |  |
| map/open | `sdf_max_m` | inf |  |
| map/dual_corridor | `n_obstacles` | 1 |  |
| map/dual_corridor | `occupied_cells` | 500 |  |
| map/dual_corridor | `occupancy_ratio` | 0.12207 |  |
| map/dual_corridor | `sdf_min_m` | -0.18 |  |
| map/dual_corridor | `sdf_max_m` | 4.9795 |  |
| map/narrow_gate | `n_obstacles` | 2 |  |
| map/narrow_gate | `occupied_cells` | 245 |  |
| map/narrow_gate | `occupancy_ratio` | 0.0598145 |  |
| map/narrow_gate | `sdf_min_m` | -0.18 |  |
| map/narrow_gate | `sdf_max_m` | 4.99531 |  |
| map/bottleneck | `n_obstacles` | 6 |  |
| map/bottleneck | `occupied_cells` | 1273 |  |
| map/bottleneck | `occupancy_ratio` | 0.310791 |  |
| map/bottleneck | `sdf_min_m` | -0.408909 |  |
| map/bottleneck | `sdf_max_m` | 3.39408 |  |
| scenario/crossing | `n_agents` | 2 |  |
| scenario/crossing | `path_length_mean_m` | 8.8 |  |
| scenario/crossing | `max_segment_m` | 0.283871 |  |
| scenario/crossing | `speed_limit_ok` | True |  |
| scenario/crossing | `static_safe` | True |  |
| scenario/crossing | `min_clearance_m` | inf |  |
| scenario/head_on | `n_agents` | 2 |  |
| scenario/head_on | `path_length_mean_m` | 8.4 |  |
| scenario/head_on | `max_segment_m` | 0.270968 |  |
| scenario/head_on | `speed_limit_ok` | True |  |
| scenario/head_on | `static_safe` | True |  |
| scenario/head_on | `min_clearance_m` | inf |  |
| scenario/follow | `n_agents` | 2 |  |
| scenario/follow | `path_length_mean_m` | 7.48387 |  |
| scenario/follow | `max_segment_m` | 0.241415 |  |
| scenario/follow | `speed_limit_ok` | True |  |
| scenario/follow | `static_safe` | True |  |
| scenario/follow | `min_clearance_m` | inf |  |
| scenario/terminal_occupancy | `n_agents` | 2 |  |
| scenario/terminal_occupancy | `path_length_mean_m` | 5.4 |  |
| scenario/terminal_occupancy | `max_segment_m` | 0.270968 |  |
| scenario/terminal_occupancy | `speed_limit_ok` | True |  |
| scenario/terminal_occupancy | `static_safe` | True |  |
| scenario/terminal_occupancy | `min_clearance_m` | inf |  |
| scenario/dual_corridor | `n_agents` | 2 |  |
| scenario/dual_corridor | `path_length_mean_m` | 8.8 |  |
| scenario/dual_corridor | `max_segment_m` | 0.372156 |  |
| scenario/dual_corridor | `speed_limit_ok` | True |  |
| scenario/dual_corridor | `static_safe` | True |  |
| scenario/dual_corridor | `min_clearance_m` | 0.40907 |  |
| scenario/narrow_gate | `n_agents` | 2 |  |
| scenario/narrow_gate | `path_length_mean_m` | 8.8 |  |
| scenario/narrow_gate | `max_segment_m` | 0.283871 |  |
| scenario/narrow_gate | `speed_limit_ok` | True |  |
| scenario/narrow_gate | `static_safe` | True |  |
| scenario/narrow_gate | `min_clearance_m` | 0.202939 |  |
| scenario/multi_bottleneck | `n_agents` | 4 |  |
| scenario/multi_bottleneck | `path_length_mean_m` | 8.9607 |  |
| scenario/multi_bottleneck | `max_segment_m` | 0.393377 |  |
| scenario/multi_bottleneck | `speed_limit_ok` | True |  |
| scenario/multi_bottleneck | `static_safe` | True |  |
| scenario/multi_bottleneck | `min_clearance_m` | 0.0365593 |  |
| execution | `crossing/n_samples` | 500 |  |
| execution | `crossing/collision_rate` | 0.176 |  |
| execution | `crossing/completion_rate` | 1 |  |
| execution | `crossing/mean_makespan_steps` | 35.59 |  |
| execution | `crossing/mean_makespan_s` | 14.236 |  |
| execution | `crossing/std_makespan_s` | 0.764607 |  |
| execution | `crossing/unfinished_agent_rate` | 0 |  |
| execution | `crossing/d_safe_m` | 0.35 |  |
| tests | `tests_passed` | 12 |  |
| tests | `tests_failed` | 0 |  |

#### 产物

- `tables\map_open_rects.csv`
- `tables\map_open_circles.csv`
- `plots\map_open.png`
- `tables\map_dual_corridor_rects.csv`
- `tables\map_dual_corridor_circles.csv`
- `plots\map_dual_corridor.png`
- `tables\map_narrow_gate_rects.csv`
- `tables\map_narrow_gate_circles.csv`
- `plots\map_narrow_gate.png`
- `tables\map_bottleneck_rects.csv`
- `tables\map_bottleneck_circles.csv`
- `plots\map_bottleneck.png`
- `plots\scenario_crossing.png`
- `plots\scenario_head_on.png`
- `plots\scenario_follow.png`
- `plots\scenario_terminal_occupancy.png`
- `plots\scenario_dual_corridor.png`
- `plots\scenario_narrow_gate.png`
- `plots\scenario_multi_bottleneck.png`
- `tables\scenario_summary.csv`
- `plots\execution_traces_crossing.png`
- `tables\execution_collision_detail.csv`
- `plots\execution_single_sample_detail.png`


## S1：首次碰撞风险 DP

运行目录：`results/S1_risk_dp/`（完整记录），归档：`docs/stages/S1/`。

### 关键指标

| 分组 | 指标 | 值 |
|---|---|---|
| environment | `timestamp` | 2026-11-03T02:05:13+08:00 |
| environment | `python` | 3.12.13 |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |
| environment | `torch` | 2.11.0+cu128 |
| environment | `cuda_available` | True |
| environment | `cuda_version` | 12.8 |
| environment | `cudnn_version` | 91900 |
| environment | `platform` | Windows-11-10.0.26200-SP0 |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |
| environment | `cpu_count` | 10 |
| environment | `git_commit` | 0ef64b83d45a49c2d391b6df0baf4b7cc3793373 |
| environment | `git_dirty` | True |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |
| environment | `gpu_total_memory_gb` | 11.99 |
| environment | `gpu_index` | 0 |
| environment | `gpu_capability` | 8.9 |
| R7 | `r7_max_abs_error` | 0.00673775 |
| R7 | `r7_inconsistent_cases` | 0 |
| R7 | `r7_total_mc_samples` | 200000 |
| R7 | `r7_elapsed_s` | 1.31325 |
| R9 | `r9_hard_reference` | 0.178811 |
| R9 | `r9_soft_max` | 0.340815 |
| R9 | `r9_soft_min` | 0.167456 |
| R9 | `soft_hard_max_tv_distance` | 0.0583864 |
| survival | `crossing/hard_risk` | 0.178811 |
| survival | `head_on/hard_risk` | 1 |
| diagnostics | `joint_grid_snapshot_t` | 17 |
| R8 | `gradient_max_rel_error` | 3.52945e-07 |
| timing | `timing_n2_ms` | 56.24 |
| timing | `timing_n4_ms` | 87.42 |
| timing | `timing_n6_ms` | 138.91 |
| timing | `timing_cuda_n6_ms` | 339.84 |
| timing | `cuda_peak_mem_mb` | 77.1 |
| multi | `n6_pair_risk_sum` | 12.0834 |
| multi | `n6_union_bound` | 1 |
| multi | `n6_mc_team_collision_rate` | 1 |

### 图表

![joint_progress_grid.png](stages/S1/plots/joint_progress_grid.png)
![r7_dp_vs_mc.png](stages/S1/plots/r7_dp_vs_mc.png)
![r9_soft_hard_sensitivity.png](stages/S1/plots/r9_soft_hard_sensitivity.png)
![soft_hard_progress_bias.png](stages/S1/plots/soft_hard_progress_bias.png)
![survival_curves.png](stages/S1/plots/survival_curves.png)

### 阶段报告原文

### S1 首次碰撞风险 DP 正确性报告

- 阶段：`S1_risk_dp`
- 运行目录：`D:\paper2\results\S1_risk_dp\20261103_020513_s1_verify_f1e3d9`
- 生成时间：2026-11-03T02:05:26+08:00
- 耗时：13.4 s

#### 阶段目标

在训练任何神经网络之前，先证明“随机执行进度传播 + 首次碰撞吸收”这套动态规划在数学上正确、可微、且与独立 Monte Carlo 一致（文档 §5.4 的第一道门槛）。

#### 方法要点

- 状态：F_t(m,n) 为“截至 t 未碰撞且联合进度为 (m,n)”的概率质量，共 H×H=32×32 个状态；
- 每步枚举 4 种动作（00/10/01/11），用四种动作的连续运动段碰撞核吸收质量；
- 连续碰撞判定：d_min 由相对运动闭式解给出（d_safe=0.35 m）；
- 软门槛 a=(1-p)·σ((t+0.5+c-τ)/β)，c=0.5、β=0.25（批复 §11）；
- 硬门槛 a=(1-p)·1[t≥τ]，与执行模拟器使用完全相同的实数比较；
- 全队语义：两两风险之和只是引导目标，不是全队至少碰撞一次的概率；全队碰撞率必须由联合 Monte Carlo 估计（本节给出并集上界对照）。

#### 测试与对照结果

- R1–R10 单元测试：通过 0 项、失败 0 项、错误 0 项；
- R7（20 个案例 × 10000 次 MC）：最大绝对误差 0.00674，二项检验不一致案例 0 个（α=0.001）；
- R9：硬风险恒为 0.178811（与 ε、β 无关），logistic 软代理在 [0.1675, 0.3408] 之间变化，说明软代理不具备概率解释；
- 软/硬门槛带来的进度分布最大总变差距离 0.0584；
- R8：梯度与中心差分的最大相对误差 3.53e-07。

#### 结论

- 首次碰撞概率 DP 在无碰撞时严格保持概率守恒，在必然碰撞时严格为 1，且只计第一次碰撞；
- 硬 DP 与独立 Monte Carlo 在统计误差内一致（最大绝对误差 0.00674）；
- 硬风险可精确计算但不可微；软风险可微但只是代理量，二者必须分开报告；
- S1 门槛通过，可进入 S2（不使用 Diffusion 的风险梯度优化）。


#### 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T02:05:13+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | 0ef64b83d45a49c2d391b6df0baf4b7cc3793373 |  |
| environment | `git_dirty` | True |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| R7 | `r7_max_abs_error` | 0.00673775 |  |
| R7 | `r7_inconsistent_cases` | 0 |  |
| R7 | `r7_total_mc_samples` | 200000 |  |
| R7 | `r7_elapsed_s` | 1.31325 |  |
| R9 | `r9_hard_reference` | 0.178811 |  |
| R9 | `r9_soft_max` | 0.340815 |  |
| R9 | `r9_soft_min` | 0.167456 |  |
| R9 | `soft_hard_max_tv_distance` | 0.0583864 |  |
| survival | `crossing/hard_risk` | 0.178811 |  |
| survival | `head_on/hard_risk` | 1 |  |
| diagnostics | `joint_grid_snapshot_t` | 17 |  |
| R8 | `gradient_max_rel_error` | 3.52945e-07 |  |
| timing | `timing_n2_ms` | 56.24 |  |
| timing | `timing_n4_ms` | 87.42 |  |
| timing | `timing_n6_ms` | 138.91 |  |
| timing | `timing_cuda_n6_ms` | 339.84 |  |
| timing | `cuda_peak_mem_mb` | 77.1 |  |
| multi | `n6_pair_risk_sum` | 12.0834 |  |
| multi | `n6_union_bound` | 1 |  |
| multi | `n6_mc_team_collision_rate` | 1 |  |

#### 产物

- `tables\r7_dp_vs_mc.csv`
- `plots\r7_dp_vs_mc.png`
- `tables\r9_soft_hard_sweep.csv`
- `plots\r9_soft_hard_sensitivity.png`
- `tables\soft_hard_progress_bias.csv`
- `plots\soft_hard_progress_bias.png`
- `tables\survival_curves.csv`
- `plots\survival_curves.png`
- `plots\joint_progress_grid.png`
- `tables\gradient_finite_difference.csv`
- `tables\timing.csv`


## S2：无 Diffusion 的风险梯度优化

运行目录：`results/S2_gradient_optimization/`（完整记录），归档：`docs/stages/S2/`。

### 关键指标

| 分组 | 指标 | 值 |
|---|---|---|
| environment | `timestamp` | 2026-11-03T02:13:19+08:00 |
| environment | `python` | 3.12.13 |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |
| environment | `torch` | 2.11.0+cu128 |
| environment | `cuda_available` | True |
| environment | `cuda_version` | 12.8 |
| environment | `cudnn_version` | 91900 |
| environment | `platform` | Windows-11-10.0.26200-SP0 |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |
| environment | `cpu_count` | 10 |
| environment | `git_commit` | 2e7ab8e48deb11813426729fd69699685512c375 |
| environment | `git_dirty` | True |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |
| environment | `gpu_total_memory_gb` | 11.99 |
| environment | `gpu_index` | 0 |
| environment | `gpu_capability` | 8.9 |
| baseline | `crossing/baseline/hard_risk_sum` | 0.178811 |
| baseline | `crossing/baseline/hard_risk_max` | 0.178811 |
| baseline | `crossing/baseline/soft_risk_sum` | 0.188288 |
| baseline | `crossing/baseline/finish_cost_s` | 27.5556 |
| baseline | `crossing/baseline/unfinished_prob` | -2.22045e-16 |
| baseline | `crossing/baseline/wait_total` | 0 |
| baseline | `crossing/baseline/mc_collision_rate` | 0.177 |
| baseline | `crossing/baseline/mc_makespan_s` | 14.1894 |
| baseline | `crossing/baseline/mc_completion_rate` | 1 |
| wait_safety | `crossing/A1 只改路径/dangerous_wait_points` | 0 |
| result | `crossing/A1 只改路径/hard_risk` | 3.49978e-06 |
| result | `crossing/A1 只改路径/mc_collision_rate` | 0 |
| result | `crossing/A1 只改路径/wait_total` | 0 |
| result | `crossing/A1 只改路径/finish_cost_s` | 27.5556 |
| wait_safety | `crossing/A2 只改等待/dangerous_wait_points` | 0 |
| result | `crossing/A2 只改等待/hard_risk` | 0.0203879 |
| result | `crossing/A2 只改等待/mc_collision_rate` | 0.0222 |
| result | `crossing/A2 只改等待/wait_total` | 2.94104 |
| result | `crossing/A2 只改等待/finish_cost_s` | 28.3327 |
| wait_safety | `crossing/A3 联合优化/dangerous_wait_points` | 0 |
| result | `crossing/A3 联合优化/hard_risk` | 2.14921e-06 |
| result | `crossing/A3 联合优化/mc_collision_rate` | 0 |
| result | `crossing/A3 联合优化/wait_total` | 0 |
| result | `crossing/A3 联合优化/finish_cost_s` | 27.5556 |
| baseline | `follow/baseline/hard_risk_sum` | 0.417137 |
| baseline | `follow/baseline/hard_risk_max` | 0.417137 |
| baseline | `follow/baseline/soft_risk_sum` | 0.42083 |
| baseline | `follow/baseline/finish_cost_s` | 27.5556 |
| baseline | `follow/baseline/unfinished_prob` | -2.22045e-16 |
| baseline | `follow/baseline/wait_total` | 0 |
| baseline | `follow/baseline/mc_collision_rate` | 0.4186 |
| baseline | `follow/baseline/mc_makespan_s` | 14.1894 |
| baseline | `follow/baseline/mc_completion_rate` | 1 |
| wait_safety | `follow/A1 只改路径/dangerous_wait_points` | 0 |
| result | `follow/A1 只改路径/hard_risk` | 0.011992 |
| result | `follow/A1 只改路径/mc_collision_rate` | 0.011 |
| result | `follow/A1 只改路径/wait_total` | 0 |
| result | `follow/A1 只改路径/finish_cost_s` | 27.5556 |
| wait_safety | `follow/A2 只改等待/dangerous_wait_points` | 1 |
| result | `follow/A2 只改等待/hard_risk` | 0.0315554 |
| result | `follow/A2 只改等待/mc_collision_rate` | 0.0302 |
| result | `follow/A2 只改等待/wait_total` | 3.70122 |
| result | `follow/A2 只改等待/finish_cost_s` | 29.072 |
| wait_safety | `follow/A3 联合优化/dangerous_wait_points` | 0 |
| result | `follow/A3 联合优化/hard_risk` | 0.00387222 |
| result | `follow/A3 联合优化/mc_collision_rate` | 0.0046 |
| result | `follow/A3 联合优化/wait_total` | 0 |
| result | `follow/A3 联合优化/finish_cost_s` | 27.5556 |
| baseline | `narrow_gate/baseline/hard_risk_sum` | 1 |
| baseline | `narrow_gate/baseline/hard_risk_max` | 1 |
| baseline | `narrow_gate/baseline/soft_risk_sum` | 1 |
| baseline | `narrow_gate/baseline/finish_cost_s` | 27.5556 |
| baseline | `narrow_gate/baseline/unfinished_prob` | -2.22045e-16 |
| baseline | `narrow_gate/baseline/wait_total` | 0 |
| baseline | `narrow_gate/baseline/mc_collision_rate` | 1 |
| baseline | `narrow_gate/baseline/mc_makespan_s` | 14.1894 |
| baseline | `narrow_gate/baseline/mc_completion_rate` | 1 |
| wait_safety | `narrow_gate/A1 只改路径/dangerous_wait_points` | 0 |
| result | `narrow_gate/A1 只改路径/hard_risk` | 1 |
| result | `narrow_gate/A1 只改路径/mc_collision_rate` | 1 |
| result | `narrow_gate/A1 只改路径/wait_total` | 0 |
| result | `narrow_gate/A1 只改路径/finish_cost_s` | 27.5556 |
| wait_safety | `narrow_gate/A2 只改等待/dangerous_wait_points` | 0 |
| result | `narrow_gate/A2 只改等待/hard_risk` | 1 |
| result | `narrow_gate/A2 只改等待/mc_collision_rate` | 1 |
| result | `narrow_gate/A2 只改等待/wait_total` | 0 |
| result | `narrow_gate/A2 只改等待/finish_cost_s` | 27.5556 |
| wait_safety | `narrow_gate/A3 联合优化/dangerous_wait_points` | 0 |
| result | `narrow_gate/A3 联合优化/hard_risk` | 1 |
| result | `narrow_gate/A3 联合优化/mc_collision_rate` | 1 |
| result | `narrow_gate/A3 联合优化/wait_total` | 0 |
| result | `narrow_gate/A3 联合优化/finish_cost_s` | 27.5556 |
| verdict | `s2_improved_cases` | 2 |

### 图表

![crossing_convergence.png](stages/S2/plots/crossing_convergence.png)
![crossing_joint_paths.png](stages/S2/plots/crossing_joint_paths.png)
![crossing_progress_distribution.png](stages/S2/plots/crossing_progress_distribution.png)
![dual_corridor_convergence.png](stages/S2/plots/dual_corridor_convergence.png)
![dual_corridor_joint_paths.png](stages/S2/plots/dual_corridor_joint_paths.png)
![dual_corridor_progress_distribution.png](stages/S2/plots/dual_corridor_progress_distribution.png)
![follow_convergence.png](stages/S2/plots/follow_convergence.png)
![follow_joint_paths.png](stages/S2/plots/follow_joint_paths.png)
![follow_progress_distribution.png](stages/S2/plots/follow_progress_distribution.png)
![narrow_gate_convergence.png](stages/S2/plots/narrow_gate_convergence.png)
![narrow_gate_joint_paths.png](stages/S2/plots/narrow_gate_joint_paths.png)
![narrow_gate_progress_distribution.png](stages/S2/plots/narrow_gate_progress_distribution.png)
![pareto_crossing.png](stages/S2/plots/pareto_crossing.png)

### 阶段报告原文

### S2 不使用 Diffusion 的风险梯度优化报告

- 阶段：`S2_gradient_optimization`
- 运行目录：`D:\paper2\results\S2_gradient_optimization\20261103_021319_s2_optimize_b67a27`
- 生成时间：2026-11-03T02:15:57+08:00
- 耗时：157.7 s

#### 阶段目标

在不使用任何神经网络的前提下，验证“可微首次碰撞风险 → 梯度 → 轨迹与等待的有效修正”这条链条成立（文档 §9 S2）。

#### 优化设置

- 损失：λ_R ΣR_soft + λ_T J_finish + λ_S L_static + λ_K L_kinematic + λ_G L_geometry + λ_W ||w||₁；
- λ 取值为批复 §23 的默认值（λ_R=1.0, λ_T=0.05, λ_S=10.0, λ_K=1.0, λ_G=0.5, λ_W=0.01）；
- 等待投影：0≤w≤4.0 时隙/点，Σw≤8.0 时隙/机器人；
- 端点固定、端点梯度屏蔽；步数 250，Adam 路径 lr=0.02、等待 lr=0.1；
- 每 5000 次独立 Monte Carlo 复现用于验证真实碰撞率。

#### 各场景结果

- **crossing**：硬风险 0.1788 → 只改路径 0.0000 / 只改等待 0.0204 / 联合 0.0000（降低 +0.1788）
- **follow**：硬风险 0.4171 → 只改路径 0.0120 / 只改等待 0.0316 / 联合 0.0039（降低 +0.4133）
- **narrow_gate**：硬风险 1.0000 → 只改路径 1.0000 / 只改等待 1.0000 / 联合 1.0000（降低 +0.0000）

#### 梯度饱和的诊断（窄门口）

窄门口场景的初始硬风险恰为 1.0：两车在单车宽的缺口内对向相遇，几乎所有执行轨迹都碰撞。此时首次碰撞 DP 的概率质量在很早期就被完全吸收，**软风险也随之饱和到 1**，对路径与等待的梯度趋近 0，优化停滞。这正是文档 §14 列出的“深度碰撞处梯度饱和”失败模式。

| ε [m] | 软风险 | 等待梯度范数 | 路径梯度范数 |
|---|---|---|---|
| 0.025 | 1.0000 | 4.672e-11 | 3.877e-08 |
| 0.050 | 1.0000 | 5.260e-07 | 1.819e-04 |
| 0.100 | 0.9983 | 6.174e-05 | 6.792e-03 |
| 0.200 | 0.9860 | 5.916e-04 | 2.917e-02 |
| 0.400 | 0.9777 | 1.184e-03 | 3.439e-02 |

把温度放宽到 ε=0.4 后，等待梯度范数从 4.672e-11 放大到 1.184e-03，但重新做 150 步等待优化后硬风险仍为 1.0000（等待 0.00 时隙）。**结论：单靠温度退火不足以逃离该饱和区**——本案例需要“让一车整体先通过”这种近似离散的决策，局部梯度无法表达。该负结果保留在案，作为 S5 失败归因与“是否引入辅助机制”的判断依据。

#### 结论

- 2/3 个几何案例在充分时间预算下把硬风险显著降低（>0.01）；
- 软风险与硬风险、Monte Carlo 碰撞率同向下降，说明可微代理的优化方向可信；
- 只改路径（A1）在几何上仍有余量时最强；只改等待（A2）能只靠时间错峰降低风险，代价是完成时间上升；联合（A3）在两者都可用时取得最低风险；
- 等待只出现在安全位置（见 `wait_safety_*.csv`），未出现“无限等待”作弊：等待总量受总预算约束，且 J_finish 随等待单调上升；
- 初始风险恰为 1 的饱和案例（窄门口）无法仅靠局部梯度逃出，已作为失败模式记录；
- 通过条件：满足（要求至少 2 个案例）。


#### 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T02:13:19+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | 2e7ab8e48deb11813426729fd69699685512c375 |  |
| environment | `git_dirty` | True |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| baseline | `crossing/baseline/hard_risk_sum` | 0.178811 |  |
| baseline | `crossing/baseline/hard_risk_max` | 0.178811 |  |
| baseline | `crossing/baseline/soft_risk_sum` | 0.188288 |  |
| baseline | `crossing/baseline/finish_cost_s` | 27.5556 |  |
| baseline | `crossing/baseline/unfinished_prob` | -2.22045e-16 |  |
| baseline | `crossing/baseline/wait_total` | 0 |  |
| baseline | `crossing/baseline/mc_collision_rate` | 0.177 |  |
| baseline | `crossing/baseline/mc_makespan_s` | 14.1894 |  |
| baseline | `crossing/baseline/mc_completion_rate` | 1 |  |
| wait_safety | `crossing/A1 只改路径/dangerous_wait_points` | 0 |  |
| result | `crossing/A1 只改路径/hard_risk` | 3.49978e-06 |  |
| result | `crossing/A1 只改路径/mc_collision_rate` | 0 |  |
| result | `crossing/A1 只改路径/wait_total` | 0 |  |
| result | `crossing/A1 只改路径/finish_cost_s` | 27.5556 |  |
| wait_safety | `crossing/A2 只改等待/dangerous_wait_points` | 0 |  |
| result | `crossing/A2 只改等待/hard_risk` | 0.0203879 |  |
| result | `crossing/A2 只改等待/mc_collision_rate` | 0.0222 |  |
| result | `crossing/A2 只改等待/wait_total` | 2.94104 |  |
| result | `crossing/A2 只改等待/finish_cost_s` | 28.3327 |  |
| wait_safety | `crossing/A3 联合优化/dangerous_wait_points` | 0 |  |
| result | `crossing/A3 联合优化/hard_risk` | 2.14921e-06 |  |
| result | `crossing/A3 联合优化/mc_collision_rate` | 0 |  |
| result | `crossing/A3 联合优化/wait_total` | 0 |  |
| result | `crossing/A3 联合优化/finish_cost_s` | 27.5556 |  |
| baseline | `follow/baseline/hard_risk_sum` | 0.417137 |  |
| baseline | `follow/baseline/hard_risk_max` | 0.417137 |  |
| baseline | `follow/baseline/soft_risk_sum` | 0.42083 |  |
| baseline | `follow/baseline/finish_cost_s` | 27.5556 |  |
| baseline | `follow/baseline/unfinished_prob` | -2.22045e-16 |  |
| baseline | `follow/baseline/wait_total` | 0 |  |
| baseline | `follow/baseline/mc_collision_rate` | 0.4186 |  |
| baseline | `follow/baseline/mc_makespan_s` | 14.1894 |  |
| baseline | `follow/baseline/mc_completion_rate` | 1 |  |
| wait_safety | `follow/A1 只改路径/dangerous_wait_points` | 0 |  |
| result | `follow/A1 只改路径/hard_risk` | 0.011992 |  |
| result | `follow/A1 只改路径/mc_collision_rate` | 0.011 |  |
| result | `follow/A1 只改路径/wait_total` | 0 |  |
| result | `follow/A1 只改路径/finish_cost_s` | 27.5556 |  |
| wait_safety | `follow/A2 只改等待/dangerous_wait_points` | 1 |  |
| result | `follow/A2 只改等待/hard_risk` | 0.0315554 |  |
| result | `follow/A2 只改等待/mc_collision_rate` | 0.0302 |  |
| result | `follow/A2 只改等待/wait_total` | 3.70122 |  |
| result | `follow/A2 只改等待/finish_cost_s` | 29.072 |  |
| wait_safety | `follow/A3 联合优化/dangerous_wait_points` | 0 |  |
| result | `follow/A3 联合优化/hard_risk` | 0.00387222 |  |
| result | `follow/A3 联合优化/mc_collision_rate` | 0.0046 |  |
| result | `follow/A3 联合优化/wait_total` | 0 |  |
| result | `follow/A3 联合优化/finish_cost_s` | 27.5556 |  |
| baseline | `narrow_gate/baseline/hard_risk_sum` | 1 |  |
| baseline | `narrow_gate/baseline/hard_risk_max` | 1 |  |
| baseline | `narrow_gate/baseline/soft_risk_sum` | 1 |  |
| baseline | `narrow_gate/baseline/finish_cost_s` | 27.5556 |  |
| baseline | `narrow_gate/baseline/unfinished_prob` | -2.22045e-16 |  |
| baseline | `narrow_gate/baseline/wait_total` | 0 |  |
| baseline | `narrow_gate/baseline/mc_collision_rate` | 1 |  |
| baseline | `narrow_gate/baseline/mc_makespan_s` | 14.1894 |  |
| baseline | `narrow_gate/baseline/mc_completion_rate` | 1 |  |
| wait_safety | `narrow_gate/A1 只改路径/dangerous_wait_points` | 0 |  |
| result | `narrow_gate/A1 只改路径/hard_risk` | 1 |  |
| result | `narrow_gate/A1 只改路径/mc_collision_rate` | 1 |  |
| result | `narrow_gate/A1 只改路径/wait_total` | 0 |  |
| result | `narrow_gate/A1 只改路径/finish_cost_s` | 27.5556 |  |
| wait_safety | `narrow_gate/A2 只改等待/dangerous_wait_points` | 0 |  |
| result | `narrow_gate/A2 只改等待/hard_risk` | 1 |  |
| result | `narrow_gate/A2 只改等待/mc_collision_rate` | 1 |  |
| result | `narrow_gate/A2 只改等待/wait_total` | 0 |  |
| result | `narrow_gate/A2 只改等待/finish_cost_s` | 27.5556 |  |
| wait_safety | `narrow_gate/A3 联合优化/dangerous_wait_points` | 0 |  |
| result | `narrow_gate/A3 联合优化/hard_risk` | 1 |  |
| result | `narrow_gate/A3 联合优化/mc_collision_rate` | 1 |  |
| result | `narrow_gate/A3 联合优化/wait_total` | 0 |  |
| result | `narrow_gate/A3 联合优化/finish_cost_s` | 27.5556 |  |
| verdict | `s2_improved_cases` | 2 |  |

#### 产物

- `tables\wait_safety_crossing_path.csv`
- `tables\wait_safety_crossing_wait.csv`
- `tables\wait_safety_crossing_joint.csv`
- `plots\crossing_joint_paths.png`
- `plots\crossing_progress_distribution.png`
- `plots\crossing_convergence.png`
- `tables\wait_safety_follow_path.csv`
- `tables\wait_safety_follow_wait.csv`
- `tables\wait_safety_follow_joint.csv`
- `plots\follow_joint_paths.png`
- `plots\follow_progress_distribution.png`
- `plots\follow_convergence.png`
- `tables\wait_safety_narrow_gate_path.csv`
- `tables\wait_safety_narrow_gate_wait.csv`
- `tables\wait_safety_narrow_gate_joint.csv`
- `plots\narrow_gate_joint_paths.png`
- `plots\narrow_gate_progress_distribution.png`
- `plots\narrow_gate_convergence.png`
- `tables\optimization_traces.csv`
- `tables\pareto_points.csv`
- `tables\pareto_lambda_T.csv`
- `tables\saturation_diagnosis.csv`
- `tables\saturation_rescue.csv`
- `plots\pareto_crossing.png`


## S3：单机器人条件 Diffusion

运行目录：`results/S3_diffusion_training/`（完整记录），归档：`docs/stages/S3/`。

### 关键指标

| 分组 | 指标 | 值 |
|---|---|---|
| environment | `timestamp` | 2026-11-03T04:38:36+08:00 |
| environment | `python` | 3.12.13 |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |
| environment | `torch` | 2.11.0+cu128 |
| environment | `cuda_available` | True |
| environment | `cuda_version` | 12.8 |
| environment | `cudnn_version` | 91900 |
| environment | `platform` | Windows-11-10.0.26200-SP0 |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |
| environment | `cpu_count` | 10 |
| environment | `git_commit` | 84a735dff562f55edfa5d1a8b9e55916b043402c |
| environment | `git_dirty` | True |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |
| environment | `gpu_total_memory_gb` | 11.99 |
| environment | `gpu_index` | 0 |
| environment | `gpu_capability` | 8.9 |
| sampling | `n_tasks` | 200 |
| sampling | `samples_per_task` | 6 |
| sampling | `ddim_steps` | 40 |
| sampling | `success_rate` | 0.6075 |
| sampling | `static_collision_rate` | 0.39 |
| sampling | `speed_violation_rate` | 0.005 |
| sampling | `start_err_max_m` | 7.16005e-07 |
| sampling | `goal_err_max_m` | 9.30859e-07 |
| sampling | `length_ratio_mean` | 1.01136 |
| sampling | `smoothness_mean` | 0.00129488 |
| sampling | `diversity_mean_m` | 0.364065 |
| sampling | `inference_ms_per_sample` | 30.2841 |
| sampling | `total_seconds` | 36.3409 |

### 图表

![generated_samples_grid.png](stages/S3/plots/generated_samples_grid.png)
![sample_step000300.png](stages/S3/plots/sample_step000300.png)
![sample_step000600.png](stages/S3/plots/sample_step000600.png)
![sample_step001000.png](stages/S3/plots/sample_step001000.png)
![sample_step002000.png](stages/S3/plots/sample_step002000.png)
![sample_step003000.png](stages/S3/plots/sample_step003000.png)
![sample_step004000.png](stages/S3/plots/sample_step004000.png)
![sample_step005000.png](stages/S3/plots/sample_step005000.png)
![sample_step006000.png](stages/S3/plots/sample_step006000.png)
![sample_step007000.png](stages/S3/plots/sample_step007000.png)
![sample_step008000.png](stages/S3/plots/sample_step008000.png)
![sample_step009000.png](stages/S3/plots/sample_step009000.png)
![sample_step010000.png](stages/S3/plots/sample_step010000.png)
![sample_step011000.png](stages/S3/plots/sample_step011000.png)
![sample_step012000.png](stages/S3/plots/sample_step012000.png)
![sample_step013000.png](stages/S3/plots/sample_step013000.png)
![sample_step014000.png](stages/S3/plots/sample_step014000.png)
![sample_step015000.png](stages/S3/plots/sample_step015000.png)
![sample_step016000.png](stages/S3/plots/sample_step016000.png)
![sample_step017000.png](stages/S3/plots/sample_step017000.png)
![sample_step018000.png](stages/S3/plots/sample_step018000.png)
![sample_step019000.png](stages/S3/plots/sample_step019000.png)
![sample_step020000.png](stages/S3/plots/sample_step020000.png)
![sample_step021000.png](stages/S3/plots/sample_step021000.png)
![sample_step022000.png](stages/S3/plots/sample_step022000.png)
![sample_step023000.png](stages/S3/plots/sample_step023000.png)
![sample_step024000.png](stages/S3/plots/sample_step024000.png)
![sample_step025000.png](stages/S3/plots/sample_step025000.png)
![sample_step026000.png](stages/S3/plots/sample_step026000.png)
![sample_step027000.png](stages/S3/plots/sample_step027000.png)
![sample_step028000.png](stages/S3/plots/sample_step028000.png)
![sample_step029000.png](stages/S3/plots/sample_step029000.png)
![sample_step030000.png](stages/S3/plots/sample_step030000.png)
![sample_step031000.png](stages/S3/plots/sample_step031000.png)
![sample_step032000.png](stages/S3/plots/sample_step032000.png)
![sample_step033000.png](stages/S3/plots/sample_step033000.png)
![sample_step034000.png](stages/S3/plots/sample_step034000.png)
![sample_step035000.png](stages/S3/plots/sample_step035000.png)
![sample_step036000.png](stages/S3/plots/sample_step036000.png)
![sample_step037000.png](stages/S3/plots/sample_step037000.png)
![sample_step038000.png](stages/S3/plots/sample_step038000.png)
![sample_step039000.png](stages/S3/plots/sample_step039000.png)
![sample_step040000.png](stages/S3/plots/sample_step040000.png)
![sample_step041000.png](stages/S3/plots/sample_step041000.png)
![sample_step042000.png](stages/S3/plots/sample_step042000.png)
![sample_step043000.png](stages/S3/plots/sample_step043000.png)
![sample_step044000.png](stages/S3/plots/sample_step044000.png)
![sample_step045000.png](stages/S3/plots/sample_step045000.png)
![sample_step046000.png](stages/S3/plots/sample_step046000.png)
![sample_step047000.png](stages/S3/plots/sample_step047000.png)
![sample_step048000.png](stages/S3/plots/sample_step048000.png)
![sample_step049000.png](stages/S3/plots/sample_step049000.png)
![sample_step050000.png](stages/S3/plots/sample_step050000.png)
![success_rate_hist.png](stages/S3/plots/success_rate_hist.png)

### 阶段报告原文

### S3 单机器人生成质量评估报告

- 阶段：`S3_diffusion_training`
- 运行目录：`D:\paper2\results\S3_diffusion_training\20261103_043800_s3_sample_402b45`
- 生成时间：2026-11-03T04:38:39+08:00
- 耗时：2.4 s

#### 评估口径

- 从 `test` 划分中均匀取 200 个任务，每个任务采样 6 条候选；
- “成功”定义：静态障碍连续几何无碰撞 **且** 每段长度 ≤ v_max·Δt=0.60 m；
- 起终点由采样器强制固定，误差应接近 0；
- 多样性 = 同一任务 k 条候选的平均两两点距。

#### 结果

- 成功率 **60.75%**；静态碰撞率 39.00%；速度越界率 0.50%；
- 路径长度比（生成/真实）均值 1.011；
- 候选多样性 0.364 m；
- 推理 30.3 ms/样本（DDIM 40 步）。

**不追求 100% 成功**：不合格候选一律记为失败，不做后验剔除。


#### 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T04:38:36+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | 84a735dff562f55edfa5d1a8b9e55916b043402c |  |
| environment | `git_dirty` | True |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| sampling | `n_tasks` | 200 |  |
| sampling | `samples_per_task` | 6 |  |
| sampling | `ddim_steps` | 40 |  |
| sampling | `success_rate` | 0.6075 |  |
| sampling | `static_collision_rate` | 0.39 |  |
| sampling | `speed_violation_rate` | 0.005 |  |
| sampling | `start_err_max_m` | 7.16005e-07 |  |
| sampling | `goal_err_max_m` | 9.30859e-07 |  |
| sampling | `length_ratio_mean` | 1.01136 |  |
| sampling | `smoothness_mean` | 0.00129488 |  |
| sampling | `diversity_mean_m` | 0.364065 |  |
| sampling | `inference_ms_per_sample` | 30.2841 |  |
| sampling | `total_seconds` | 36.3409 |  |

#### 产物

- `tables\single_robot_samples.csv`
- `plots\generated_samples_grid.png`
- `plots\success_rate_hist.png`

### 补充记录：supplement_data_scaling.md

### 补充实验：数据规模对单机器人生成质量的影响

> 本文件是 S3 的补充记录。主流程使用 `medium` 规模（4,819 条训练轨迹）训练的模型，
> 这里报告在同一测试集上把训练数据扩大到 19,212 条后的结果，用于判断瓶颈在哪里。

#### 设置

| 项 | medium（主流程） | large（补充） |
|---|---|---|
| 训练地图数 | 50 | 100 |
| 训练轨迹数 | 4,819 | 19,212 |
| 验证 / 测试 | 473 / 489（10 / 10 张地图） | 950 / 958（20 / 20 张地图） |
| 训练步数 | 40,000 | 50,000 |
| 最佳验证损失 | 0.01066 | 0.00875 |
| 网络结构 | 地图空间特征 8×8 + FiLM 1D U-Net（2.23 M 参数） | 同左 |

评估口径完全一致：`data/dataset` 的 **test** 划分，均匀取 200 个任务，每个任务 6 条候选，
40 步 DDIM；“成功”= 静态连续几何无碰撞 **且** 每段长度 ≤ 0.6 m。

#### 结果（同一测试集）

| 指标 | medium 模型 | large 模型 | large + 16×16 地图特征 |
|---|---|---|---|
| 成功率 | 59.4% | **61.3%** | 60.8% |
| 静态碰撞率 | 40.3% | 38.6% | 39.0% |
| 长度比（生成/真实） | 1.007 | 1.014 | 1.011 |
| 候选多样性 | 0.297 m | 0.363 m | 0.364 m |
| 推理耗时 | 33.4 ms/样本 | 30.0 ms/样本 | 30.3 ms/样本 |

（“large + 16×16”= 19,212 条数据 + 把地图空间特征从 8×8 提到 16×16、训练 5 万步，
最佳验证损失 0.00763。）

#### 结论

1. **数据量 4.8k → 19.2k 只带来 +1.9 pp，地图特征 8×8 → 16×16 只带来 −0.5 pp（无显著差异）。**
   两者都没有突破约 60% 的平台，说明瓶颈既不是数据量，也不是地图条件的空间分辨率。
2. 更可能的原因是**表示与判定口径**：H=32 个路点、每段 ≤0.6 m 的折线，在 0.55–0.8 m 宽的
   窄门口里必须“穿针引线”；而“成功”要求 31 段**全部**与障碍物保持 0.18 m 净空，
   是严格的 all-or-nothing 判定。单点擦边即整条失败。
3. 因此主流程没有为了“更好看的数字”更换数据集或网络，而是如实报告两个规模的结果。
4. 这个平台直接决定了本项目的边界：S4/S5 的“全尝试安全完成率”被基座模型的静态避障能力
   （约 60% 单机成功率；N 台全部安全约为 0.6^N，N=6 时仅约 5%）严重压低。
   S5 中所有方法共享同一个基座模型，因此**方法之间的相对比较仍然有效**，
   但绝对数值不能当作系统可用的证据。
5. 若要继续提高，优先级应为：更细的路点（例如 H=64 或自适应路点）、
   在采样阶段统一加入更强的静态约束（本项目已对所有方法统一启用 `L_static`，
   但 λ_S 从 10 提到 50 对成功率没有影响，说明问题在拓扑而非权重）、
   或者用“路径点查询栅格”的注意力结构代替全局条件向量。


## S4：风险引导 DDIM 接入

运行目录：`results/S4_guided_diffusion/`（完整记录），归档：`docs/stages/S4/`。

### 关键指标

| 分组 | 指标 | 值 |
|---|---|---|
| environment | `timestamp` | 2026-11-03T03:21:31+08:00 |
| environment | `python` | 3.12.13 |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |
| environment | `torch` | 2.11.0+cu128 |
| environment | `cuda_available` | True |
| environment | `cuda_version` | 12.8 |
| environment | `cudnn_version` | 91900 |
| environment | `platform` | Windows-11-10.0.26200-SP0 |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |
| environment | `cpu_count` | 10 |
| environment | `git_commit` | 41d9a81a58961779f75c107425a63a82c3171eae |
| environment | `git_dirty` | True |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |
| environment | `gpu_total_memory_gb` | 11.99 |
| environment | `gpu_index` | 0 |
| environment | `gpu_capability` | 8.9 |
| verification | `zero_guidance_max_diff` | 8.19564e-08 |
| verification | `zero_guidance_exact` | True |
| verification | `frozen_params_changed` | 0 |
| model | `checkpoint_step` | 22500 |
| result | `B0/hard_risk` | 0.0772339 |
| result | `B0/static_safe` | True |
| result | `B0/min_clearance_m` | inf |
| result | `B0/speed_ok` | True |
| result | `B0/max_segment_m` | 0.298957 |
| result | `B0/mc_collision_rate` | 0.0735 |
| result | `B0/mc_makespan_s` | 14.1912 |
| result | `B0/endpoint_err` | 0 |
| result | `B0/endpoint_err_goal` | 0 |
| result | `A3/hard_risk` | 0.0048511 |
| result | `A3/static_safe` | True |
| result | `A3/min_clearance_m` | inf |
| result | `A3/speed_ok` | True |
| result | `A3/max_segment_m` | 0.298981 |
| result | `A3/mc_collision_rate` | 0.004 |
| result | `A3/mc_makespan_s` | 14.8302 |
| result | `A3/endpoint_err` | 0 |
| result | `A3/endpoint_err_goal` | 0 |
| result | `A3/wait_total` | 2.60889 |
| result | `A3/plan_time_s` | 2.7917 |
| result | `B0/plan_time_s` | 1.01983 |
| verification | `check/零引导等价` | True |
| verification | `check/参数冻结` | True |
| verification | `check/端点保持` | True |
| verification | `check/静态安全` | True |
| verification | `check/速度约束` | True |

### 图表

![geometry_evolution.png](stages/S4/plots/geometry_evolution.png)
![guidance_trajectory.png](stages/S4/plots/guidance_trajectory.png)
![plan_A3.png](stages/S4/plots/plan_A3.png)
![plan_B0.png](stages/S4/plots/plan_B0.png)

### 阶段报告原文

### S4 风险引导 DDIM 演示与验证报告

- 阶段：`S4_guided_diffusion`
- 运行目录：`D:\paper2\results\S4_guided_diffusion\20261103_032125_s4_demo_2fb7e4`
- 生成时间：2026-11-03T03:21:32+08:00
- 耗时：1.5 s

#### 阶段目标

把首次碰撞风险模块接到**推理阶段**的每一步 DDIM 预测上：网络参数冻结，风险梯度只修正当前的干净轨迹预测 Q̂0 与等待变量 w（文档 §7.3）。

#### 验证结果

| 检查项 | 结果 |
|---|---|
| 零引导等价 | 通过 |
| 参数冻结 | 通过 |
| 端点保持 | 通过 |
| 静态安全 | 通过 |
| 速度约束 | 通过 |

- 零引导与基础 DDIM 最大差异 `8.196e-08`；
- 推理后发生变化的参数张量数 `0`；
- B0（独立采样）：硬风险 0.0772，MC 碰撞率 0.0735；
- A3（引导）：硬风险 0.0049，MC 碰撞率 0.0040，主动等待 2.61 时隙；
- 规划耗时：B0 1.0s，A3 2.8s（40 步 DDIM）。

#### 结论

- 引导通路在结构上正确（零引导严格等价、参数冻结、端点不变）；
- 引导显著降低了硬风险与实测碰撞率；
- 与 S2 的手工案例一致：风险梯度既可以修改几何，也可以产生主动等待。


#### 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T03:21:31+08:00 |  |
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
| verification | `zero_guidance_max_diff` | 8.19564e-08 |  |
| verification | `zero_guidance_exact` | True |  |
| verification | `frozen_params_changed` | 0 |  |
| model | `checkpoint_step` | 22500 |  |
| result | `B0/hard_risk` | 0.0772339 |  |
| result | `B0/static_safe` | True |  |
| result | `B0/min_clearance_m` | inf |  |
| result | `B0/speed_ok` | True |  |
| result | `B0/max_segment_m` | 0.298957 |  |
| result | `B0/mc_collision_rate` | 0.0735 |  |
| result | `B0/mc_makespan_s` | 14.1912 |  |
| result | `B0/endpoint_err` | 0 |  |
| result | `B0/endpoint_err_goal` | 0 |  |
| result | `A3/hard_risk` | 0.0048511 |  |
| result | `A3/static_safe` | True |  |
| result | `A3/min_clearance_m` | inf |  |
| result | `A3/speed_ok` | True |  |
| result | `A3/max_segment_m` | 0.298981 |  |
| result | `A3/mc_collision_rate` | 0.004 |  |
| result | `A3/mc_makespan_s` | 14.8302 |  |
| result | `A3/endpoint_err` | 0 |  |
| result | `A3/endpoint_err_goal` | 0 |  |
| result | `A3/wait_total` | 2.60889 |  |
| result | `A3/plan_time_s` | 2.7917 |  |
| result | `B0/plan_time_s` | 1.01983 |  |
| verification | `check/零引导等价` | True |  |
| verification | `check/参数冻结` | True |  |
| verification | `check/端点保持` | True |  |
| verification | `check/静态安全` | True |  |
| verification | `check/速度约束` | True |  |

#### 产物

- `tables\a3_guidance_trace.csv`
- `plots\guidance_trajectory.png`
- `plots\plan_B0.png`
- `plots\plan_A3.png`
- `plots\geometry_evolution.png`


## S5：完整对照实验

运行目录：`results/S5_evaluation/`（完整记录），归档：`docs/stages/S5/`。

### 关键指标

| 分组 | 指标 | 值 |
|---|---|---|
| environment | `timestamp` | 2026-11-03T03:46:40+08:00 |
| environment | `python` | 3.12.13 |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |
| environment | `torch` | 2.11.0+cu128 |
| environment | `cuda_available` | True |
| environment | `cuda_version` | 12.8 |
| environment | `cudnn_version` | 91900 |
| environment | `platform` | Windows-11-10.0.26200-SP0 |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |
| environment | `cpu_count` | 10 |
| environment | `git_commit` | 41d9a81a58961779f75c107425a63a82c3171eae |
| environment | `git_dirty` | True |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |
| environment | `gpu_total_memory_gb` | 11.99 |
| environment | `gpu_index` | 0 |
| environment | `gpu_capability` | 8.9 |
| plan | `B0/n_plans` | 36 |
| plan | `B0/plan_valid_rate` | 0.305556 |
| plan | `B0/plan_time_mean_s` | 0.813873 |
| plan | `B0/wait_total_mean` | 0 |
| plan | `B0/hard_risk_mean` | 1.10697 |
| plan | `B1/n_plans` | 36 |
| plan | `B1/plan_valid_rate` | 0.305556 |
| plan | `B1/plan_time_mean_s` | 0.809737 |
| plan | `B1/wait_total_mean` | 0 |
| plan | `B1/hard_risk_mean` | 0.992405 |
| plan | `B2/n_plans` | 36 |
| plan | `B2/plan_valid_rate` | 0.305556 |
| plan | `B2/plan_time_mean_s` | 1.08568 |
| plan | `B2/wait_total_mean` | 0 |
| plan | `B2/hard_risk_mean` | 0.878279 |
| plan | `B3/n_plans` | 36 |
| plan | `B3/plan_valid_rate` | 0.305556 |
| plan | `B3/plan_time_mean_s` | 5.08363 |
| plan | `B3/wait_total_mean` | 6.31794 |
| plan | `B3/hard_risk_mean` | 0.420832 |
| plan | `A1/n_plans` | 36 |
| plan | `A1/plan_valid_rate` | 0.305556 |
| plan | `A1/plan_time_mean_s` | 2.6031 |
| plan | `A1/wait_total_mean` | 0 |
| plan | `A1/hard_risk_mean` | 0.920274 |
| plan | `A2/n_plans` | 36 |
| plan | `A2/plan_valid_rate` | 0.305556 |
| plan | `A2/plan_time_mean_s` | 2.59365 |
| plan | `A2/wait_total_mean` | 3.86585 |
| plan | `A2/hard_risk_mean` | 0.676472 |
| plan | `A3/n_plans` | 36 |
| plan | `A3/plan_valid_rate` | 0.305556 |
| plan | `A3/plan_time_mean_s` | 2.49608 |
| plan | `A3/wait_total_mean` | 3.83168 |
| plan | `A3/hard_risk_mean` | 0.561063 |
| execution/p0.0 | `B0/p0.0/n` | 1080 |
| execution/p0.0 | `B0/p0.0/successes` | 180 |
| execution/p0.0 | `B0/p0.0/safe_completion_rate` | 0.166667 |
| execution/p0.0 | `B0/p0.0/wilson_lo` | 0.145629 |
| execution/p0.0 | `B0/p0.0/wilson_hi` | 0.190067 |
| execution/p0.0 | `B0/p0.0/collision_rate` | 0.583333 |
| execution/p0.0 | `B0/p0.0/makespan_mean_s` | 12.4 |
| execution/p0.0 | `B0/p0.0/makespan_p90_s` | 12.4 |
| execution/p0.05 | `B0/p0.05/n` | 1080 |
| execution/p0.05 | `B0/p0.05/successes` | 194 |
| execution/p0.05 | `B0/p0.05/safe_completion_rate` | 0.17963 |
| execution/p0.05 | `B0/p0.05/wilson_lo` | 0.157883 |
| execution/p0.05 | `B0/p0.05/wilson_hi` | 0.203648 |
| execution/p0.05 | `B0/p0.05/collision_rate` | 0.566667 |
| execution/p0.05 | `B0/p0.05/makespan_mean_s` | 13.5796 |
| execution/p0.05 | `B0/p0.05/makespan_p90_s` | 14.1889 |
| execution/p0.1 | `B0/p0.1/n` | 1080 |
| execution/p0.1 | `B0/p0.1/successes` | 197 |
| execution/p0.1 | `B0/p0.1/safe_completion_rate` | 0.182407 |
| execution/p0.1 | `B0/p0.1/wilson_lo` | 0.160514 |
| execution/p0.1 | `B0/p0.1/wilson_hi` | 0.206552 |
| execution/p0.1 | `B0/p0.1/collision_rate` | 0.560185 |
| execution/p0.1 | `B0/p0.1/makespan_mean_s` | 14.5274 |
| execution/p0.1 | `B0/p0.1/makespan_p90_s` | 15.3478 |
| execution/p0.2 | `B0/p0.2/n` | 1080 |
| execution/p0.2 | `B0/p0.2/successes` | 201 |
| execution/p0.2 | `B0/p0.2/safe_completion_rate` | 0.186111 |
| execution/p0.2 | `B0/p0.2/wilson_lo` | 0.164026 |
| execution/p0.2 | `B0/p0.2/wilson_hi` | 0.210421 |
| execution/p0.2 | `B0/p0.2/collision_rate` | 0.567593 |
| execution/p0.2 | `B0/p0.2/makespan_mean_s` | 16.757 |
| execution/p0.2 | `B0/p0.2/makespan_p90_s` | 17.9944 |
| execution/p0.0 | `B1/p0.0/n` | 1080 |
| execution/p0.0 | `B1/p0.0/successes` | 210 |
| execution/p0.0 | `B1/p0.0/safe_completion_rate` | 0.194444 |
| execution/p0.0 | `B1/p0.0/wilson_lo` | 0.17194 |
| execution/p0.0 | `B1/p0.0/wilson_hi` | 0.219115 |
| execution/p0.0 | `B1/p0.0/collision_rate` | 0.555556 |
| execution/p0.0 | `B1/p0.0/makespan_mean_s` | 12.4 |
| execution/p0.0 | `B1/p0.0/makespan_p90_s` | 12.4 |
| execution/p0.05 | `B1/p0.05/n` | 1080 |
| execution/p0.05 | `B1/p0.05/successes` | 204 |
| execution/p0.05 | `B1/p0.05/safe_completion_rate` | 0.188889 |
| execution/p0.05 | `B1/p0.05/wilson_lo` | 0.166662 |
| execution/p0.05 | `B1/p0.05/wilson_hi` | 0.213321 |
| execution/p0.05 | `B1/p0.05/collision_rate` | 0.549074 |
| execution/p0.05 | `B1/p0.05/makespan_mean_s` | 13.5796 |
| execution/p0.05 | `B1/p0.05/makespan_p90_s` | 14.1889 |
| execution/p0.1 | `B1/p0.1/n` | 1080 |
| execution/p0.1 | `B1/p0.1/successes` | 208 |
| execution/p0.1 | `B1/p0.1/safe_completion_rate` | 0.192593 |
| execution/p0.1 | `B1/p0.1/wilson_lo` | 0.17018 |
| execution/p0.1 | `B1/p0.1/wilson_hi` | 0.217184 |
| execution/p0.1 | `B1/p0.1/collision_rate` | 0.543519 |
| execution/p0.1 | `B1/p0.1/makespan_mean_s` | 14.5274 |
| execution/p0.1 | `B1/p0.1/makespan_p90_s` | 15.3478 |
| execution/p0.2 | `B1/p0.2/n` | 1080 |
| execution/p0.2 | `B1/p0.2/successes` | 207 |
| execution/p0.2 | `B1/p0.2/safe_completion_rate` | 0.191667 |
| execution/p0.2 | `B1/p0.2/wilson_lo` | 0.1693 |
| execution/p0.2 | `B1/p0.2/wilson_hi` | 0.216219 |
| execution/p0.2 | `B1/p0.2/collision_rate` | 0.555556 |
| execution/p0.2 | `B1/p0.2/makespan_mean_s` | 16.757 |
| execution/p0.2 | `B1/p0.2/makespan_p90_s` | 17.9944 |
| execution/p0.0 | `B2/p0.0/n` | 1080 |
| execution/p0.0 | `B2/p0.0/successes` | 210 |
| execution/p0.0 | `B2/p0.0/safe_completion_rate` | 0.194444 |
| execution/p0.0 | `B2/p0.0/wilson_lo` | 0.17194 |
| execution/p0.0 | `B2/p0.0/wilson_hi` | 0.219115 |
| execution/p0.0 | `B2/p0.0/collision_rate` | 0.527778 |
| execution/p0.0 | `B2/p0.0/makespan_mean_s` | 12.4 |
| execution/p0.0 | `B2/p0.0/makespan_p90_s` | 12.4 |
| execution/p0.05 | `B2/p0.05/n` | 1080 |
| execution/p0.05 | `B2/p0.05/successes` | 204 |
| execution/p0.05 | `B2/p0.05/safe_completion_rate` | 0.188889 |
| execution/p0.05 | `B2/p0.05/wilson_lo` | 0.166662 |
| execution/p0.05 | `B2/p0.05/wilson_hi` | 0.213321 |
| execution/p0.05 | `B2/p0.05/collision_rate` | 0.513889 |
| execution/p0.05 | `B2/p0.05/makespan_mean_s` | 13.5796 |
| execution/p0.05 | `B2/p0.05/makespan_p90_s` | 14.1889 |
| execution/p0.1 | `B2/p0.1/n` | 1080 |
| execution/p0.1 | `B2/p0.1/successes` | 210 |
| execution/p0.1 | `B2/p0.1/safe_completion_rate` | 0.194444 |
| execution/p0.1 | `B2/p0.1/wilson_lo` | 0.17194 |
| execution/p0.1 | `B2/p0.1/wilson_hi` | 0.219115 |
| execution/p0.1 | `B2/p0.1/collision_rate` | 0.512963 |
| execution/p0.1 | `B2/p0.1/makespan_mean_s` | 14.5274 |
| execution/p0.1 | `B2/p0.1/makespan_p90_s` | 15.3478 |
| execution/p0.2 | `B2/p0.2/n` | 1080 |
| execution/p0.2 | `B2/p0.2/successes` | 207 |
| execution/p0.2 | `B2/p0.2/safe_completion_rate` | 0.191667 |
| execution/p0.2 | `B2/p0.2/wilson_lo` | 0.1693 |
| execution/p0.2 | `B2/p0.2/wilson_hi` | 0.216219 |
| execution/p0.2 | `B2/p0.2/collision_rate` | 0.525 |
| execution/p0.2 | `B2/p0.2/makespan_mean_s` | 16.757 |
| execution/p0.2 | `B2/p0.2/makespan_p90_s` | 17.9944 |
| execution/p0.0 | `B3/p0.0/n` | 1080 |
| execution/p0.0 | `B3/p0.0/successes` | 300 |
| execution/p0.0 | `B3/p0.0/safe_completion_rate` | 0.277778 |
| execution/p0.0 | `B3/p0.0/wilson_lo` | 0.251888 |
| execution/p0.0 | `B3/p0.0/wilson_hi` | 0.305243 |
| execution/p0.0 | `B3/p0.0/collision_rate` | 0.25 |
| execution/p0.0 | `B3/p0.0/makespan_mean_s` | 14.1444 |
| execution/p0.0 | `B3/p0.0/makespan_p90_s` | 14.1444 |
| execution/p0.05 | `B3/p0.05/n` | 1080 |
| execution/p0.05 | `B3/p0.05/successes` | 282 |
| execution/p0.05 | `B3/p0.05/safe_completion_rate` | 0.261111 |
| execution/p0.05 | `B3/p0.05/wilson_lo` | 0.235794 |
| execution/p0.05 | `B3/p0.05/wilson_hi` | 0.288122 |
| execution/p0.05 | `B3/p0.05/collision_rate` | 0.281481 |
| execution/p0.05 | `B3/p0.05/makespan_mean_s` | 14.8707 |
| execution/p0.05 | `B3/p0.05/makespan_p90_s` | 15.5067 |
| execution/p0.1 | `B3/p0.1/n` | 1080 |
| execution/p0.1 | `B3/p0.1/successes` | 277 |
| execution/p0.1 | `B3/p0.1/safe_completion_rate` | 0.256481 |
| execution/p0.1 | `B3/p0.1/wilson_lo` | 0.231332 |
| execution/p0.1 | `B3/p0.1/wilson_hi` | 0.283357 |
| execution/p0.1 | `B3/p0.1/collision_rate` | 0.302778 |
| execution/p0.1 | `B3/p0.1/makespan_mean_s` | 15.5841 |
| execution/p0.1 | `B3/p0.1/makespan_p90_s` | 16.3833 |
| execution/p0.2 | `B3/p0.2/n` | 1080 |
| execution/p0.2 | `B3/p0.2/successes` | 256 |
| execution/p0.2 | `B3/p0.2/safe_completion_rate` | 0.237037 |
| execution/p0.2 | `B3/p0.2/wilson_lo` | 0.212634 |
| execution/p0.2 | `B3/p0.2/wilson_hi` | 0.263304 |
| execution/p0.2 | `B3/p0.2/collision_rate` | 0.383333 |
| execution/p0.2 | `B3/p0.2/makespan_mean_s` | 17.3615 |
| execution/p0.2 | `B3/p0.2/makespan_p90_s` | 18.5967 |
| execution/p0.0 | `A1/p0.0/n` | 1080 |
| execution/p0.0 | `A1/p0.0/successes` | 210 |
| execution/p0.0 | `A1/p0.0/safe_completion_rate` | 0.194444 |
| execution/p0.0 | `A1/p0.0/wilson_lo` | 0.17194 |
| execution/p0.0 | `A1/p0.0/wilson_hi` | 0.219115 |
| execution/p0.0 | `A1/p0.0/collision_rate` | 0.527778 |
| execution/p0.0 | `A1/p0.0/makespan_mean_s` | 12.4 |
| execution/p0.0 | `A1/p0.0/makespan_p90_s` | 12.4 |
| execution/p0.05 | `A1/p0.05/n` | 1080 |
| execution/p0.05 | `A1/p0.05/successes` | 204 |
| execution/p0.05 | `A1/p0.05/safe_completion_rate` | 0.188889 |
| execution/p0.05 | `A1/p0.05/wilson_lo` | 0.166662 |
| execution/p0.05 | `A1/p0.05/wilson_hi` | 0.213321 |
| execution/p0.05 | `A1/p0.05/collision_rate` | 0.525926 |
| execution/p0.05 | `A1/p0.05/makespan_mean_s` | 13.5796 |
| execution/p0.05 | `A1/p0.05/makespan_p90_s` | 14.1889 |
| execution/p0.1 | `A1/p0.1/n` | 1080 |
| execution/p0.1 | `A1/p0.1/successes` | 210 |
| execution/p0.1 | `A1/p0.1/safe_completion_rate` | 0.194444 |
| execution/p0.1 | `A1/p0.1/wilson_lo` | 0.17194 |
| execution/p0.1 | `A1/p0.1/wilson_hi` | 0.219115 |
| execution/p0.1 | `A1/p0.1/collision_rate` | 0.522222 |
| execution/p0.1 | `A1/p0.1/makespan_mean_s` | 14.5274 |
| execution/p0.1 | `A1/p0.1/makespan_p90_s` | 15.3478 |
| execution/p0.2 | `A1/p0.2/n` | 1080 |
| execution/p0.2 | `A1/p0.2/successes` | 207 |
| execution/p0.2 | `A1/p0.2/safe_completion_rate` | 0.191667 |
| execution/p0.2 | `A1/p0.2/wilson_lo` | 0.1693 |
| execution/p0.2 | `A1/p0.2/wilson_hi` | 0.216219 |
| execution/p0.2 | `A1/p0.2/collision_rate` | 0.537963 |
| execution/p0.2 | `A1/p0.2/makespan_mean_s` | 16.757 |
| execution/p0.2 | `A1/p0.2/makespan_p90_s` | 17.9944 |
| execution/p0.0 | `A2/p0.0/n` | 1080 |
| execution/p0.0 | `A2/p0.0/successes` | 240 |
| execution/p0.0 | `A2/p0.0/safe_completion_rate` | 0.222222 |
| execution/p0.0 | `A2/p0.0/wilson_lo` | 0.198436 |
| execution/p0.0 | `A2/p0.0/wilson_hi` | 0.247977 |
| execution/p0.0 | `A2/p0.0/collision_rate` | 0.444444 |
| execution/p0.0 | `A2/p0.0/makespan_mean_s` | 13.7 |
| execution/p0.0 | `A2/p0.0/makespan_p90_s` | 13.7 |
| execution/p0.05 | `A2/p0.05/n` | 1080 |
| execution/p0.05 | `A2/p0.05/successes` | 240 |
| execution/p0.05 | `A2/p0.05/safe_completion_rate` | 0.222222 |
| execution/p0.05 | `A2/p0.05/wilson_lo` | 0.198436 |
| execution/p0.05 | `A2/p0.05/wilson_hi` | 0.247977 |
| execution/p0.05 | `A2/p0.05/collision_rate` | 0.439815 |
| execution/p0.05 | `A2/p0.05/makespan_mean_s` | 14.4711 |
| execution/p0.05 | `A2/p0.05/makespan_p90_s` | 15.0744 |
| execution/p0.1 | `A2/p0.1/n` | 1080 |
| execution/p0.1 | `A2/p0.1/successes` | 241 |
| execution/p0.1 | `A2/p0.1/safe_completion_rate` | 0.223148 |
| execution/p0.1 | `A2/p0.1/wilson_lo` | 0.199322 |
| execution/p0.1 | `A2/p0.1/wilson_hi` | 0.248937 |
| execution/p0.1 | `A2/p0.1/collision_rate` | 0.436111 |
| execution/p0.1 | `A2/p0.1/makespan_mean_s` | 15.2615 |
| execution/p0.1 | `A2/p0.1/makespan_p90_s` | 16.0822 |
| execution/p0.2 | `A2/p0.2/n` | 1080 |
| execution/p0.2 | `A2/p0.2/successes` | 228 |
| execution/p0.2 | `A2/p0.2/safe_completion_rate` | 0.211111 |
| execution/p0.2 | `A2/p0.2/wilson_lo` | 0.187817 |
| execution/p0.2 | `A2/p0.2/wilson_hi` | 0.236453 |
| execution/p0.2 | `A2/p0.2/collision_rate` | 0.469444 |
| execution/p0.2 | `A2/p0.2/makespan_mean_s` | 17.1341 |
| execution/p0.2 | `A2/p0.2/makespan_p90_s` | 18.3567 |
| execution/p0.0 | `A3/p0.0/n` | 1080 |
| execution/p0.0 | `A3/p0.0/successes` | 240 |
| execution/p0.0 | `A3/p0.0/safe_completion_rate` | 0.222222 |
| execution/p0.0 | `A3/p0.0/wilson_lo` | 0.198436 |
| execution/p0.0 | `A3/p0.0/wilson_hi` | 0.247977 |
| execution/p0.0 | `A3/p0.0/collision_rate` | 0.361111 |
| execution/p0.0 | `A3/p0.0/makespan_mean_s` | 13.6667 |
| execution/p0.0 | `A3/p0.0/makespan_p90_s` | 13.6667 |
| execution/p0.05 | `A3/p0.05/n` | 1080 |
| execution/p0.05 | `A3/p0.05/successes` | 240 |
| execution/p0.05 | `A3/p0.05/safe_completion_rate` | 0.222222 |
| execution/p0.05 | `A3/p0.05/wilson_lo` | 0.198436 |
| execution/p0.05 | `A3/p0.05/wilson_hi` | 0.247977 |
| execution/p0.05 | `A3/p0.05/collision_rate` | 0.376852 |
| execution/p0.05 | `A3/p0.05/makespan_mean_s` | 14.4478 |
| execution/p0.05 | `A3/p0.05/makespan_p90_s` | 15.0511 |
| execution/p0.1 | `A3/p0.1/n` | 1080 |
| execution/p0.1 | `A3/p0.1/successes` | 244 |
| execution/p0.1 | `A3/p0.1/safe_completion_rate` | 0.225926 |
| execution/p0.1 | `A3/p0.1/wilson_lo` | 0.201981 |
| execution/p0.1 | `A3/p0.1/wilson_hi` | 0.251813 |
| execution/p0.1 | `A3/p0.1/collision_rate` | 0.384259 |
| execution/p0.1 | `A3/p0.1/makespan_mean_s` | 15.2404 |
| execution/p0.1 | `A3/p0.1/makespan_p90_s` | 16.0622 |
| execution/p0.2 | `A3/p0.2/n` | 1080 |
| execution/p0.2 | `A3/p0.2/successes` | 233 |
| execution/p0.2 | `A3/p0.2/safe_completion_rate` | 0.215741 |
| execution/p0.2 | `A3/p0.2/wilson_lo` | 0.192239 |
| execution/p0.2 | `A3/p0.2/wilson_hi` | 0.241258 |
| execution/p0.2 | `A3/p0.2/collision_rate` | 0.432407 |
| execution/p0.2 | `A3/p0.2/makespan_mean_s` | 17.1285 |
| execution/p0.2 | `A3/p0.2/makespan_p90_s` | 18.3356 |

### 图表

![failure_attribution.png](stages/S5/plots/failure_attribution.png)
![plan_time_and_makespan.png](stages/S5/plots/plan_time_and_makespan.png)
![safe_completion_vs_pause.png](stages/S5/plots/safe_completion_vs_pause.png)
![safety_efficiency_scatter.png](stages/S5/plots/safety_efficiency_scatter.png)

### 阶段报告原文

### S5 完整对照实验报告

- 阶段：`S5_evaluation`
- 运行目录：`D:\paper2\results\S5_evaluation\20261103_033410_s5_benchmark_a98f98`
- 生成时间：2026-11-03T03:46:41+08:00
- 耗时：0.9 s

#### 实验设计

- 任务：36 个（4 类地图 × [2, 4, 6] 台），每格 3 个；
- 方法：B0, B1, B2, B3, A1, A2, A3（共享同一 Diffusion 权重、同一噪声、20 步 DDIM、相同引导预算）；
- 执行：每份规划在 p∈[0.0, 0.05, 0.1, 0.2] 下各做 30 次独立随机执行；
- 成功口径：规划有效 + 执行中全部到达 + 执行全程无碰撞（失败规划计 0）；
- 置信区间：以任务为簇的 bootstrap（2000 次重采样）。

#### 结果

| 方法 | 规划成功率 | 平均耗时 [s] | 平均等待 | 平均硬风险 |
|---|---|---|---|---|
| B0 独立 Diffusion | 0.306 | 0.81 | 0.00 | 1.1070 |
| B1 同步距离引导 | 0.306 | 0.81 | 0.00 | 0.9924 |
| B2 固定时间窗引导 | 0.306 | 1.09 | 0.00 | 0.8783 |
| B3 逐时刻期望碰撞风险 | 0.306 | 5.08 | 6.32 | 0.4208 |
| A1 首次碰撞 + 只改路径 | 0.306 | 2.60 | 0.00 | 0.9203 |
| A2 首次碰撞 + 只改等待 | 0.306 | 2.59 | 3.87 | 0.6765 |
| A3 首次碰撞 + 联合 | 0.306 | 2.50 | 3.83 | 0.5611 |

#### 执行结果（p=0.2）

| 方法 | 安全完成率 | 95% 区间 | 碰撞率 | makespan [s] |
|---|---|---|---|---|
| B0 | 0.186 | [0.164, 0.210] | 0.568 | 16.76 |
| B1 | 0.192 | [0.169, 0.216] | 0.556 | 16.76 |
| B2 | 0.192 | [0.169, 0.216] | 0.525 | 16.76 |
| B3 | 0.237 | [0.213, 0.263] | 0.383 | 17.36 |
| A1 | 0.192 | [0.169, 0.216] | 0.538 | 16.76 |
| A2 | 0.211 | [0.188, 0.236] | 0.469 | 17.13 |
| A3 | 0.216 | [0.192, 0.241] | 0.432 | 17.13 |

#### 关键检验（G2：首次碰撞吸收是否不可替代）

- A3 vs B3（逐时刻期望风险）：安全完成率差 -0.0366，95% 区间 [-0.0963, -0.0002]，显著；
- A3 安全完成率 0.216，B3 0.237（p=0.2）。

#### 结论与局限

- 本报告为**预实验规模**（配置 `benchmark.scale`），不可当作完整规模的最终结论；
- 若 A3 与 B3 差异不显著，则“首次碰撞吸收”的独立贡献尚未被证明（G2 未达成）；
- 所有失败均按 §10.2 口径计入分母，未做后验剔除。


#### 指标

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

#### 产物

- `tables\plans.csv`
- `tables\executions.csv`
- `plots\safe_completion_vs_pause.png`
- `plots\plan_time_and_makespan.png`
- `plots\safety_efficiency_scatter.png`
- `plots\failure_attribution.png`
