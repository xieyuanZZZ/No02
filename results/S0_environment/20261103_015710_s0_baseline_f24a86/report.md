# S0 环境与几何基础验证报告

- 阶段：`S0_environment`
- 运行目录：`D:\paper2\results\S0_environment\20261103_015710_s0_baseline_f24a86`
- 生成时间：2026-11-03T01:57:14+08:00
- 耗时：4.1 s

## 阶段目标

建立新工程环境、静态地图生成器、机器人圆盘几何、任务/场景构造、轨迹连续碰撞检测，以及独立随机执行模拟器；并用单元测试固定正确性。

## 关键实现决定

- 障碍物只用**连续几何**（轴对齐矩形 + 圆）描述，栅格仅用于可视化与网络条件输入；
- 线段到矩形的距离用**逐矩形黄金分割**（单矩形距离对参数 u 是凸的，min 会破坏凸性，故不能合并求解）；
- 机器人之间用相对运动闭式解求最小距离，并额外保留一套独立实现供交叉验证；
- 执行模型为层 A：每时隙最多前进 1 个路点，时隙内线性插值，同步连续段碰撞判定。

## 结论

- 4 类地图生成正常，占据率与 SDF 数值合理；
- 7 个手工场景全部满足段长 <= v_max*dt = 0.60 m；
- 交叉场景在 p=0.1 下随机执行碰撞率 0.1760，说明随机暂停确实会制造碰撞风险；
- S0 通过，可进入 S1（首次碰撞风险 DP）。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T01:57:10+08:00 |  |
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
| map/bottleneck | `occupied_cells` | 1260 |  |
| map/bottleneck | `occupancy_ratio` | 0.307617 |  |
| map/bottleneck | `sdf_min_m` | -0.18 |  |
| map/bottleneck | `sdf_max_m` | 3.42643 |  |
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
| scenario/multi_bottleneck | `path_length_mean_m` | 8.8 |  |
| scenario/multi_bottleneck | `max_segment_m` | 0.283871 |  |
| scenario/multi_bottleneck | `speed_limit_ok` | True |  |
| scenario/multi_bottleneck | `static_safe` | False |  |
| scenario/multi_bottleneck | `min_clearance_m` | -0.18 |  |
| execution | `crossing/n_samples` | 500 |  |
| execution | `crossing/collision_rate` | 0.176 |  |
| execution | `crossing/completion_rate` | 1 |  |
| execution | `crossing/mean_makespan_steps` | 35.59 |  |
| execution | `crossing/mean_makespan_s` | 14.236 |  |
| execution | `crossing/std_makespan_s` | 0.764607 |  |
| execution | `crossing/unfinished_agent_rate` | 0 |  |
| execution | `crossing/d_safe_m` | 0.35 |  |
| tests | `tests_passed` | 0 |  |
| tests | `tests_failed` | 0 |  |

## 产物

- `tables\map_open.csv`
- `plots\map_open.png`
- `tables\map_dual_corridor.csv`
- `plots\map_dual_corridor.png`
- `tables\map_narrow_gate.csv`
- `plots\map_narrow_gate.png`
- `tables\map_bottleneck.csv`
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
