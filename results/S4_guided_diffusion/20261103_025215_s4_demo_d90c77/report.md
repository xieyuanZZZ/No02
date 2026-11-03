# S4 风险引导 DDIM 演示与验证报告

- 阶段：`S4_guided_diffusion`
- 运行目录：`D:\paper2\results\S4_guided_diffusion\20261103_025215_s4_demo_d90c77`
- 生成时间：2026-11-03T02:52:23+08:00
- 耗时：1.6 s

## 阶段目标

把首次碰撞风险模块接到**推理阶段**的每一步 DDIM 预测上：网络参数冻结，风险梯度只修正当前的干净轨迹预测 Q̂0 与等待变量 w（文档 §7.3）。

## 验证结果

| 检查项 | 结果 |
|---|---|
| 零引导等价 | 通过 |
| 参数冻结 | 通过 |
| 端点保持 | 通过 |
| 静态安全 | 通过 |
| 速度约束 | 通过 |

- 零引导与基础 DDIM 最大差异 `1.788e-07`；
- 推理后发生变化的参数张量数 `0`；
- B0（独立采样）：硬风险 0.0776，MC 碰撞率 0.0740；
- A3（引导）：硬风险 0.0783，MC 碰撞率 0.0745，主动等待 0.00 时隙；
- 规划耗时：B0 1.5s，A3 2.8s（40 步 DDIM）。

## 结论

- 引导通路在结构上正确（零引导严格等价、参数冻结、端点不变）；
- 本次引导未降低风险，需要检查步长与权重；
- 与 S2 的手工案例一致：风险梯度既可以修改几何，也可以产生主动等待。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T02:52:21+08:00 |  |
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
| verification | `zero_guidance_max_diff` | 1.78814e-07 |  |
| verification | `zero_guidance_exact` | True |  |
| verification | `frozen_params_changed` | 0 |  |
| model | `checkpoint_step` | 26500 |  |
| result | `B0/hard_risk` | 0.0776404 |  |
| result | `B0/static_safe` | True |  |
| result | `B0/min_clearance_m` | inf |  |
| result | `B0/speed_ok` | True |  |
| result | `B0/max_segment_m` | 0.335036 |  |
| result | `B0/mc_collision_rate` | 0.074 |  |
| result | `B0/mc_makespan_s` | 14.1912 |  |
| result | `B0/endpoint_err` | 0 |  |
| result | `B0/endpoint_err_goal` | 0 |  |
| result | `A3/hard_risk` | 0.0783379 |  |
| result | `A3/static_safe` | True |  |
| result | `A3/min_clearance_m` | inf |  |
| result | `A3/speed_ok` | True |  |
| result | `A3/max_segment_m` | 0.335036 |  |
| result | `A3/mc_collision_rate` | 0.0745 |  |
| result | `A3/mc_makespan_s` | 14.1912 |  |
| result | `A3/endpoint_err` | 0 |  |
| result | `A3/endpoint_err_goal` | 0 |  |
| result | `A3/wait_total` | 0 |  |
| result | `A3/plan_time_s` | 2.83967 |  |
| result | `B0/plan_time_s` | 1.46944 |  |
| verification | `check/零引导等价` | True |  |
| verification | `check/参数冻结` | True |  |
| verification | `check/端点保持` | True |  |
| verification | `check/静态安全` | True |  |
| verification | `check/速度约束` | True |  |

## 产物

- `tables\a3_guidance_trace.csv`
- `plots\guidance_trajectory.png`
- `plots\plan_B0.png`
- `plots\plan_A3.png`
- `plots\geometry_evolution.png`
