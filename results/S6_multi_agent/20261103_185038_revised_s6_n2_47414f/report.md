# S6 多机器人联合实验（N=2）报告

- 阶段：`S6_multi_agent`
- 运行目录：`D:\paper2\results\S6_multi_agent\20261103_185038_revised_s6_n2_47414f`
- 生成时间：2026-11-03T18:52:37+08:00
- 耗时：0.1 s

## 结果

| 方法 | 规划有效率 | 硬风险 | 安全完成率 | 全到达率 | makespan [时隙] | 等待总量 | 耗时 [s] |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.750 | 0.7569 | 0.252 | 1.000 | 35.6 | 0.00 | 0.30 |
| B1 | 0.833 | 0.7208 | 0.282 | 1.000 | 35.6 | 0.00 | 2.13 |
| B2 | 0.750 | 0.7546 | 0.253 | 1.000 | 35.6 | 0.00 | 4.73 |
| B3 | 0.750 | 0.4772 | 0.523 | 1.000 | 37.5 | 2.58 | 7.22 |

## 配对比较（安全完成率）

- B3_vs_B1: +0.2417 [+0.0854, +0.4117] 显著
- B3_vs_B2: +0.2700 [+0.0946, +0.4588] 显著
- B3_vs_B0: +0.2712 [+0.0958, +0.4588] 显著
- B2_vs_B0: +0.0012 [+0.0000, +0.0037] 不显著
- B1_vs_B0: +0.0296 [-0.0038, +0.0887] 不显著


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T18:52:37+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | a75050cbf265616cecb769b79034e64f66a9e1bd |  |
| environment | `git_dirty` | True |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| setup | `agents` | 2 |  |
| setup | `pause_p` | 0.1 |  |
| method | `B0/plan_valid_rate` | 0.75 |  |
| method | `B0/hard_risk_mean` | 0.756855 |  |
| method | `B0/safe_completion_mean` | 0.252083 |  |
| method | `B0/all_arrived_rate` | 1 |  |
| method | `B0/makespan_steps` | 35.5767 |  |
| method | `B0/wait_total_mean` | 0 |  |
| method | `B0/elapsed_s_mean` | 0.30316 |  |
| method | `B1/plan_valid_rate` | 0.833333 |  |
| method | `B1/hard_risk_mean` | 0.720767 |  |
| method | `B1/safe_completion_mean` | 0.281667 |  |
| method | `B1/all_arrived_rate` | 1 |  |
| method | `B1/makespan_steps` | 35.5767 |  |
| method | `B1/wait_total_mean` | 0 |  |
| method | `B1/elapsed_s_mean` | 2.12617 |  |
| method | `B2/plan_valid_rate` | 0.75 |  |
| method | `B2/hard_risk_mean` | 0.754618 |  |
| method | `B2/safe_completion_mean` | 0.253333 |  |
| method | `B2/all_arrived_rate` | 1 |  |
| method | `B2/makespan_steps` | 35.5767 |  |
| method | `B2/wait_total_mean` | 0 |  |
| method | `B2/elapsed_s_mean` | 4.73409 |  |
| method | `B3/plan_valid_rate` | 0.75 |  |
| method | `B3/hard_risk_mean` | 0.477242 |  |
| method | `B3/safe_completion_mean` | 0.523333 |  |
| method | `B3/all_arrived_rate` | 1 |  |
| method | `B3/makespan_steps` | 37.5175 |  |
| method | `B3/wait_total_mean` | 2.57892 |  |
| method | `B3/elapsed_s_mean` | 7.2192 |  |
| paired | `B3_vs_B1/diff` | 0.241667 |  |
| paired | `B3_vs_B1/significant` | True |  |
| paired | `B3_vs_B2/diff` | 0.27 |  |
| paired | `B3_vs_B2/significant` | True |  |
| paired | `B3_vs_B0/diff` | 0.27125 |  |
| paired | `B3_vs_B0/significant` | True |  |
| paired | `B2_vs_B0/diff` | 0.00125 |  |
| paired | `B2_vs_B0/significant` | False |  |
| paired | `B1_vs_B0/diff` | 0.0295833 |  |
| paired | `B1_vs_B0/significant` | False |  |

## 产物

- `tables\s6_per_task.csv`
