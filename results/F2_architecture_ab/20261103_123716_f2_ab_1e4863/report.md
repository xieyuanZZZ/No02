# F2 地图条件架构 A/B 对照报告

- 阶段：`F2_architecture_ab`
- 运行目录：`D:\paper2\results\F2_architecture_ab\20261103_123716_f2_ab_1e4863`
- 生成时间：2026-11-03T12:37:30+08:00
- 耗时：0.1 s

## 对照设置

- 训练数据：`data/dataset`（medium：4,819 条训练轨迹 / 50 张地图）
- 两模型唯一差异：`diffusion.map_conditioning`（interface `index` vs `grid_sample`）
- 其余完全相同：网络宽度、`map_spatial_res=16`、batch 96、8,000 步、随机种子 20261103
- 评估：20 个固定任务（每类地图 5 个），每任务 8 条候选，40 步 DDIM，两模型共用同一任务与同一噪声种子
- 成功率 = 静态连续几何安全 **且** 每段 ≤0.6 m

## 结果

| 模型 | 任务数 | 成功率 | 任务级 95% CI | 静态安全率 | 长度比 |
|---|---:|---:|---|---:|---:|
| 旧（index） | 20 | 0.537 | [0.350, 0.713] | 0.544 | 1.027 |
| 新（grid_sample） | 20 | 0.506 | [0.319, 0.694] | 0.512 | 1.044 |

| 地图类型 | 旧架构 | 新架构 | 差值 |
|---|---:|---:|---:|
| 空旷交叉 | 0.925 | 0.875 | -0.050 |
| 双通路绕障 | 0.275 | 0.300 | +0.025 |
| 窄通道/门口 | 0.600 | 0.575 | -0.025 |
| 多交叉/瓶颈 | 0.350 | 0.275 | -0.075 |

## 配对比较

- 新 − 旧 = **-0.0312**，任务级配对 bootstrap 95% CI [-0.0625, +0.0000]，**不显著**
- 任务数 20；8,000 步只是流程验证规模，**不能当作最终结论**


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T12:37:30+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | 1abeef873790ce10ddce85b48bdb2382be3c0aae |  |
| environment | `git_dirty` | True |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| summary | `old_index/success_rate` | 0.5375 |  |
| summary | `old_index/ci_lo` | 0.35 |  |
| summary | `old_index/ci_hi` | 0.712656 |  |
| summary | `old_index/static_safe_rate` | 0.54375 |  |
| summary | `old_index/length_ratio` | 1.02748 |  |
| by_kind | `old_index/open/success_rate` | 0.925 |  |
| by_kind | `old_index/dual_corridor/success_rate` | 0.275 |  |
| by_kind | `old_index/narrow_gate/success_rate` | 0.6 |  |
| by_kind | `old_index/bottleneck/success_rate` | 0.35 |  |
| summary | `new_grid/success_rate` | 0.50625 |  |
| summary | `new_grid/ci_lo` | 0.31875 |  |
| summary | `new_grid/ci_hi` | 0.69375 |  |
| summary | `new_grid/static_safe_rate` | 0.5125 |  |
| summary | `new_grid/length_ratio` | 1.04448 |  |
| by_kind | `new_grid/open/success_rate` | 0.875 |  |
| by_kind | `new_grid/dual_corridor/success_rate` | 0.3 |  |
| by_kind | `new_grid/narrow_gate/success_rate` | 0.575 |  |
| by_kind | `new_grid/bottleneck/success_rate` | 0.275 |  |
| paired | `paired_diff_new_minus_old` | -0.03125 |  |
| paired | `paired_diff_lo` | -0.0625 |  |
| paired | `paired_diff_hi` | 0 |  |
| paired | `paired_significant` | False |  |

## 产物

- `tables\f2_per_task.csv`
