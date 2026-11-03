# F2 地图条件架构 A/B 对照报告

- 阶段：`F2_architecture_ab`
- 运行目录：`D:\paper2\results\F2_architecture_ab\20261103_125758_f2_ab_348295`
- 生成时间：2026-11-03T12:58:11+08:00
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
| 旧（index） | 20 | 0.575 | [0.394, 0.750] | 0.575 | 1.019 |
| 新（grid_sample） | 20 | 0.631 | [0.450, 0.800] | 0.631 | 1.028 |

| 地图类型 | 旧架构 | 新架构 | 差值 |
|---|---:|---:|---:|
| 空旷交叉 | 0.950 | 0.975 | +0.025 |
| 双通路绕障 | 0.375 | 0.625 | +0.250 |
| 窄通道/门口 | 0.525 | 0.525 | +0.000 |
| 多交叉/瓶颈 | 0.450 | 0.400 | -0.050 |

## 地图敏感度探针（机制是否真的接通）

同一批轨迹与时间步，只替换输入地图，测量 ε 预测的 RMS 变化：

| 模型 | 基线 eps RMS | 平移地图 | 全空地图 | 换另一张图 |
|---|---:|---:|---:|---:|
| 旧（index） | 1.8070 | 0.0103 | 0.1363 | 0.0431 |
| 新（grid_sample） | 1.9288 | 0.0358 | 0.1961 | 0.0653 |

若「平移 / 全空 / 换图」三列都接近 0，说明模型几乎不看地图；此时新增的空间查询即使接线正确，也无法在 8000 步内体现出成功率差异。

## 配对比较

- 新 − 旧 = **+0.0563**，任务级配对 bootstrap 95% CI [-0.0500, +0.1562]，**不显著**
- 任务数 20；8,000 步只是流程验证规模，**不能当作最终结论**


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T12:58:11+08:00 |  |
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
| summary | `old_index/success_rate` | 0.575 |  |
| summary | `old_index/ci_lo` | 0.39375 |  |
| summary | `old_index/ci_hi` | 0.75 |  |
| summary | `old_index/static_safe_rate` | 0.575 |  |
| summary | `old_index/length_ratio` | 1.01875 |  |
| by_kind | `old_index/open/success_rate` | 0.95 |  |
| by_kind | `old_index/dual_corridor/success_rate` | 0.375 |  |
| by_kind | `old_index/narrow_gate/success_rate` | 0.525 |  |
| by_kind | `old_index/bottleneck/success_rate` | 0.45 |  |
| summary | `new_grid/success_rate` | 0.63125 |  |
| summary | `new_grid/ci_lo` | 0.45 |  |
| summary | `new_grid/ci_hi` | 0.8 |  |
| summary | `new_grid/static_safe_rate` | 0.63125 |  |
| summary | `new_grid/length_ratio` | 1.02753 |  |
| by_kind | `new_grid/open/success_rate` | 0.975 |  |
| by_kind | `new_grid/dual_corridor/success_rate` | 0.625 |  |
| by_kind | `new_grid/narrow_gate/success_rate` | 0.525 |  |
| by_kind | `new_grid/bottleneck/success_rate` | 0.4 |  |
| paired | `paired_diff_new_minus_old` | 0.05625 |  |
| paired | `paired_diff_lo` | -0.05 |  |
| paired | `paired_diff_hi` | 0.15625 |  |
| paired | `paired_significant` | False |  |

## 产物

- `tables\f2_per_task.csv`
