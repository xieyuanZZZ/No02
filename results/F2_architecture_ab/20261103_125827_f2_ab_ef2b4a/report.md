# F2 地图条件架构 A/B 对照报告

- 阶段：`F2_architecture_ab`
- 运行目录：`D:\paper2\results\F2_architecture_ab\20261103_125827_f2_ab_ef2b4a`
- 生成时间：2026-11-03T12:59:20+08:00
- 耗时：0.1 s

## 对照设置

- 训练数据：`data/dataset`（medium：4,819 条训练轨迹 / 50 张地图）
- 两模型唯一差异：`diffusion.map_conditioning`（interface `index` vs `grid_sample`）
- 其余完全相同：网络宽度、`map_spatial_res=16`、batch 96、8,000 步、随机种子 20261103
- 评估：100 个固定任务（每类地图 5 个），每任务 8 条候选，40 步 DDIM，两模型共用同一任务与同一噪声种子
- 成功率 = 静态连续几何安全 **且** 每段 ≤0.6 m

## 结果

| 模型 | 任务数 | 成功率 | 任务级 95% CI | 静态安全率 | 长度比 |
|---|---:|---:|---|---:|---:|
| 旧（index） | 100 | 0.526 | [0.445, 0.613] | 0.530 | 1.010 |
| 新（grid_sample） | 100 | 0.593 | [0.515, 0.677] | 0.596 | 1.006 |

| 地图类型 | 旧架构 | 新架构 | 差值 |
|---|---:|---:|---:|
| 空旷交叉 | 0.925 | 0.940 | +0.015 |
| 双通路绕障 | 0.410 | 0.625 | +0.215 |
| 窄通道/门口 | 0.325 | 0.350 | +0.025 |
| 多交叉/瓶颈 | 0.445 | 0.455 | +0.010 |

## 地图敏感度探针（机制是否真的接通）

同一批轨迹与时间步，只替换输入地图，测量 ε 预测的 RMS 变化：

| 模型 | 基线 eps RMS | 平移地图 | 全空地图 | 换另一张图 |
|---|---:|---:|---:|---:|
| 旧（index） | 1.8001 | 0.0041 | 0.0611 | 0.0000 |
| 新（grid_sample） | 1.9331 | 0.0399 | 0.0421 | 0.0000 |

若「平移 / 全空 / 换图」三列都接近 0，说明模型几乎不看地图；此时新增的空间查询即使接线正确，也无法在 8000 步内体现出成功率差异。

## 配对比较

- 新 − 旧 = **+0.0663**，任务级配对 bootstrap 95% CI [+0.0337, +0.1037]，显著
- 任务数 100；8,000 步只是流程验证规模，**不能当作最终结论**


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T12:59:20+08:00 |  |
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
| summary | `old_index/success_rate` | 0.52625 |  |
| summary | `old_index/ci_lo` | 0.445 |  |
| summary | `old_index/ci_hi` | 0.6125 |  |
| summary | `old_index/static_safe_rate` | 0.53 |  |
| summary | `old_index/length_ratio` | 1.01047 |  |
| by_kind | `old_index/open/success_rate` | 0.925 |  |
| by_kind | `old_index/dual_corridor/success_rate` | 0.41 |  |
| by_kind | `old_index/narrow_gate/success_rate` | 0.325 |  |
| by_kind | `old_index/bottleneck/success_rate` | 0.445 |  |
| summary | `new_grid/success_rate` | 0.5925 |  |
| summary | `new_grid/ci_lo` | 0.515 |  |
| summary | `new_grid/ci_hi` | 0.6775 |  |
| summary | `new_grid/static_safe_rate` | 0.59625 |  |
| summary | `new_grid/length_ratio` | 1.00647 |  |
| by_kind | `new_grid/open/success_rate` | 0.94 |  |
| by_kind | `new_grid/dual_corridor/success_rate` | 0.625 |  |
| by_kind | `new_grid/narrow_gate/success_rate` | 0.35 |  |
| by_kind | `new_grid/bottleneck/success_rate` | 0.455 |  |
| paired | `paired_diff_new_minus_old` | 0.06625 |  |
| paired | `paired_diff_lo` | 0.0337188 |  |
| paired | `paired_diff_hi` | 0.10375 |  |
| paired | `paired_significant` | True |  |

## 产物

- `tables\f2_per_task.csv`
