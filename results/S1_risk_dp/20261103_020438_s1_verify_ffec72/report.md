# S1 首次碰撞风险 DP 正确性报告

- 阶段：`S1_risk_dp`
- 运行目录：`D:\paper2\results\S1_risk_dp\20261103_020438_s1_verify_ffec72`
- 生成时间：2026-11-03T02:04:58+08:00
- 耗时：19.7 s

## 阶段目标

在训练任何神经网络之前，先证明“随机执行进度传播 + 首次碰撞吸收”这套动态规划在数学上正确、可微、且与独立 Monte Carlo 一致（文档 §5.4 的第一道门槛）。

## 方法要点

- 状态：F_t(m,n) 为“截至 t 未碰撞且联合进度为 (m,n)”的概率质量，共 H×H=32×32 个状态；
- 每步枚举 4 种动作（00/10/01/11），用四种动作的连续运动段碰撞核吸收质量；
- 连续碰撞判定：d_min 由相对运动闭式解给出（d_safe=0.35 m）；
- 软门槛 a=(1-p)·σ((t+0.5+c-τ)/β)，c=0.5、β=0.25（批复 §11）；
- 硬门槛 a=(1-p)·1[t≥τ]，与执行模拟器使用完全相同的实数比较；
- 全队语义：两两风险之和只是引导目标，不是全队至少碰撞一次的概率；全队碰撞率必须由联合 Monte Carlo 估计（本节给出并集上界对照）。

## 测试与对照结果

- R1–R10 单元测试：通过 35 项、失败 0 项、错误 0 项；
- R7（20 个案例 × 10000 次 MC）：最大绝对误差 0.00674，二项检验不一致案例 0 个（α=0.001）；
- R9：硬风险恒为 0.178811（与 ε、β 无关），logistic 软代理在 [0.1675, 0.3408] 之间变化，说明软代理不具备概率解释；
- 软/硬门槛带来的进度分布最大总变差距离 0.0584；
- R8：梯度与中心差分的最大相对误差 3.53e-07。

## 结论

- 首次碰撞概率 DP 在无碰撞时严格保持概率守恒，在必然碰撞时严格为 1，且只计第一次碰撞；
- 硬 DP 与独立 Monte Carlo 在统计误差内一致（最大绝对误差 0.00674）；
- 硬风险可精确计算但不可微；软风险可微但只是代理量，二者必须分开报告；
- S1 门槛通过，可进入 S2（不使用 Diffusion 的风险梯度优化）。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T02:04:38+08:00 |  |
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
| tests | `tests_passed` | 35 |  |
| tests | `tests_failed` | 0 |  |
| tests | `tests_error` | 0 |  |
| R7 | `r7_max_abs_error` | 0.00673775 |  |
| R7 | `r7_inconsistent_cases` | 0 |  |
| R7 | `r7_total_mc_samples` | 200000 |  |
| R7 | `r7_elapsed_s` | 1.31437 |  |
| R9 | `r9_hard_reference` | 0.178811 |  |
| R9 | `r9_soft_max` | 0.340815 |  |
| R9 | `r9_soft_min` | 0.167456 |  |
| R9 | `soft_hard_max_tv_distance` | 0.0583864 |  |
| survival | `crossing/hard_risk` | 0.178811 |  |
| survival | `head_on/hard_risk` | 1 |  |
| R8 | `gradient_max_rel_error` | 3.52945e-07 |  |
| timing | `timing_n2_ms` | 57.73 |  |
| timing | `timing_n4_ms` | 87.8 |  |
| timing | `timing_n6_ms` | 144.99 |  |
| timing | `timing_cuda_n6_ms` | 334.77 |  |
| timing | `cuda_peak_mem_mb` | 77.1 |  |
| multi | `n6_pair_risk_sum` | 12.0834 |  |
| multi | `n6_union_bound` | 1 |  |
| multi | `n6_mc_team_collision_rate` | 1 |  |

## 产物

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
