# S2 不使用 Diffusion 的风险梯度优化报告

- 阶段：`S2_gradient_optimization`
- 运行目录：`D:\paper2\results\S2_gradient_optimization\20261103_020636_s2_optimize_84e8c4`
- 生成时间：2026-11-03T02:07:37+08:00
- 耗时：60.8 s

## 阶段目标

在不使用任何神经网络的前提下，验证“可微首次碰撞风险 → 梯度 → 轨迹与等待的有效修正”这条链条成立（文档 §9 S2）。

## 优化设置

- 损失：λ_R ΣR_soft + λ_T J_finish + λ_S L_static + λ_K L_kinematic + λ_G L_geometry + λ_W ||w||₁；
- λ 取值为批复 §23 的默认值（λ_R=1.0, λ_T=0.05, λ_S=10.0, λ_K=1.0, λ_G=0.5, λ_W=0.01）；
- 等待投影：0≤w≤4.0 时隙/点，Σw≤8.0 时隙/机器人；
- 端点固定、端点梯度屏蔽；步数 30，Adam 路径 lr=0.02、等待 lr=0.1；
- 每 500 次独立 Monte Carlo 复现用于验证真实碰撞率。

## 各场景结果

- **crossing**：硬风险 0.1788 → 只改路径 0.0018 / 只改等待 0.0080 / 联合 0.0020（降低 +0.1768）
- **dual_corridor**：硬风险 1.0000 → 只改路径 1.0000 / 只改等待 1.0000 / 联合 1.0000（降低 +0.0000）
- **narrow_gate**：硬风险 1.0000 → 只改路径 1.0000 / 只改等待 1.0000 / 联合 1.0000（降低 +0.0000）

## 结论

- 1/3 个几何案例在充分时间预算下把硬风险显著降低（>0.01）；
- 软风险与硬风险、Monte Carlo 碰撞率同向下降，说明可微代理的优化方向可信；
- 等待只出现在安全位置（见 `wait_safety_*.csv`），未出现“无限等待”作弊：等待总量受总预算约束，且 J_finish 随等待单调上升；
- 通过条件：不满足（要求至少 2 个案例）。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T02:06:36+08:00 |  |
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
| baseline | `crossing/baseline/mc_collision_rate` | 0.166 |  |
| baseline | `crossing/baseline/mc_makespan_s` | 14.1664 |  |
| baseline | `crossing/baseline/mc_completion_rate` | 1 |  |
| wait_safety | `crossing/A1 只改路径/dangerous_wait_points` | 0 |  |
| result | `crossing/A1 只改路径/hard_risk` | 0.00183935 |  |
| result | `crossing/A1 只改路径/mc_collision_rate` | 0.002 |  |
| result | `crossing/A1 只改路径/wait_total` | 0 |  |
| result | `crossing/A1 只改路径/finish_cost_s` | 27.5556 |  |
| wait_safety | `crossing/A2 只改等待/dangerous_wait_points` | 0 |  |
| result | `crossing/A2 只改等待/hard_risk` | 0.00799324 |  |
| result | `crossing/A2 只改等待/mc_collision_rate` | 0.0065 |  |
| result | `crossing/A2 只改等待/wait_total` | 3.27946 |  |
| result | `crossing/A2 只改等待/finish_cost_s` | 28.6017 |  |
| wait_safety | `crossing/A3 联合优化/dangerous_wait_points` | 0 |  |
| result | `crossing/A3 联合优化/hard_risk` | 0.00201651 |  |
| result | `crossing/A3 联合优化/mc_collision_rate` | 0.003 |  |
| result | `crossing/A3 联合优化/wait_total` | 0 |  |
| result | `crossing/A3 联合优化/finish_cost_s` | 27.5556 |  |
| baseline | `dual_corridor/baseline/hard_risk_sum` | 1 |  |
| baseline | `dual_corridor/baseline/hard_risk_max` | 1 |  |
| baseline | `dual_corridor/baseline/soft_risk_sum` | 1 |  |
| baseline | `dual_corridor/baseline/finish_cost_s` | 27.5556 |  |
| baseline | `dual_corridor/baseline/unfinished_prob` | -2.22045e-16 |  |
| baseline | `dual_corridor/baseline/wait_total` | 0 |  |
| baseline | `dual_corridor/baseline/mc_collision_rate` | 1 |  |
| baseline | `dual_corridor/baseline/mc_makespan_s` | 14.1664 |  |
| baseline | `dual_corridor/baseline/mc_completion_rate` | 1 |  |
| wait_safety | `dual_corridor/A1 只改路径/dangerous_wait_points` | 0 |  |
| result | `dual_corridor/A1 只改路径/hard_risk` | 1 |  |
| result | `dual_corridor/A1 只改路径/mc_collision_rate` | 1 |  |
| result | `dual_corridor/A1 只改路径/wait_total` | 0 |  |
| result | `dual_corridor/A1 只改路径/finish_cost_s` | 27.5556 |  |
| wait_safety | `dual_corridor/A2 只改等待/dangerous_wait_points` | 0 |  |
| result | `dual_corridor/A2 只改等待/hard_risk` | 1 |  |
| result | `dual_corridor/A2 只改等待/mc_collision_rate` | 1 |  |
| result | `dual_corridor/A2 只改等待/wait_total` | 0 |  |
| result | `dual_corridor/A2 只改等待/finish_cost_s` | 27.5556 |  |
| wait_safety | `dual_corridor/A3 联合优化/dangerous_wait_points` | 0 |  |
| result | `dual_corridor/A3 联合优化/hard_risk` | 1 |  |
| result | `dual_corridor/A3 联合优化/mc_collision_rate` | 1 |  |
| result | `dual_corridor/A3 联合优化/wait_total` | 0 |  |
| result | `dual_corridor/A3 联合优化/finish_cost_s` | 27.5556 |  |
| baseline | `narrow_gate/baseline/hard_risk_sum` | 1 |  |
| baseline | `narrow_gate/baseline/hard_risk_max` | 1 |  |
| baseline | `narrow_gate/baseline/soft_risk_sum` | 1 |  |
| baseline | `narrow_gate/baseline/finish_cost_s` | 27.5556 |  |
| baseline | `narrow_gate/baseline/unfinished_prob` | -2.22045e-16 |  |
| baseline | `narrow_gate/baseline/wait_total` | 0 |  |
| baseline | `narrow_gate/baseline/mc_collision_rate` | 1 |  |
| baseline | `narrow_gate/baseline/mc_makespan_s` | 14.1664 |  |
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
| verdict | `s2_improved_cases` | 1 |  |

## 产物

- `tables\wait_safety_crossing_path.csv`
- `tables\wait_safety_crossing_wait.csv`
- `tables\wait_safety_crossing_joint.csv`
- `plots\crossing_joint_paths.png`
- `plots\crossing_progress_distribution.png`
- `plots\crossing_convergence.png`
- `tables\wait_safety_dual_corridor_path.csv`
- `tables\wait_safety_dual_corridor_wait.csv`
- `tables\wait_safety_dual_corridor_joint.csv`
- `plots\dual_corridor_joint_paths.png`
- `plots\dual_corridor_progress_distribution.png`
- `plots\dual_corridor_convergence.png`
- `tables\wait_safety_narrow_gate_path.csv`
- `tables\wait_safety_narrow_gate_wait.csv`
- `tables\wait_safety_narrow_gate_joint.csv`
- `plots\narrow_gate_joint_paths.png`
- `plots\narrow_gate_progress_distribution.png`
- `plots\narrow_gate_convergence.png`
- `tables\optimization_traces.csv`
- `tables\pareto_points.csv`
- `tables\pareto_lambda_T.csv`
- `plots\pareto_crossing.png`
