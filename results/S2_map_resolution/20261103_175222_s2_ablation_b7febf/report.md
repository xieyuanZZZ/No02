# S2 地图分辨率消融报告

- 阶段：`S2_map_resolution`
- 运行目录：`D:\paper2\results\S2_map_resolution\20261103_175222_s2_ablation_b7febf`
- 生成时间：2026-11-03T18:09:41+08:00
- 耗时：0.2 s

## 设置

- 数据：`data/revised_s1`（S1 重建的三类简单场景，按地图划分 train/val/test）
- 每组地图输入 `in_res` 从**连续几何重新栅格化**（不是把 64 插值成 128）
- 网络结构固定为「两个全分辨率卷积 + n 次 stride-2 下采样」，通道预算相同
- 每组训练 8000 步，同种子、同 batch；评估用同一批任务与同一噪声
- 成功率 = 静态连续几何安全 **且** 每段 ≤0.6 m

## 总体结果

| 编号 | 地图输入 | CNN 特征 | 训练步数 | 有效率 | 95% CI | TaskSuccess@8 | 静态安全率 | 长度比 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| B | 128 | 32 | 8000 | 0.675 | [0.577, 0.773] | 0.933 | 0.675 | 0.997 |
| A | 64 | 16 | 8000 | 0.673 | [0.571, 0.769] | 0.933 | 0.673 | 0.995 |
| C | 128 | 64 | 8000 | 0.658 | [0.556, 0.754] | 0.833 | 0.665 | 0.996 |

## 按地图类型

| 编号 | 开放交叉 | 单障碍绕行 | 单瓶颈通行 |
|---|---:|---:|---:|
| B | 0.938 | 0.181 | 0.906 |
| A | 0.925 | 0.169 | 0.925 |
| C | 0.881 | 0.194 | 0.900 |

## 结论

- 最优配置：**B**（128→32），有效率 0.675
- 这是 8000 步的快速筛选结果，**不作为最终结论**；选定架构后由 S3 用完整预算重训


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T18:09:41+08:00 |  |
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
| ablation | `A/in_res` | 64 |  |
| ablation | `A/feat_res` | 16 |  |
| ablation | `A/success_rate` | 0.672917 |  |
| ablation | `A/success_ci_lo` | 0.570833 |  |
| ablation | `A/success_ci_hi` | 0.768802 |  |
| ablation | `A/task_success_at_8` | 0.933333 |  |
| ablation | `A/static_safe_rate` | 0.672917 |  |
| ablation | `A/best_val` | 0.00757765 |  |
| by_kind | `A/open_crossing/success_rate` | 0.925 |  |
| by_kind | `A/single_obstacle/success_rate` | 0.16875 |  |
| by_kind | `A/bottleneck_channel/success_rate` | 0.925 |  |
| ablation | `B/in_res` | 128 |  |
| ablation | `B/feat_res` | 32 |  |
| ablation | `B/success_rate` | 0.675 |  |
| ablation | `B/success_ci_lo` | 0.577083 |  |
| ablation | `B/success_ci_hi` | 0.772917 |  |
| ablation | `B/task_success_at_8` | 0.933333 |  |
| ablation | `B/static_safe_rate` | 0.675 |  |
| ablation | `B/best_val` | 0.00752258 |  |
| by_kind | `B/open_crossing/success_rate` | 0.9375 |  |
| by_kind | `B/single_obstacle/success_rate` | 0.18125 |  |
| by_kind | `B/bottleneck_channel/success_rate` | 0.90625 |  |
| ablation | `C/in_res` | 128 |  |
| ablation | `C/feat_res` | 64 |  |
| ablation | `C/success_rate` | 0.658333 |  |
| ablation | `C/success_ci_lo` | 0.55625 |  |
| ablation | `C/success_ci_hi` | 0.754167 |  |
| ablation | `C/task_success_at_8` | 0.833333 |  |
| ablation | `C/static_safe_rate` | 0.664583 |  |
| ablation | `C/best_val` | 0.00842765 |  |
| by_kind | `C/open_crossing/success_rate` | 0.88125 |  |
| by_kind | `C/single_obstacle/success_rate` | 0.19375 |  |
| by_kind | `C/bottleneck_channel/success_rate` | 0.9 |  |

## 产物

- `tables\s2_per_task.csv`
