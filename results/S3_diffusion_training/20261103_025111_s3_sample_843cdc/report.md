# S3 单机器人生成质量评估报告

- 阶段：`S3_diffusion_training`
- 运行目录：`D:\paper2\results\S3_diffusion_training\20261103_025111_s3_sample_843cdc`
- 生成时间：2026-11-03T02:51:36+08:00
- 耗时：2.4 s

## 评估口径

- 从 `train` 划分中均匀取 120 个任务，每个任务采样 6 条候选；
- “成功”定义：静态障碍连续几何无碰撞 **且** 每段长度 ≤ v_max·Δt=0.60 m；
- 起终点由采样器强制固定，误差应接近 0；
- 多样性 = 同一任务 k 条候选的平均两两点距。

## 结果

- 成功率 **66.39%**；静态碰撞率 33.61%；速度越界率 0.42%；
- 路径长度比（生成/真实）均值 0.994；
- 候选多样性 0.235 m；
- 推理 30.7 ms/样本（DDIM 40 步）。

**不追求 100% 成功**：不合格候选一律记为失败，不做后验剔除。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T02:51:34+08:00 |  |
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
| sampling | `n_tasks` | 120 |  |
| sampling | `samples_per_task` | 6 |  |
| sampling | `ddim_steps` | 40 |  |
| sampling | `success_rate` | 0.663889 |  |
| sampling | `static_collision_rate` | 0.336111 |  |
| sampling | `speed_violation_rate` | 0.00416667 |  |
| sampling | `start_err_max_m` | 7.88052e-07 |  |
| sampling | `goal_err_max_m` | 8.86426e-07 |  |
| sampling | `length_ratio_mean` | 0.99419 |  |
| sampling | `smoothness_mean` | 0.00153293 |  |
| sampling | `diversity_mean_m` | 0.234601 |  |
| sampling | `inference_ms_per_sample` | 30.6898 |  |
| sampling | `total_seconds` | 22.0966 |  |

## 产物

- `tables\single_robot_samples.csv`
- `plots\generated_samples_grid.png`
- `plots\success_rate_hist.png`
