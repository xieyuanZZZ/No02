# S3 单机器人生成质量评估报告

- 阶段：`S3_diffusion_training`
- 运行目录：`D:\paper2\results\S3_diffusion_training\20261103_025919_s3_sample_452838`
- 生成时间：2026-11-03T02:59:53+08:00
- 耗时：2.9 s

## 评估口径

- 从 `test` 划分中均匀取 150 个任务，每个任务采样 6 条候选；
- “成功”定义：静态障碍连续几何无碰撞 **且** 每段长度 ≤ v_max·Δt=0.60 m；
- 起终点由采样器强制固定，误差应接近 0；
- 多样性 = 同一任务 k 条候选的平均两两点距。

## 结果

- 成功率 **48.67%**；静态碰撞率 50.89%；速度越界率 3.33%；
- 路径长度比（生成/真实）均值 1.030；
- 候选多样性 0.616 m；
- 推理 34.5 ms/样本（DDIM 40 步）。

**不追求 100% 成功**：不合格候选一律记为失败，不做后验剔除。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T02:59:50+08:00 |  |
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
| sampling | `n_tasks` | 150 |  |
| sampling | `samples_per_task` | 6 |  |
| sampling | `ddim_steps` | 40 |  |
| sampling | `success_rate` | 0.486667 |  |
| sampling | `static_collision_rate` | 0.508889 |  |
| sampling | `speed_violation_rate` | 0.0333333 |  |
| sampling | `start_err_max_m` | 6.82329e-07 |  |
| sampling | `goal_err_max_m` | 9.30859e-07 |  |
| sampling | `length_ratio_mean` | 1.0299 |  |
| sampling | `smoothness_mean` | 0.00363653 |  |
| sampling | `diversity_mean_m` | 0.616266 |  |
| sampling | `inference_ms_per_sample` | 34.4854 |  |
| sampling | `total_seconds` | 31.0369 |  |

## 产物

- `tables\single_robot_samples.csv`
- `plots\generated_samples_grid.png`
- `plots\success_rate_hist.png`
