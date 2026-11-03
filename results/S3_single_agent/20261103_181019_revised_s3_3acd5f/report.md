# S3 单机器人 Diffusion 训练报告

- 阶段：`S3_single_agent`
- 运行目录：`D:\paper2\results\S3_single_agent\20261103_181019_revised_s3_3acd5f`
- 生成时间：2026-11-03T18:36:49+08:00
- 耗时：0.1 s

## 训练

- 配置：`configs/s2_B_128_32.yaml`（地图输入 128，特征 32，条件 grid_sample）
- 数据：`data/revised_s1`（S1 三类简单场景）
- 步数：**27000**（CLI 覆盖值已写回配置快照）
- best val loss：0.00648；训练用时 0 s
- 曲线：`plots/s3_train_curves.png`（train/val loss + 梯度范数）

## 评估（按地图类型分别统计）

| 地图类型 | 任务数 | 单条有效率 | Task Success@8 | 静态安全率 | 速度通过率 | 长度比 | 净空 [m] | 推理 [s/条] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 开放交叉 | 25 | 0.960 | 0.960 | 0.960 | 1.000 | 1.001 | inf | 0.025 |
| 单障碍绕行 | 25 | 0.295 | 0.760 | 0.295 | 1.000 | 0.990 | -0.380 | 0.023 |
| 单瓶颈通行 | 25 | 0.995 | 1.000 | 0.995 | 1.000 | 1.001 | 0.069 | 0.023 |

## 验收判定

- 简单场景（开放交叉 + 单障碍绕行）是否同时达到 95% 有效率：**否**
- 总体单条有效率 0.750（任务级 95% CI [0.670, 0.830]）
- 是否满足进入 S4 的条件：**否，需先解决基础模型问题**

瓶颈场景单独统计，不用它掩盖基础模型是否具备正常规划能力（方案 §5.3）。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T18:36:49+08:00 |  |
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
| train | `train_steps` | 27000 |  |
| train | `best_val` | 0.0064839 |  |
| train | `train_seconds` | 0 |  |
| model | `map_input_res` | 128 |  |
| model | `map_spatial_res` | 32 |  |
| eval | `success_rate` | 0.75 |  |
| eval | `success_ci_lo` | 0.67 |  |
| eval | `success_ci_hi` | 0.83 |  |
| eval | `task_success_at_k` | 0.906667 |  |
| by_kind | `open_crossing/success_rate` | 0.96 |  |
| by_kind | `open_crossing/task_success` | 0.96 |  |
| by_kind | `single_obstacle/success_rate` | 0.295 |  |
| by_kind | `single_obstacle/task_success` | 0.76 |  |
| by_kind | `bottleneck_channel/success_rate` | 0.995 |  |
| by_kind | `bottleneck_channel/task_success` | 1 |  |

## 产物

- `tables\s3_per_task.csv`
- `tables\s3_train_log.csv`
