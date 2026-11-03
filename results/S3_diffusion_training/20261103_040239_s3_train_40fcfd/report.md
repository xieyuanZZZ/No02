# S3 单机器人条件 Diffusion 训练报告

- 阶段：`S3_diffusion_training`
- 运行目录：`D:\paper2\results\S3_diffusion_training\20261103_040239_s3_train_40fcfd`
- 生成时间：2026-11-03T04:17:52+08:00
- 耗时：0.1 s

## 训练设置

- 去噪网络：条件 1D U-Net（通道 [64, 128, 256]，FiLM 条件注入），参数约 2.22 M；
- 调度：cosine，T=1000，预测量 ε；
- 优化器 AdamW，lr=0.0002，batch=96，梯度累积 1，EMA decay=0.999；
- 训练样本 19212 条，验证样本 950 条。

## 说明

- 训练过程中的 loss/梯度范数/吞吐逐步写入 `train_log.csv`（断点续训会追加）；
- 每 `sample_every` 步用 EMA 权重做一次 DDIM 采样并出图，便于观察是否在学习；
- 检查点：`latest.pt`（可续训）、`best.pt`（验证损失最低）、`step_XXXXXX.pt`。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T04:17:52+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | bf45effe42fe9dade204a2c91d280f9321f6f7db |  |
| environment | `git_dirty` | False |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| train | `steps_completed` | 50000 |  |
| train | `best_val_loss` | 0.00874877 |  |
| train | `train_seconds` | 911.9 |  |
| train | `batch_size` | 96 |  |
| train | `grad_accum` | 1 |  |
| data | `n_train_samples` | 19212 |  |
| data | `n_val_samples` | 950 |  |
| data | `train/tasks_attempted` | 10411 |  |
| data | `train/tasks_reachable` | 10000 |  |
| data | `train/vectors_generated` | 19212 |  |
| data | `train/discarded_unreachable` | 1443 |  |
| data | `train/discarded_too_close` | 0 |  |
| data | `train/discarded_speed` | 167 |  |
| data | `train/discarded_static` | 0 |  |
| data | `train/discarded_length` | 0 |  |
| data | `train/diversity_mean_m` | 0.148461 |  |
| data | `train/n_maps_used` | 100 |  |
| data | `train/n_paths` | 19212 |  |
| data | `train/elapsed_s` | 3799.6 |  |
| data | `val/tasks_attempted` | 519 |  |
| data | `val/tasks_reachable` | 500 |  |
| data | `val/vectors_generated` | 950 |  |
| data | `val/discarded_unreachable` | 70 |  |
| data | `val/discarded_too_close` | 0 |  |
| data | `val/discarded_speed` | 18 |  |
| data | `val/discarded_static` | 0 |  |
| data | `val/discarded_length` | 0 |  |
| data | `val/diversity_mean_m` | 0.136086 |  |
| data | `val/n_maps_used` | 20 |  |
| data | `val/n_paths` | 950 |  |
| data | `val/elapsed_s` | 174.2 |  |
| data | `test/tasks_attempted` | 514 |  |
| data | `test/tasks_reachable` | 500 |  |
| data | `test/vectors_generated` | 958 |  |
| data | `test/discarded_unreachable` | 61 |  |
| data | `test/discarded_too_close` | 0 |  |
| data | `test/discarded_speed` | 9 |  |
| data | `test/discarded_static` | 0 |  |
| data | `test/discarded_length` | 0 |  |
| data | `test/diversity_mean_m` | 0.119684 |  |
| data | `test/n_maps_used` | 20 |  |
| data | `test/n_paths` | 958 |  |
| data | `test/elapsed_s` | 171 |  |

## 产物

