# S3 单机器人条件 Diffusion 训练报告

- 阶段：`S3_diffusion_training`
- 运行目录：`D:\paper2\results\S3_diffusion_training\20261103_124832_s3_train_d6b02d`
- 生成时间：2026-11-03T12:57:51+08:00
- 耗时：0.1 s

## 训练设置

- 去噪网络：条件 1D U-Net（通道 [64, 128, 256]，FiLM 条件注入），参数约 2.22 M；
- 调度：cosine，T=1000，预测量 ε；
- 优化器 AdamW，lr=0.0002，batch=96，梯度累积 1，EMA decay=0.999；
- 训练样本 4819 条，验证样本 473 条。

## 说明

- 训练过程中的 loss/梯度范数/吞吐逐步写入 `train_log.csv`（断点续训会追加）；
- 每 `sample_every` 步用 EMA 权重做一次 DDIM 采样并出图，便于观察是否在学习；
- 检查点：`latest.pt`（可续训）、`best.pt`（验证损失最低）、`step_XXXXXX.pt`。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T12:57:50+08:00 |  |
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
| train | `steps_completed` | 30000 |  |
| train | `best_val_loss` | 0.00962619 |  |
| train | `train_seconds` | 557.7 |  |
| train | `batch_size` | 96 |  |
| train | `grad_accum` | 1 |  |
| data | `n_train_samples` | 4819 |  |
| data | `n_val_samples` | 473 |  |
| data | `train/tasks_attempted` | 2604 |  |
| data | `train/tasks_reachable` | 2500 |  |
| data | `train/vectors_generated` | 4819 |  |
| data | `train/discarded_unreachable` | 341 |  |
| data | `train/discarded_too_close` | 0 |  |
| data | `train/discarded_speed` | 48 |  |
| data | `train/discarded_static` | 0 |  |
| data | `train/discarded_length` | 0 |  |
| data | `train/diversity_mean_m` | 0.144219 |  |
| data | `train/n_maps_used` | 50 |  |
| data | `train/n_paths` | 4819 |  |
| data | `train/elapsed_s` | 947.9 |  |
| data | `val/tasks_attempted` | 261 |  |
| data | `val/tasks_reachable` | 250 |  |
| data | `val/vectors_generated` | 473 |  |
| data | `val/discarded_unreachable` | 33 |  |
| data | `val/discarded_too_close` | 0 |  |
| data | `val/discarded_speed` | 16 |  |
| data | `val/discarded_static` | 0 |  |
| data | `val/discarded_length` | 0 |  |
| data | `val/diversity_mean_m` | 0.150499 |  |
| data | `val/n_maps_used` | 10 |  |
| data | `val/n_paths` | 473 |  |
| data | `val/elapsed_s` | 89.8 |  |
| data | `test/tasks_attempted` | 256 |  |
| data | `test/tasks_reachable` | 250 |  |
| data | `test/vectors_generated` | 489 |  |
| data | `test/discarded_unreachable` | 20 |  |
| data | `test/discarded_too_close` | 0 |  |
| data | `test/discarded_speed` | 3 |  |
| data | `test/discarded_static` | 0 |  |
| data | `test/discarded_length` | 0 |  |
| data | `test/diversity_mean_m` | 0.116782 |  |
| data | `test/n_maps_used` | 10 |  |
| data | `test/n_paths` | 489 |  |
| data | `test/elapsed_s` | 81.4 |  |

## 产物

