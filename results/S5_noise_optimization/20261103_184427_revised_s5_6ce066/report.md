# S5 初始噪声优化报告

- 阶段：`S5_noise_optimization`
- 运行目录：`D:\paper2\results\S5_noise_optimization\20261103_184427_revised_s5_6ce066`
- 生成时间：2026-11-03T18:47:34+08:00
- 耗时：0.1 s

## A. 噪声敏感性

- 每个任务 20 个初始噪声，轨迹两两 RMS 平均 **0.078 m**，最大 0.849 m
- 硬风险跨噪声的标准差平均 0.0643
- 结论：模型确实利用初始噪声产生不同轨迹，可以进入噪声优化

## B. 方法对照

| 方法 | 硬风险均值 | 95% CI | 静态安全率 | 最小机器人间距 | 噪声偏移 | 耗时 [s] |
|---|---:|---|---:|---:|---:|---:|
| B0 | 0.7830 | [0.6729, 0.8695] | 0.85 | 0.162 | 0.000 | 0.3 |
| B1 | 0.7226 | [0.6041, 0.8285] | 0.90 | 0.203 | 0.000 | 2.2 |
| B2 | 0.7817 | [0.6717, 0.8683] | 0.85 | 0.165 | 0.010 | 4.7 |
| R | 0.7820 | [0.6707, 0.8687] | 0.85 | 0.167 | 0.100 | 0.3 |

| 对照 | 硬风险差 | 95% CI | 显著 |
|---|---:|---|---|
| B2_vs_B0 | -0.0013 | [-0.0036, +0.0004] | 否 |
| B2_vs_B1 | +0.0591 | [+0.0126, +0.1192] | 是 |
| B2_vs_R | -0.0002 | [-0.0028, +0.0025] | 否 |

## 判定

- B2 相对 B0：-0.0013（不显著）
- B2 相对 B1：+0.0591（显著）
- B2 相对随机扰动 R：-0.0002（不显著）

只有同时优于 B1 与 R，才能认为「初始噪声优化」这个机制本身有效；否则收益可能只是「换了条轨迹」或「随机搜索」。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T18:47:34+08:00 |  |
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
| setup | `ckpt` | D:\paper2\results\S3_single_agent\20261103_181019_revised_s3_3acd5f\checkpoints\best.pt |  |
| sensitivity | `noise_sensitivity_mean_rms_m` | 0.0777842 |  |
| method | `B0/hard_risk` | 0.782983 |  |
| method | `B0/static_safe_rate` | 0.85 |  |
| method | `B0/noise_rms` | 0 |  |
| method | `B0/elapsed_s` | 0.31445 |  |
| method | `B1/hard_risk` | 0.722589 |  |
| method | `B1/static_safe_rate` | 0.9 |  |
| method | `B1/noise_rms` | 0 |  |
| method | `B1/elapsed_s` | 2.15445 |  |
| method | `B2/hard_risk` | 0.781727 |  |
| method | `B2/static_safe_rate` | 0.85 |  |
| method | `B2/noise_rms` | 0.00998622 |  |
| method | `B2/elapsed_s` | 4.67922 |  |
| method | `R/hard_risk` | 0.78197 |  |
| method | `R/static_safe_rate` | 0.85 |  |
| method | `R/noise_rms` | 0.1 |  |
| method | `R/elapsed_s` | 0.314886 |  |
| paired | `B2_vs_B0/diff` | -0.00125512 |  |
| paired | `B2_vs_B0/lo` | -0.00361462 |  |
| paired | `B2_vs_B0/hi` | 0.00038893 |  |
| paired | `B2_vs_B0/significant` | False |  |
| paired | `B2_vs_B1/diff` | 0.0591383 |  |
| paired | `B2_vs_B1/lo` | 0.0126023 |  |
| paired | `B2_vs_B1/hi` | 0.119226 |  |
| paired | `B2_vs_B1/significant` | True |  |
| paired | `B2_vs_R/diff` | -0.000242531 |  |
| paired | `B2_vs_R/lo` | -0.00275214 |  |
| paired | `B2_vs_R/hi` | 0.00246781 |  |
| paired | `B2_vs_R/significant` | False |  |

## 产物

- `tables\s5_noise_sensitivity.csv`
- `tables\s5_methods.csv`
