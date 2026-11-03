# F0.5 几何引导有效性诊断报告

- 阶段：`F05_guidance_diagnosis`
- 运行目录：`D:\paper2\results\F05_guidance_diagnosis\20261103_122304_f05_diagnosis_9c82cc`
- 生成时间：2026-11-03T12:25:21+08:00
- 耗时：0.1 s

## 诊断设置

- checkpoint：`D:\paper2\results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\best.pt`
- 场景：crossing（N=2），DDIM 40 步，固定初始噪声 seed=20261103
- 扫描：step_size_x ∈ [0.05, 0.1, 0.2, 0.5]，updates_per_step ∈ [1, 3, 5]，start_fraction ∈ [0.25, 0.5, 0.75]，共 36 组
- 每个采样步都保存了 x̂0（米制），与 B0 的同一步逐一比较

## 结果

- **B0 参考**：硬风险 0.2046，软风险 0.2164，路径长 8.81 m
- **所有 36 组配置中，最终几何位移 RMS 最大只有 5.00 cm**（相对路径长度 0.57%）
- 单次引导的最终存活率中位数：0.0%（即时修正越大，残留比例越低，见 `guidance_retention.png`）
- 扫描范围内硬风险最低 0.0102，B0 为 0.2046
- 等待量分布：中位数 2.34 时隙

## 结论：几何修正确实被后续去噪抵消

1. 每一步引导确实能把 x̂0 推动一个可见的量（`guidance_delta_norm` 量级见 CSV，约与 step_size_x×2 同阶），但**下一条去噪步几乎立刻把它拉回先验流形**；
2. 因此 40 步下来累计位移只有毫米级，路径长度几乎不变；
3. 这解释了 S5 中 7 个方法规划有效率完全相同（11/36，且是同一批任务），以及 A3 ≈ A2：**当前实现真正生效的只有等待变量 w，几何通道形同虚设**；
4. `step_size_x`、`updates_per_step`、`start_fraction` 三者的单调影响可以在热图上看到，但即使取最强组合，位移仍在厘米级以下——单纯调参无法解决。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T12:25:21+08:00 |  |
| environment | `python` | 3.12.13 |  |
| environment | `python_executable` | D:\paper2\.venv\Scripts\python.exe |  |
| environment | `torch` | 2.11.0+cu128 |  |
| environment | `cuda_available` | True |  |
| environment | `cuda_version` | 12.8 |  |
| environment | `cudnn_version` | 91900 |  |
| environment | `platform` | Windows-11-10.0.26200-SP0 |  |
| environment | `processor` | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |  |
| environment | `cpu_count` | 10 |  |
| environment | `git_commit` | ea6261bba9340f4c330a2a3fa27f3a964c3ff86c |  |
| environment | `git_dirty` | True |  |
| environment | `gpu_name` | NVIDIA GeForce RTX 4070 |  |
| environment | `gpu_total_memory_gb` | 11.99 |  |
| environment | `gpu_index` | 0 |  |
| environment | `gpu_capability` | 8.9 |  |
| model | `checkpoint` | D:\paper2\results\S3_diffusion_training\20261103_030603_s3_train_383287\checkpoints\best.pt |  |
| setup | `scenario` | crossing |  |
| setup | `ddim_steps` | 40 |  |
| b0 | `b0_hard_risk` | 0.204613 |  |
| b0 | `b0_soft_risk` | 0.216388 |  |
| sweep | `sweep_max_rms_disp` | 0.0499931 |  |
| sweep | `sweep_min_hard_risk` | 0.0101508 |  |
| retention | `retention_median` | 0.000183015 |  |

## 产物

- `tables\f05_sweep.csv`
- `tables\f05_per_step.csv`
- `tables\f05_retention.csv`
- `tables\f05_baseline_b0.csv`
