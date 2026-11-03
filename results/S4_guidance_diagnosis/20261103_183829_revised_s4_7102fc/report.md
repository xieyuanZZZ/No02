# S4 几何风险引导诊断（修正版）报告

- 阶段：`S4_guidance_diagnosis`
- 运行目录：`D:\paper2\results\S4_guidance_diagnosis\20261103_183829_revised_s4_7102fc`
- 生成时间：2026-11-03T18:38:34+08:00
- 耗时：0.2 s

## 修正了第一轮 F0.5 的什么

- 记录四段轨迹：**修改前 / 修改后 / 下一步 / 最终**，不再只比较首尾；
- 全部位移统一为**路点 RMS [m]**，不混用张量范数与逐点平均；
- 增加 DDIM 公式的解析传递系数，与实测对照。

## 公式分析

DDIM 一步为 `x_{t+1} = sqrt(ᾱ_{t+1})·x̂0 + sqrt(1−ᾱ_{t+1})·ε_ref`，其中 `ε_ref = (x_t − sqrt(ᾱ_t)·x̂0)/sqrt(1−ᾱ_t)`。对 x̂0 求导得

```
k_t = sqrt(ᾱ_{t+1}) − sqrt(1−ᾱ_{t+1})·sqrt(ᾱ_t)/sqrt(1−ᾱ_t)
```

- 本调度下 k_t 的中位数 **0.0558**，高噪声段接近 0，低噪声段才升到 0.2–0.3；
- 若之后不再引导，累积传递约为 `k_t^(剩余步数)`，因此单次修正在几十步后必然衰减到 0；
- 这与实测的「下一步残留 / 即时修正」量级一致（见 `s4_transfer_coefficients.csv`）。

## 实测结果

- B0 硬风险 0.6724；B1（每步引导）硬风险 0.4171
- B1 相对 B0 的最终位移 RMS：**0.0071 m**
- 单次引导即时修正最大 0.0050 m，最终残留最大 0.0001 m
- 单次引导存活率中位数 0.27%

## 结论

1. **不是实现写错，而是 DDIM 公式本身的性质**：改 x̂0 的修正只有 k_t 比例进入下一步，且后续每步继续衰减；
2. 因此「每步都改一点」在 40 步里只能留下厘米级以内的痕迹；
3. 这正是 S5 改从**初始噪声**入手的理由：z 是整条链的输入，不经过 k_t 衰减。


## 指标

| 分组 | 指标 | 值 | 备注 |
|---|---|---|---|
| environment | `timestamp` | 2026-11-03T18:38:34+08:00 |  |
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
| setup | `ddim_steps` | 40 |  |
| setup | `step_size` | 0.2 |  |
| final | `b0_hard_risk` | 0.67242 |  |
| final | `b1_hard_risk` | 0.417105 |  |
| setup | `task_id` | s5_002 |  |
| final | `b1_final_disp_rms_m` | 0.00708665 |  |
| single | `max_immediate_rms_m` | 0.00500001 |  |
| single | `max_final_residual_rms_m` | 8.39027e-05 |  |

## 产物

- `tables\s4_transfer_coefficients.csv`
- `tables\s4_single_shot.csv`
- `tables\s4_cumulative.csv`
