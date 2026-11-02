# 项目进度

最后更新：2026-11-03T03:46:41+08:00

## 阶段状态

| 阶段 | 状态 | 完成时间 | 耗时 | Git commit | 指标数 |
|---|---|---|---|---|---|
| S0_environment | passed | 2026-11-03T02:03:14+08:00 | 3.6 s | unknown | 88 |
| S1_risk_dp | passed | 2026-11-03T02:05:26+08:00 | 13.4 s | 0ef64b83 | 36 |
| S2_gradient_optimization | passed | 2026-11-03T02:15:57+08:00 | 157.7 s | 2e7ab8e4 | 89 |
| S3_diffusion_training | completed | 2026-11-03T03:20:51+08:00 | 2.6 s | 41d9a81a | 29 |
| S4_guided_diffusion | passed | 2026-11-03T03:21:32+08:00 | 1.5 s | 41d9a81a | 46 |
| S5_evaluation | completed | 2026-11-03T03:46:41+08:00 | 0.9 s | 41d9a81a | 275 |

## 关键结论

- **S5**：36 个任务 × 7 方法，A3 安全完成率 0.216。

## 说明

- 每个阶段的完整记录（配置快照、CSV 原始数据、PNG 图、报告）位于 `docs/stages/Sx/`，已纳入 Git。
- 完整运行目录（含全部轨迹与检查点）保留在 `results/Sx/`，不入库。
- 阶段状态由各次运行的 `stage_status.json` 自动汇总，不手工编辑。
