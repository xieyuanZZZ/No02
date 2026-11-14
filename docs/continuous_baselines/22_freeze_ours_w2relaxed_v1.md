# 22 冻结：Ours-W2plus-relaxed-v1

**配置**：`results/Continuous_Baseline_Comparison/credible_v1/frozen_v1/frozen_config.json`

## 冻结内容

- 方法名：`Ours-W2plus-relaxed-v1`
- 来源提交：`a807ec1`
- W2+：
  - `require_p0=False`
  - `p0_weight=1.0`
  - `initial_stagger=True`
  - `max_evals=800, restarts=3, lam_T=0.05`
- First-Hit：`horizon_s=3.2, hold_slots=1, d_safe=0.35`
- 物理：`dt=0.1, dt_slot=0.4, vmax=1.5, amax=2.0`
- 任务：A1/A2/B1/B2，manifest SHA256
  `CE0A5191DABB70483BF325092769D80B08D60BC5850C0AFE5472CA92E74B1B9E`
- 数据：180 条，`per_run.jsonl` SHA256
  `BAB7509DFA21BC33380BE948BA23C4791547D7A15A98251188941CB71C29F46F`

## 说明

这不是原冻结 W2+ 默认行为；默认 `require_p0=True` 未变。  
本冻结只用于新实验配置，旧 Tag `7019d54` 的冻结结果不受影响。

## 冻结数据

- p=0/0.1/0.2 各 5 seeds；
- 4 个 A/B 任务；
- 3 方法 × 180 条；
- 原始 NPZ 轨迹已保存。

后续大规模测试必须引用本配置文件与提交，不能重新调 W2+ 参数。
