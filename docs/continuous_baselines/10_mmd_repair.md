# 10 MMD 修复报告

**结果**：`results/Continuous_Baseline_Comparison/repair_v2/mmd/native_validation/`

## 修复内容

- 不再使用 32 点弧长重采样轨迹；
- 直接使用 MMD 原生 `[3,64,4]` 轨迹；
- 时间参数改用 MMD 默认 `trajectory_duration=5.0`、`horizon=64`，
  `dt=5/64=0.078125 s`；
- 低层用双积分器跟踪，速度/加速度限制 `1.5/2.0`，加入位置反馈；
- 不使用 No02 First-Hit/W2+；
- 保留原生速度字段的说明：本执行用原始 64 点位置差分作为参考速度，
  未把未经验证的导出速度直接当物理速度。

## 原生执行结果

```text
n_agents=3
duration=5.0 s
dt_mmd=0.078125
collision_at_0.10=False
collision_at_0.35=True
all_arrived=True
min_pair_distance=0.1326 m
goal_error_max=0.0132 m
max_speed=1.023 m/s
```

## 解释

- MMD 原始半径约 0.05 m，接触距离 0.10 m；轨迹最小间距 0.1326 m，
  因此在**原生尺度**下无接触碰撞；
- 但 0.1326 m < No02 的 0.35 m 安全距离，所以按 No02 口径必然不安全；
- 这不是 MMD 规划失败，而是两者安全距离/机器人尺度不同。

## 未完成

- 未在 No02 相同地图/起终点上运行 MMD；
- 未验证 MMD 导出速度字段与物理速度的对应关系；
- 未做共同场景；因此不能与 Ours 做 Experiment B 对比。

## 状态

**ADAPTED（原生场景执行可用），共同场景 BLOCKED**。
