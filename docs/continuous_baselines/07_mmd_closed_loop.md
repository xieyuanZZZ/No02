# 07 MMD 原生规划执行状态

**日期**：2026-11-06  
**结果**：`results/Continuous_Baseline_Comparison/pilot/mmd/`

## 已做

- 使用官方 MMD 推理结果（3 agents，Empty 地图，`mmd_paths.json`）；
- 保留原始 64 点位置/速度/时间信息，没有只用 32 点弧长重采样；
- 用双积分器跟踪 MMD 原始时空轨迹：`dt=0.1`、`|v|<=1.5`、`|a|<=2.0`；
- 注入 0.1 概率、0.4 s 暂停事件；
- 未加入 First-Hit 或 No02 在线等待。

## 实际结果

```text
n_agents=3
collision=True
safe_completion=False
all_arrived=False
min_pair_distance=0.032 m
duration=3.0 s
max_speed=0.568 m/s
```

## 解释

这次只是**官方 MMD 示例轨迹的动力学执行验证**，不是 No02 地图上的实验 B：

- 官方 MMD 使用自己的 2×2 Empty 地图和起终点，No02 是固定 32 航点任务；
- 当前跟踪器是简单双积分跟踪，不是 MMD 原生的完整执行器；
- 3 s 固定时长假设来自 MMD 参数（官方 trial 没有直接给出执行时长字段），
  这一点需要从 MMD 配置中进一步核实；
- 因此出现碰撞/未到达不能归因于 MMD 规划失败，也不能作为论文对比数字。

## 未完成

- 在 No02 相同地图/起终点上运行 MMD 官方模型；
- 使用 MMD 原生的时间/速度执行器；
- 与 Ours 的 Experiment B 对照。

**MMD 暂标记为“官方规划可运行，No02 共同场景未完成”，不参与正式对比。**
