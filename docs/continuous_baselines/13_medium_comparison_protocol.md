# 13 中等规模对比协议

**日期**：2026-11-06  
**结果**：`results/Continuous_Baseline_Comparison/medium_v1/`

## 实际执行范围

- N=4，4 个任务（`open_crossing_300000_n4_0/1`、`open_crossing_300003_n4_0/1`）；
- p=0（1 个确定性 seed）、p=0.1（3 seeds）、p=0.2（3 seeds）；
- 方法：Ours（冻结 W2+＋First-Hit）、ORCA/RVO2、MPC-CBF；
- 共 4 × (1+3+3) × 3 = **84 条执行记录**；
- 每条记录实时追加到 `per_run.jsonl`，支持跳过已完成项；
- 失败也写入记录，不删除样本；
- 静态障碍：**not_evaluated**，因为当前三类执行器未接入相同障碍几何。

## MMD

MMD 不在本表：其共同地图规划未完成（见 `12_mmd_common_scene.md`）。
没有把 MMD 原生 3 机器人 Empty 结果混入 4 机器人 No02 表。

## 暂停

三类方法使用同一 `make_pause_events(task,p,seed)` 事件表。
ORCA/MPC-CBF 执行自己的制动/保持逻辑；暂停接受/拒绝差异在原始记录中保留。

## 评价口径

- 碰撞：ORCA 连续子步、MPC-CBF 连续子步、Ours 冻结连续碰撞；
- 安全完成：全员到达且无机器人间碰撞；静态障碍未评价，故不能称为“总体安全”；
- Makespan：成功/超时都保留，表中为记录均值；
- 路径偏离：Ours 固定路径为 0；ORCA 用实际轨迹到参考折线距离；
- 在线时间：Ours 为决策 P95；MPC-CBF 为 IPOPT P95；ORCA 当前未做逐步计时（0 表示缺失，不是零开销）。

## 公平性边界

- Ours 使用冻结 W2+ 初始等待；ORCA/MPC-CBF 使用原始等待 `w0`；
  这是“完整系统 vs 固定路径执行器”的差异，解读时必须说明；
- Ours 的 0 路径偏离是设计属性（固定路径），不是测量缺失；
- 没有按结果筛选任务；4 个任务在运行前固定。
