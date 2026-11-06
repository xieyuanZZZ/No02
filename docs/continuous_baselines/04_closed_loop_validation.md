# 04 闭环验证：Coordination ORU + No02 MPC

**日期**：2026-11-06  
**方法**：ORU 原生 trajectory envelope/critical section/precedence 机制 + No02 冻结 MPC 执行器  
**结果**：`results/Continuous_Baseline_Comparison/pilot/oru/`

## 1. 实现

- `src/fhd/adapters/coordination_oru_adapter.py`
- `third_party`：`D:\No02_external_repos\coordination_oru`（GPL-3.0，commit `7332cde...`）
- Java bridge：`No02OruBridge`，JSON-lines 协议；每个 0.4 s 接收机器人实际状态
  `(x,y,vx,vy,path_index)`，返回 ORU 选择的 critical point。
- ORU 的原始 Java 协调器运行在独立进程；Python 只做状态输入/停车目标转换。
- 停车目标优先选 **critical point 之前、按当前速度可刹停的最晚可行航点**；
  目标交给冻结 `request_stop_at()` 检查，拒绝原因进入统计。
- ORU 的 critical point 是“不得越过”的路径索引，不是瞬时速度指令。

**没有**修改 First-Hit、W2+、联合方法、MPC、Diffusion、路径和旧结果。

## 2. 闭环配置

- 4 个 N4 任务（`open_crossing_300000`/`300003` 各 2 个）；
- p = 0 / 0.1 / 0.2；每格 5 seeds；
- 每格 20 个配对 run，共 120 条执行（Ours + ORU）；
- 同路径、同基线等待、同随机事件表、同 `d_safe=0.35`、同 MPC 参数；
- ORU 适配器 hold = 3.0 s（等待 ORU 释放 critical point；不是 ORU 参数）。

## 3. 实际结果

| 方法 | p | 机器人间碰撞 | 安全完成 | Makespan | 在线等待 | 在线 P95 |
|---|---:|---:|---:|---:|---:|---:|
| Ours（冻结） | 0 | 0.00 | 1.00 | 18.40 s | 0.00 s | 6.7 ms |
| Ours（冻结） | 0.1 | 0.00 | 1.00 | 21.13 s | 0.48 s | 16.2 ms |
| Ours（冻结） | 0.2 | 0.05 | 0.95 | 24.10 s | 1.18 s | 20.4 ms |
| Coordination ORU | 0 | 0.55 | 0.45 | 26.48 s | 21.15 s | 91.9 ms |
| Coordination ORU | 0.1 | 0.80 | 0.20 | 28.99 s | 22.65 s | 91.3 ms |
| Coordination ORU | 0.2 | 0.60 | 0.40 | 33.19 s | 26.25 s | 91.4 ms |

- 所有 run 均无 MPC 失败/超时；ORU 侧有少量 `cannot_stop` 拒绝，已计入日志；
- ORU 在线 P95 约 92 ms，低于 400 ms 周期；
- Ours 在线 P95 6.7–20.4 ms。

## 4. 如何解释

- ORU 的 critical point 是高层“某机器人应在冲突区前停车”的决策；映射到
  No02 的离散航点停车接口时，**丢失了 ORU 原生 RK4 tracker 的连续减速/速度跟踪**。
  因此这组 ORU 数字是“ORU 高层协调 + No02 低层停车接口”的适配结果，
  不是 ORU 原生控制器的最好成绩。
- ORU 在 p=0 也出现 55% 碰撞，说明当前映射仍有明显问题（它等待很久但未完全
  消除冲突），不能据此宣称“ORU 本身不安全”。
- Ours 的优势是**在线决策快、等待少**；但这是 4 任务/5 seeds 的 pilot，
  不能作为论文最终结论。

## 5. 结论与下一步

- ORU 已真正接入 No02 MPC 并产生闭环动作/碰撞/等待数据；
- 但 ORU 适配器仍需改进：需要把 critical point 映射到**连续速度剖面**，
  或让 ORU 的低层 tracker 与 No02 MPC 共用状态，而不是只在航点停车；
- 在修正 ORU 适配前，不把这张表作为“Ours 优于 ORU”的论文证据；
- ORCA、MPC-CBF、MMD 尚未完成同等闭环适配。
