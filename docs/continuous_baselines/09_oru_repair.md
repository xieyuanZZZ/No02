# 09 Coordination ORU 修复报告

**结果**：`results/Continuous_Baseline_Comparison/repair_v2/coordination_oru/pilot/`

## 已修复

- 去掉固定 3 s 等待，改为每 0.4 s 重新读取 critical point；
- 停车点从离散航点改为“CP 之前、按当前速度可刹停的最晚航点”；
- CP 已越过时不再伪造停车，记录 `late_inside`；
- 无可行停车点时不再伪造，记录 `no_candidate`；
- 执行器拒绝命令不再抑制下一次命令，`_last_t` 只在 `on_result(accepted=True)` 更新；
- Java bridge 保留 ORU trajectory envelope/critical section/precedence。

## 实际结果（4 任务 × 3 p × 5 seeds）

| 方法 | p=0 | p=0.1 | p=0.2 |
|---|---:|---:|---:|
| Ours 碰撞 | 0.00 | 0.00 | 0.05 |
| ORU 碰撞 | 0.75 | 0.80 | 0.55 |
| ORU 主动等待 | 1.56 s | 1.42 s | 1.64 s |
| ORU P95 | 94.9 ms | 94.9 ms | 94.2 ms |

去掉固定 3 s 后没有改善，说明问题不是等待时长，而是 **CP→离散停车接口**：

- ORU 原始 tracker 用连续减速曲线；
- No02 冻结执行器只有航点/弧长停车，且一旦 HOLD 结束自动恢复；
- CP 可能在机器人已接近冲突区时才更新，适配器只能记录 `late_inside`；
- 没有真正的连续 HOLD/动态 RELEASE，因此 p=0 仍大量碰撞。

## 状态

**ADAPTED，未 VERIFIED**。下一步必须在外部基线专用执行封装中实现
APPROACH/BRAKE/HOLD/RELEASE 状态机，并让 Java bridge 输出连续 CP/冲突区位置；
不能靠增加固定等待时间。

## 公平性

ORU 使用 `w0` 初始等待，Ours 使用 W2+；同一 CP 适配下的“共同 w0”对照已在
`oru_pilot.py` 中以 `w0` 运行，但尚未作为正式表报告。ORU 仍不能进入正式比较。
