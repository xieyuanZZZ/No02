# 04 闭环验证：Coordination ORU + No02 MPC

**日期**：2026-11-06  
**结果**：`results/Continuous_Baseline_Comparison/pilot/oru/`

## 修复内容

- 取消固定 3 s 等待；在线每 0.4 s 重新读取 ORU critical point，等待时长 0.4 s；
- 停车目标改为 critical point 之前、按当前速度可刹停的最晚可行航点；
- 若 critical point 已到达/越过，不伪造停车，计入 `late_inside`；
- 若无可行停车点，不伪造目标，计入 `no_candidate`；
- 执行器拒绝不再抑制下一次命令：`_last_t` 只在 `on_result(accepted=True)` 更新；
- Java bridge 保留 ORU 原 trajectory envelope/critical section/precedence。

## 实际结果

4 个 N4 任务、p=0/0.1/0.2、每格 5 seeds：

| 方法 | p | 碰撞 | 安全完成 | Makespan | 主动在线等待 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Ours（冻结） | 0 | 0.00 | 1.00 | 18.40 s | 0.00 s | 6.3 ms |
| Ours（冻结） | 0.1 | 0.00 | 1.00 | 21.13 s | 0.48 s | 15.0 ms |
| Ours（冻结） | 0.2 | 0.05 | 0.95 | 24.10 s | 1.18 s | 20.7 ms |
| ORU | 0 | 0.75 | 0.25 | 17.84 s | 1.56 s | 94.9 ms |
| ORU | 0.1 | 0.80 | 0.20 | 20.57 s | 1.42 s | 94.9 ms |
| ORU | 0.2 | 0.55 | 0.45 | 23.71 s | 1.64 s | 94.2 ms |

ORU 的主动等待只有 1.4–1.6 s（不再靠固定 3 s 拖时间），但仍大量碰撞；
这不是 ORU 原算法结论，而是当前适配的失败。

## 诊断

- ORU critical point 是高层“不得越过”索引；No02 冻结执行器只有离散航点停车接口，
  没有 ORU 原生连续减速/速度跟踪；
- 即使选择 CP 前最晚可刹停航点，ORU 仍可能在机器人已接近冲突区时才更新 CP，
  `late_inside`/`no_candidate` 会记录这些不可行情况；
- 因此当前 ORU 仍是“高层协调 + 低层停车”的近似，**未 VERIFIED**，不能进入正式表。

## 下一步

在 ORU Java bridge 中输出 continuous critical-section entry/exit 与建议速度上限，
由外部专用执行封装做连续制动/保持/释放；不能修改冻结 `request_stop_at`。
