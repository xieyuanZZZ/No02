# 02 在线时序协调原型设计（A/B/C）

**日期**：2026-11-05  
**代码**：`src/fhd/execution/coordinator.py`、`src/fhd/optim/online_baseline.py`、`src/fhd/optim/online_firsthit.py`、`scripts/online_temporal_benchmark.py`  
**参数**：`configs/online_temporal.yaml`

---

## 1. 统一接口与公平性

```text
Simulator 每个 dt_slot=0.4 s：
  1) 先注入预生成的随机暂停事件（只有仿真器持有事件表）
  2) 读取所有机器人真实状态 -> Observation
  3) Coordinator.decide(Observation) -> [Action]
  4) Executor.request_stop_at(...) 检查可行性并接受/拒绝
  5) 所有机器人推进一个 dt_ctrl=0.1 s 控制周期
  6) 记录碰撞/进度/事件/决策耗时
```

- `Observation`：时间、每台机器人 `[x,y,vx,vy]`、速度、弧长 `s`、模式、
  剩余保持时间、已接受在线等待、最后一个在线停车时刻、长度、候选停车点；
  **不包含**随机事件表、未来轨迹或测试 seed 派生信息。
- `Action`：机器人编号、`kind="stop"`、目标弧长、等待时长、候选航点、原因、
  预测风险前后值与额外时间。
- 随机暂停与在线等待在 `SingleRobotExecutor` 中用 `source=random|online` 分开；
  同一机器人处于 BRAKE/HOLD 时新命令拒绝，不覆盖。
- A 的 Coordinator 为 `None`（W2+ 计划，执行期不看状态）；B/C 使用相同的
  `simulate_mpc`、MPC 参数、事件表和候选点接口。
- 所有方法在 `dt_slot` 决策；C 的单次搜索被限制在少量候选动作，耗时实测在
  §4；超 0.4 s 计入 `coord_over_budget`，不延长物理时间。

---

## 2. 物理停车命令

`request_stop_at(s_target, duration_s, source, reason, margin=0.05)`：

```text
dist = s_target - s_act
brake = v_now^2 / (2 a_brake)
lead  = v_now * dt_ctrl
接受 iff dist >= brake + lead + margin 且 s_target 未越过、在路径内、机器人 NORMAL
```

拒绝原因：`mode_BRAKE_*`/`mode_HOLD`、`target_behind`、`target_beyond_path`、
`cannot_stop`、`already_done`、`planned_stop_imminent`。理论刹车距离只用于
筛选，实际可行性仍由 MPC 仿真检验。

候选点 `feasible_stop_candidates(lookahead_m)`：只取未越过、不是计划等待点、
不是终点、且满足上式的前方航点。B/C 共用同一候选集合。

---

## 3. B：简单在线冲突预测

1. 用当前速度沿冻结路径外推 3.0 s，并截断在下一计划停车点；制动/等待中的
   机器人按静止处理；
2. 每对求预测最小距离，`d_min < d_safe + 0.15 m` 记为冲突；
3. 冲突对中“离冲突点更远”的机器人让行；距离接近（<0.10 m）时编号大者让行；
4. 在冲突点之前、候选集合中选最靠后的可行停车点，等待 1 个时隙；
5. `cooldown=1.2 s`、`max_online_waits=6` 防抖；无可行点记录
   `no_feasible_stop`，不发命令。

B 不使用 First-Hit DP。

---

## 4. C：当前状态条件化的 First-Hit 在线等待

1. 对每台机器人，从当前 `s` 截断出剩余路径：`[current_pos, wp_next, ..., goal]`；
   剩余等待映射到对应航点；HOLD/BRAKE 的机器人把当前等待写进首个位置；
2. 用现有 `FirstHitRisk`（`collision_mode="hard"`, `gate_mode="hard"`）在
   3.2 s（8 个时隙）上计算 `pair_sum_risk`；它只作为排序目标，不是全队概率；
3. 枚举“某机器人停在某个可行候选点 1 个时隙”的动作（每机器人最多 2 个候选）；
4. 选择 `risk_after + λ_time·extra_time` 最小且风险下降 ≥0.02 的动作；
5. 每次只提交一个动作，下一周期重新观察；若决策超 0.4 s，退回 B 的动作并标记。

**明确近似**：这是条件化近似风险，不是校准的 MPC 真实碰撞概率；剩余路径 DP
从当前进度 0 开始，首段长度可能短于名义段长；不读取未来事件。

---

## 5. 指标与统计

每个 run 记录：机器人间碰撞、逐对首次时间/最小距离、静态障碍碰撞、安全完成、
超时、控制器失败/降级、完成时间、总等待/计划等待/随机等待/在线等待、请求/接受/
拒绝、MPC 求解统计、在线决策 mean/P95/P99/max、超预算次数。

A/B/C 按 `(task_id, p, seed)` 配对；以任务为聚类单位计算 B−A、C−B、C−A 的
任务级 bootstrap 区间。不把 6 个 pair 或 30 个 seed 当独立任务。

---

## 6. 设计边界（首版故意不做）

- 不做无通信/通信延迟、分布式协调；
- 不做几何重规划、绕路、速度瞬时修改；
- 不做“提前恢复”动作（等待时长到期由执行器自动恢复，避免状态机竞争）；
- 不做在线 W2+ 全量搜索、不做神经网络；
- 不声称已有安全保证；静态障碍碰撞与机器人间碰撞分开报告。
