# 00 在线时序协调：代码审计与冻结基线

**日期**：2026-11-05  
**仓库**：`https://github.com/xieyuanZZZ/No02`  
**起点审查提交**：`419b6cd`  
**本轮工作分支**：`experiment/online-temporal-coordination`  
**本轮实际 base**：`9f47d19`（= `419b6cd` + 已完成的 P0-v2 修复 + M1/M2/M3 负结果；本轮不重置、不覆盖已有成果）  
**远端核对**：`git fetch origin` 后 `origin/experiment/online-dynamic-firsthit` 仍为 `419b6cd`；本地领先的提交均为本轮之前已提交的修复/报告。

**环境**：Python 3.12.13，numpy 2.5.4，scipy 1.18.1，torch 2.11.0+cu128，OSQP 1.1.3，Windows 16 逻辑核。  
**工作区状态**：创建分支时干净（`git status` 无修改）。

---

## 1. 已存在且可复用的接口

| 文件 / 函数 | 实际输入 → 输出 | 现状 |
|---|---|---|
| `src/fhd/control/linear_mpc.py::LinearMPC` | `x0[4]`, `x_ref[Np+1,4]` → `U[Np,2]`, `info` | 预测矩阵正确；Δu 首项已与相邻差分分离；失败有可行制动回退；`SolveStats` 记录每次调用 |
| `src/fhd/control/linear_mpc.py::LinearMPC._assemble_qp` | `x0/x_ref` → `H,f,A,l,u` | 与 `explicit_cost` 一致；k=0 速度上界有数值容差 |
| `src/fhd/dynamics/trajectory_timing.py::TimeParameterizedPath` | 路点 `[H,2]` + 等待 `[H]` → `s_wp/v_grid/t_grid/dwell` | 到达/离开时刻已分离；等待归属为“到达航点后” |
| `...::MonotoneProgress` | 位置 `[2]` → `(s, lateral)` | 历史邻域约束，进度单调 |
| `src/fhd/execution/mpc_executor.py::SingleRobotExecutor` | 路径、双积分器、MPC 参数；`step()` 推进 `dt_ctrl` | NORMAL/BRAKE_DWELL/BRAKE_PAUSE/HOLD/CREEP；停稳后计时；事件账本 |
| `...::request_pause(duration_s)` | 暂停时长 → `bool` | **只按当前速度预测停车点**，没有“指定未来停车航点”的接口；没有区分随机/在线来源 |
| `...::_next_dwell` | `s_act` → 下一个计划等待航点 | 只服务计划等待，不暴露在线候选停车点 |
| `src/fhd/execution/mpc_simulator.py::simulate_mpc` | `paths[N,H,2]`, `wait[N,H-1]`, `p`, `cfg`, `seed` | 0.1 s 循环；子步连续碰撞；静态障碍；暂停事件表在内部生成 |
| `...::make_pause_events` | `(seed,p,N,T)` → `[(slot,robot,duration)]` | 预生成，独立于方法 |
| `src/fhd/geometry/continuous_collision.py` | 位置序列 → 首次越界/最小距离 | `dt_dyn` 子步级解析解；t=0 重叠处理 |
| `src/fhd/optim/wait_policies.py::policy_w0` | `w0/mask/budget/risk/paths/pause_p/T` → `WaitResult` | 原始等待基准 |
| `...::policy_w2plus` | 同上 + `lam_T/max_evals/restarts/seed` → `WaitResult` | 离线整数坐标下降；`hard_risk`、`finish_s`、`dp_evals` |
| `src/fhd/risk/first_hit_dp.py::first_hit_dp` | `collision[P,4,H,H]`, `a_i/a_j[P,T,H]` | 两两首次碰撞 DP；**初始进度固定为 0**，没有条件化起点 |
| `src/fhd/risk/multi_pair.py::FirstHitRisk` | `paths[N,H,2]`, `wait[N,H-1]`, `pause_p`, `T` | 硬/软 DP；`pair_risks`、`pair_sum_risk`；没有“当前进度”入口 |
| `scripts/mpc_execution_benchmark.py` | D1/D2 配对基准 | 已输出 `per_run/per_task/summary`；本轮沿用其统计口径 |
| `tests/test_mpc_*.py`, `test_execution_*.py` | 199 项 | 全过；已有 QP/时序/碰撞/统计回归 |

---

## 2. 与本任务书要求的差距

### 2.1 停车命令

- `request_pause()` 把停车目标设为 `当前 s + v²/(2a)`；没有输入“未来允许停车的路径位置”，也没有返回拒绝原因结构。
- 在线协调需要：给定候选航点 `s_target`，检查 `s_target - s_act >= v²/(2a) + v·dt_ctrl + margin`，再决定接受/拒绝。
- 需要区分 `source=random|online`，并分别统计请求/接受/拒绝/等待时长；随机暂停与在线等待不能重复计时。
- 机器人处于 BRAKE/HOLD 时，新命令首版必须拒绝，不能静默覆盖。

### 2.2 在线循环

- `simulate_mpc` 在控制循环内直接处理随机暂停事件，没有 Coordinator 钩子。
- 需要新增可选参数（如 `coordinator=None`），在 `dt_slot` 边界构建 `Observation → decide() → Action`，再执行一个控制周期。
- 随机事件表只能由仿真器用于“注入随机暂停”，不能传给 Coordinator。
- 需要记录在线决策耗时（mean/P95/P99、超 0.4 s 次数）、动作、接受/拒绝原因。

### 2.3 B/C 的方法边界

- B 需要短时预测（当前速度 + 可行加减速/名义参考）和固定优先级让行；不使用 First-Hit DP。
- C 需要从**当前实际进度**初始化 First-Hit；现有 `FirstHitRisk` 从 0 进度开始，不能直接复用。
- 首版 C 的可行做法：把每台机器人从当前位置截断成“剩余路径”，用剩余航点+剩余等待构造 `paths_res/wait_res`，再调用现有 `FirstHitRisk`；明确这是条件化近似，不是校准的 MPC 概率。
- A/B/C 必须共享同一停车点集合、动作约束、决策周期、MPC 参数和随机事件表。

### 2.4 实验口径

- 需区分：机器人间碰撞、静态障碍碰撞、安全完成（无机器人间碰撞且全部到达）、超时、控制器失败/降级、不可行停车。
- 统计以任务/地图为聚类单位，不能把 6 个 pair × 30 seeds 当 180 个独立任务。
- 不允许用无限等待换低碰撞；必须同时看安全完成率、makespan、总等待。

---

## 3. 本轮拟改动范围（最小）

1. `mpc_executor.py`
   - 新增 `feasible_stop_candidates(lookahead_m)`；
   - 新增 `request_stop_at(s_target, duration_s, source, reason)`，带可行性检查与拒绝原因；
   - `request_pause()` 改为内部调用，`source="random"`；
   - 事件账本/统计新增 source、online 等待时长、拒绝原因计数。
2. `mpc_simulator.py`
   - 新增 `coordinator` 可选参数与 `dt_slot` 决策边界；
   - 构建 `Observation`，应用 `Action`，记录在线决策日志/耗时；
   - 保持 `coordinator=None` 时行为与现有基准一致。
3. 新增 `src/fhd/execution/coordinator.py`：`RobotObservation`/`Observation`/`Action` 与 `Coordinator` 基类。
4. 新增 `src/fhd/optim/online_baseline.py`（B）与 `src/fhd/optim/online_firsthit.py`（C）。
5. 新增 `configs/online_temporal.yaml`、`scripts/online_temporal_benchmark.py`、`scripts/online_temporal_visualize.py`、测试与文档。

**不会改动**：Diffusion 权重、几何路径、W2+ 目标函数/预算、`d_safe=0.35`、轨迹 H=32、MPC 速度/加速度默认界、M1/M2/M3 模型。

---

## 4. 阶段一已有修复的对应关系

本分支已包含任务书 §1.1–§1.3 的主体修复（提交 `1fb1f2f`、`16de71b`、`8f8e55d`）：

| 任务书条目 | 已有实现 | 本轮是否需补 |
|---|---|---|
| 1.1 Δu 代价 | `_build_difference_matrix` + `explicit_cost` 测试 | 否 |
| 1.2 可行制动/等待 | 制动剖面 + CREEP + 停稳计时 | 补“指定目标停车点/来源”接口 |
| 1.3 连续碰撞/到达/终止 | `continuous_collision` + 弧长跨越 + 超时/安全完成 | 补控制器失败显式字段与在线日志 |
| 1.4 阶段验收 | `results/Online_FirstHit_v2/02_mpc_benchmark` 已有 10,560 条；仍需按本任务书在冻结 N4 任务上重跑并保存到 `results/Online_Temporal/01_execution_fix` | 是（重跑/归档） |

审计结论：**不重复重写已正确的预测矩阵、QP 与连续碰撞数学**；本轮只在执行器/仿真器上增加在线所需的最小接口，并用新实验重新落盘。
