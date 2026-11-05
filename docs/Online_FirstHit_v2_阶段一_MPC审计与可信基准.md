# Online First-Hit v2 阶段一：MPC 审计、执行时序与可信基准

**日期**：2026-11-05  
**工作分支**：`fix/mpc-firsthit-foundation-v2`（基线 `experiment/online-dynamic-firsthit`，起点 `419b6cd`）  
**阶段代码提交**：`1fb1f2f`（MPC 增量代价）、`16de71b`（等待/到达/时间参数化）、`8f8e55d`（连续碰撞与基准）  
**范围**：只做阶段一（P0-v2）。未进入在线滚动调整、P3 条件风险更新、P4 在线等待修正。

> **一句话结论**：MPC 增量代价、执行时序、连续碰撞与统计口径已修复并通过 199 项测试；
> D2 正式集 10,560 条运行 0 次 MPC 失败、0 次不可行停车、0 次超时。W2+ 在 D2 下
> 仍显著降低 N2/N4 碰撞率，但收益小于 D1；冻结计划库在 4 张地图上存在真实的
> **静态障碍碰撞**，已如实单列，未混入 First-Hit 标签。

---

## 0. 检查点状态

| 检查点 | 内容 | 状态 |
|---|---|---|
| G0 | 核对分支/HEAD/环境，冻结配置与旧结果 | **通过** |
| G1 | QP、状态机、到达记录、连续碰撞、统计修复，回归全过 | **通过** |
| G2 | D1/D2 基准重跑，结果逐任务可追溯 | **通过**（见 §4） |
| G3 | M1/M2 修复与隔离 | 见阶段二报告 |
| G4 | M3 状态模型与 smoke | 见阶段二报告 |
| G5 | M0–M3 盲测 | 见阶段二报告 |

**环境**：Python 3.12.13，numpy 2.5.4，scipy 1.18.1，torch 2.11.0+cu128，OSQP 1.1.3，Windows，16 逻辑核。

**G0 冻结记录**：开始时 `git status` 干净、HEAD=`419b6cd`；旧分支与旧结果目录只读保留。本轮只向
`results/Online_FirstHit_v2/` 写新数据。

> 说明：正式基准进程在本地提交完成前启动，因此其 `run_meta.json` 的 `git_sha` 记录为启动时的
> `419b6cd`；进程启动后只有一处求解器容差（`linear_mpc._assemble_qp` 的 k=0 速度上界）
> 在运行期间被修改。该改动只影响“当前速度因浮点误差略超 `v_max` 时是否误报
> primal infeasible”，不改变碰撞几何、执行器时序或风险模型；运行结果中的
> `d2_mpc_failures` / `d2_solve_statuses` 会如实暴露任何回退。代码对应提交为
> `1fb1f2f` + `16de71b` + `8f8e55d`。

---

## 1. 逐项修复

### 1.1 MPC 增量代价（`src/fhd/control/linear_mpc.py`）

**旧错误**：目标里的平滑项写成“每个相邻差分都减去上一拍控制”：

```python
du_prev = np.tile(self._last_u, Np - 1)
f = f - 2 * (D.T @ S @ du_prev)
```

这等于把 `u_{-1}` 施加到所有 `Np-1` 个差分上。正确目标只有第一项使用 `u_{-1}`：

$$J_{\Delta u}=\lambda_{\Delta u}\left(\|u_0-u_{-1}\|^2+\sum_{k=1}^{N_p-1}\|u_k-u_{k-1}\|^2\right).$$

**修复**：新增 `_build_difference_matrix()`，构造 `D ∈ R^{2Np×2Np}`：
第一行块为 `I`（`u_0`），其余行块为 `[-I, I]`（`u_k-u_{k-1}`）；
常数向量 `d_prev=[u_{-1},0,…,0]`。于是

$$H=2(G^\top QG+R+D^\top SD),\qquad f=2G^\top Q(\text{bias}-X_{ref})-2D^\top S d_{prev}.$$

`explicit_cost()` 与 QP 使用同一 `H/f`，测试逐项对比。

**同时修复**：

- `SolveStats` 记录**每一次**调用的耗时与状态，给出 mean/P50/P95/P99/failure_rate，而不是“最后一次耗时”；
- solver 缓存用 `H/A` 的 blake2b 签名判断，只有数值/稀疏结构变化才重建；
- 求解失败不再“重用上一拍加速度”，改为受 `a_max` 限幅的制动/零控制，并检查
  速度界与不反向；连该动作也不可行时返回 `usable=False`；
- 删除积分后的速度裁剪；越界只如实计数，由 QP 约束与保守回退负责；
- k=0 速度上界允许当前状态的浮点微小超界，避免把数值误差误报为 primal infeasible。

**定向测试**（`tests/test_mpc_qp_fixes.py`）：

- `test_qp_objective_matches_explicit_cost`：QP 二次型与显式目标一致（差 <1e-6，来自 1e-8 正则）；
- `test_qp_gradient_matches_explicit_cost_finite_difference`：梯度与有限差分一致（<1e-5）；
- `test_first_increment_penalty_is_separate_from_adjacent_differences`：`u_0-u_{-1}` 与相邻差分分离；
- `test_solver_cache_rebuilt_only_when_matrices_change`；
- `test_failed_solve_uses_feasible_braking_fallback`、`test_infeasible_fallback_is_reported_not_silently_fixed`；
- `test_solve_does_not_clip_to_velocity_limit_in_postprocessing`。

### 1.2 等待、暂停、停车与到达事件（`mpc_executor.py`、`trajectory_timing.py`）

**旧问题与修复**：

1. **计划等待**：旧实现进入 HOLDING 就开始计时；现在必须 `v<V_STOP` 才 `hold_start`，
   `hold_until = hold_start + dwell[k]`。制动时间、等待时间、爬行时间分账。
2. **随机暂停**：`request_pause()` 只在 NORMAL 接受；接受时设置 `_hold_kind='pause'`、
   清空 `_dwell_idx`。暂停时长从**停稳**起算。若下一计划停车点在制动距离 +0.15 m 内，
   暂停被拒绝并记录原因（避免与计划停车重叠）。
3. **物理可停车性**：进入制动前用“一步预瞄 + 最坏加速”判断；确实刹不住时
   `infeasible_stops += 1` 并跳过该等待点，不把越过航点当作执行成功。
4. **航点到达**：改为按实际弧长跨越检测；一次控制步可跨越多个航点，按比例插值记录全部；
   制动停稳时补记到达；起点 `t=0` 记到达。
5. **投影稳定性**：`MonotoneProgress` 在历史邻域内搜索最近点，进度单调，保存横向误差；
   弯折/近距离平行段不再倒跳。
6. **时间参数化**：`nominal_arrival_times()` 改为
   `arrival[k] = t_grid(s_wp[k]) + Σ_{h<k} dwell[h]`，
   `departure[k] = arrival[k] + dwell[k]`；不再把累计时间重复累加。
7. **等待归属**：`w[k]` 统一为“到达航点 k 后、离开前等待”。`_scheduled_speed_cap()`
   不再把等待摊进前一段的可用时间，每段行驶时间仍为 1 个时隙；等待由状态机停车执行。
8. **统计**：每航点/每段保存 `arrival_time/depart_time/move_time/planned_hold/
   random_hold/brake_time/resume_time/valid/censored`，右删失段 `dt=None`，不用中位数冒充。

**执行器实现细节**（本轮定位并修复的实际问题）：

- 制动参考若钉在“预测停车点”而实际已越过，会让 MPC 反向加速，出现 0.04 m/s 长时间
  爬行与二次制动；现在制动段改用显式速度伺服：
  `u=(v_des·tangent − v)/dt`，`v_des=sqrt(2 a_brake max(d−v dt,0))`，
  限制在 L1 加速度球内。
- 停短后增加 `CREEP_DWELL`：以 0.12 m/s 爬到精确停车点，再开始等待计时；
  不会把“差 2 cm”变成死锁。
- 这两个修复把单次制动的 1.5–2 s 量级拖尾降回 0.5–0.9 s，并消除了大量
  `infeasible_stops`（阶段一正式集中为 0，见 §4）。

**定向测试**（`tests/test_execution_timing.py`）：

无等待直线、起点等待、中间航点等待、连续两次等待、制动中请求暂停被拒、
停稳后计时、dwell 后暂停不污染、一次跨越多个航点、U 形回程投影单调、
超时右删失、完成后拒绝暂停。

### 1.3 连续时间碰撞（`geometry/continuous_collision.py`、`mpc_simulator.py`）

**旧错误**：每个 `dt_ctrl` 区间取 4 个离散位置；时间戳 `t-dt_ctrl+sub*...` 可能为负或
提前一个控制周期；t=0 重叠、终点停留未覆盖。

**修复**：执行器每个 `dt_dyn` 子步输出位置；对同一子步区间：

$$r_0=p_i(t)-p_j(t),\quad \Delta r=[p_i(t+\Delta t)-p_i(t)]-[p_j(t+\Delta t)-p_j(t)],$$
$$\alpha^*=\mathrm{clip}\left(-\frac{r_0^\top\Delta r}{\|\Delta r\|^2},0,1\right),$$
$$d_{\min}=\|r_0+\alpha^*\Delta r\|.$$

首次越界用二次方程最小根，时间戳 = 区间起点 + α·Δt；t=0 初始重叠 α=0。
`dr≈0` 按静止处理；恰好切阈值（disc=0）不算碰撞。每对分别保存 `first_hit_time/
min_distance/collision`；全队首次时间取各对最小值。

**静态障碍**：按 `map_id` 从 `data/revised_s1/maps_*.json` 载入障碍，用机器人圆盘沿实际
子步轨迹扫掠，记录 `min_obstacle_clearance` 与障碍碰撞；地图库没有对应项时显式
`not_evaluated`，不输出虚假的 0 碰撞。

**定向测试**（`tests/test_mpc_continuous_collision.py`）：

- 两端安全、中途交叉（解析 α=0.325/0.376）；
- 同向追尾、平行不碰撞、完全静止重合、恰好切阈值；
- 初始重叠 t=0；
- 时间戳在 `[0,T]` 内且等于各对最小值；
- `dt_dyn=0.05/0.025` 检测稳定；
- N=4 六对独立。

### 1.4 统计与复现协议（`mpc_simulator.py`、`mpc_execution_benchmark.py`）

- `solve_stats` 汇总**全程所有** MPC 调用：mean/P50/P95/P99/max、failed/infeasible/
  inaccurate/fallback/projected、statuses、failure_rate。
- `completion_time` 区分真实完成与超时截断；`time_censored/timeout/all_arrived` 分开。
- 区分 `pair_collision / team_collision / safe_completion / timeout /
  infeasible_stop / constraint_violation`；两两风险之和不等于全队概率。
- `record=True/False` 只影响 `trajectories/modes` 是否保存；安全、完成、跟踪、求解统计逐项一致
  （`tests/test_execution_stats.py::test_record_true_false_have_identical_metrics`；
  墙钟时间除外）。
- 预生成暂停事件表 `(slot, robot, duration)`，同种子同表；D1/D2 的事件语义差异写入
  `run_meta.json`，不宣称“同一物理扰动”。
- 输出 `config.json / run_meta.json / per_run.csv / per_task.csv / summary.json /
  figures/`；per_task 含 Wilson 区间与任务级 bootstrap 区间。

---

## 2. 测试证据

**阶段一专项回归**（实际执行）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_mpc_qp_fixes.py `
    tests\test_mpc_fixes.py tests\test_mpc_execution.py `
    tests\test_execution_timing.py tests\test_execution_stats.py `
    tests\test_mpc_continuous_collision.py -q
```

结果：**全部通过**；新增定向回归 42 项（QP 9、时序 12、统计 7、连续碰撞 14），
另有原有 MPC 测试 23 项。全套 `python -m pytest tests -q` 最终为 **199 passed, 0 failed**
（2026-11-05 实际运行；耗时约 2 min；包含阶段二新增测试）。

**关键的“旧代码会错、新代码通过”证据**：

| 测试 | 旧行为 | 新行为 |
|---|---|---|
| QP 目标 vs 显式目标 | `u_prev` 进入所有差分，误差 O(Np) | 误差 <1e-6（仅 1e-8 正则） |
| `test_dwell_starts_only_after_full_stop` | 进入 HOLD 即计时，等待偏短 | 停稳后 0.8 s，误差 <0.12 s |
| `test_per_pair_collision_stats` | 只记全队第一次 | N=4 六对独立 |
| 连续碰撞解析 α | 4 点采样可漏、时间戳偏移 | 与二次方程解析解一致 |
| `record=False` 跟踪误差 | 显示 0 | >0，且与 record=True 一致 |

**未删除/放宽任何失败断言**；旧测试全部保留并通过。

---

## 3. 阶段一结果

### 3.1 验收小样本（`01_mpc_benchmark_smoke`）

设置：N2 8 任务、N4 8 任务，p∈{0,0.1,0.2}，每格 6 MC，配对种子。

| 组 | 计划 | p | D1 碰撞 | D2 碰撞 | D2 安全完成 | 超时 | 不可行停车 | MPC 失败 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| N2 | 原始 | 0 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| N2 | 原始 | 0.1 | 0.125 | 0.125 | 0.875 | 0 | 0 | 0 |
| N2 | 原始 | 0.2 | 0.104 | 0.208 | 0.792 | 0 | 0 | 0 |
| N2 | W2+ | 0 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| N2 | W2+ | 0.1 | 0.000 | 0.021 | 0.979 | 0 | 0 | 1 |
| N2 | W2+ | 0.2 | 0.083 | 0.125 | 0.875 | 0 | 0 | 0 |
| N4 | 原始 | 0 | 0.000 | 0.125 | 0.875 | 0 | 0 | 0 |
| N4 | 原始 | 0.1 | 0.667 | 0.771 | 0.229 | 0 | 0 | 0 |
| N4 | 原始 | 0.2 | 0.833 | 0.854 | 0.146 | 0 | 0 | 2 |
| N4 | W2+ | 0 | 0.000 | 0.125 | 0.875 | 0 | 0 | 0 |
| N4 | W2+ | 0.1 | 0.062 | 0.438 | 0.562 | 0 | 0 | 1 |
| N4 | W2+ | 0.2 | 0.312 | 0.646 | 0.354 | 0 | 0 | 3 |

其它：跟踪 RMSE 均值 0.0081 m，最大速度 1.40 m/s，最大加速度 2.00 m/s²，
P99 求解耗时 2.51 ms，`infeasible_stops=0`、`constraint_violations=0`，
MPC 失败 7/576（1.2%，全部为 `primal infeasible` 后的可行制动回退）；
静态障碍指标对 576/576 条运行均为 `evaluated`（这些地图无障碍，净空 +∞）。

### 3.2 正式固定集（`02_mpc_benchmark`）

设置：N2 全部 24 任务、N4 全部 20 任务；p∈{0,0.05,0.1,0.2}；每格 30 个独立事件种子；
同一任务/计划/种子配对 D1 与 D2。共 **10,560 条 per-run 记录**，总耗时 120.7 min。

| 组 | 计划 | p | D1 碰撞 | D2 碰撞 | D2 安全完成 | 超时 | 不可行停车 | MPC 失败 | 约束违反 | 求解均值/P95 (ms) | 跟踪 RMSE (m) | 最小间距 (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N2 | 原始 | 0 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 | 0 | 0.46 / 0.64 | 0.0021 | 0.440 |
| N2 | 原始 | 0.05 | 0.046 | 0.085 | 0.915 | 0 | 0 | 0 | 0 | 0.47 / 0.68 | 0.0061 | 0.003 |
| N2 | 原始 | 0.1 | 0.099 | 0.126 | 0.874 | 0 | 0 | 0 | 0 | 0.51 / 0.76 | 0.0091 | 0.000 |
| N2 | 原始 | 0.2 | 0.132 | 0.144 | 0.856 | 0 | 0 | 0 | 0 | 0.51 / 0.79 | 0.0136 | 0.004 |
| N2 | W2+ | 0 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 | 0 | 0.47 / 0.67 | 0.0025 | 0.648 |
| N2 | W2+ | 0.05 | 0.000 | 0.004 | 0.996 | 0 | 0 | 0 | 0 | 0.49 / 0.70 | 0.0062 | 0.207 |
| N2 | W2+ | 0.1 | 0.007 | 0.013 | 0.988 | 0 | 0 | 0 | 0 | 0.49 / 0.72 | 0.0092 | 0.027 |
| N2 | W2+ | 0.2 | 0.061 | 0.046 | 0.954 | 0 | 0 | 0 | 0 | 0.51 / 0.78 | 0.0137 | 0.004 |
| N4 | 原始 | 0 | 0.000 | 0.050 | 0.950 | 0 | 0 | 0 | 0 | 0.39 / 0.52 | 0.0022 | 0.341 |
| N4 | 原始 | 0.05 | 0.425 | 0.533 | 0.467 | 0 | 0 | 0 | 0 | 0.41 / 0.57 | 0.0068 | 0.000 |
| N4 | 原始 | 0.1 | 0.555 | 0.670 | 0.330 | 0 | 0 | 0 | 0 | 0.41 / 0.59 | 0.0100 | 0.000 |
| N4 | 原始 | 0.2 | 0.737 | 0.803 | 0.197 | 0 | 0 | 0 | 0 | 0.42 / 0.63 | 0.0148 | 0.000 |
| N4 | W2+ | 0 | 0.000 | 0.050 | 0.950 | 0 | 0 | 0 | 0 | 0.40 / 0.54 | 0.0035 | 0.317 |
| N4 | W2+ | 0.05 | 0.003 | 0.182 | 0.818 | 0 | 0 | 0 | 0 | 0.41 / 0.57 | 0.0074 | 0.013 |
| N4 | W2+ | 0.1 | 0.047 | 0.308 | 0.692 | 0 | 0 | 0 | 1 | 0.41 / 0.59 | 0.0103 | 0.002 |
| N4 | W2+ | 0.2 | 0.325 | 0.513 | 0.487 | 0 | 0 | 0 | 2 | 0.42 / 0.63 | 0.0147 | 0.000 |

**全表健康度**：`timeout=0`、`time_censored=0`、`mpc_failures=0`、`infeasible_stops=0`；
约束违反仅 3/10,560（N4 W2+ p=0.1 一次、p=0.2 两次，最大速度 1.50005 m/s，属数值边界）；
跟踪 RMSE 均值 0.0082 m（最大 0.0471 m）；单次求解 P99 最大 6.19 ms。

**W2+ 配对差**（同任务、同 p；负=W2+ 碰撞更少）：

| 组 | p | 任务数 | 平均差 | 95% 任务级 bootstrap |
|---|---:|---:|---:|---|
| N2 | 0 | 24 | 0.000 | [0.000, 0.000] |
| N2 | 0.05 | 24 | −0.081 | [−0.129, −0.039] |
| N2 | 0.1 | 24 | −0.114 | [−0.181, −0.057] |
| N2 | 0.2 | 24 | −0.099 | [−0.157, −0.047] |
| N4 | 0 | 20 | 0.000 | [−0.150, +0.150] |
| N4 | 0.05 | 20 | −0.352 | [−0.452, −0.253] |
| N4 | 0.1 | 20 | −0.362 | [−0.452, −0.265] |
| N4 | 0.2 | 20 | −0.290 | [−0.360, −0.222] |

**静态障碍检查（必须单独报告，不属于 First-Hit 目标）**：

| 组 | map_id | 运行数 | 障碍碰撞率 | 最小净空 (m) |
|---|---|---:|---:|---:|
| N2 | open_crossing_300009 | 1200 | **0.600** | −0.211 |
| N4 | open_crossing_300015 | 480 | **0.558** | −0.104 |
| N4 | single_obstacle_300004 | 480 | **1.000** | −0.106 |
| N4 | single_obstacle_300007 | 480 | **1.000** | −0.168 |
| N4 | single_obstacle_300001 | 480 | 0.000 | +0.051 |
| 其它 | — | 7920 | 0.000 | +∞ |

**这是冻结计划库的真实问题，不是碰撞检测误报**：多个任务的**计划路径本身**已经穿过
障碍物（例如 `single_obstacle_300004` 的 4 条路径净空分别为 −0.044/−0.023/−0.047/−0.038 m；
`open_crossing_300015_n4_1` 有一条路径净空 −0.022 m），执行偏差会让净空进一步恶化。
按实施任务书要求，本轮**没有删除这些任务、没有修改地图、也没有把这些障碍碰撞混入
First-Hit 风险标签**；阶段二的 pair 风险只针对机器人-机器人首次碰撞。

### 3.3 结果解读（必须先看的事实）

1. **D1 与 D2 不是同一个执行模型**。D1 的时隙是原子推进；D2 有加速/制动/停稳/重启。
   D2 的碰撞率高于 D1 是“执行侵蚀安全余量”，不是 D1 或 D2 哪个“算错了”。
2. **W2+ 的收益在 D2 下缩水**。W2+ 是在 D1 时序上优化的；D2 每个计划停车点额外引入
   制动/重启时间，相对相位被推离 D1 最优点。正式表中 W2+ 在多数格子仍优于原始计划，
   但收益远小于 D1；p 越高、N4 越明显。
3. **p=0 不再保证零碰撞**。冻结计划库的 p=0 最小间距有的只比 `d_safe=0.35 m` 高几毫米；
   D2 的连续轨迹可以把最小间距压到阈值以下。这类碰撞**如实保留**，没有通过增加等待、
   降低 `d_safe` 或删除难例来“修好”。
4. **W2+ p=0 的旧“25% 碰撞”结论失效**。旧结果的很大一部分来自执行器的制动振荡
   （参考在身后导致反向加速、二次制动）与 4 点碰撞采样，而不是计划本身；修复后
   W2+ p=0 与原始计划同量级。旧报告中的 N4 W2+ p=0 25%、N2 p=0.2 22.9% 等数字
   不可再引用。

---

## 4. 哪些旧结论失效

| 旧结论（`docs/Online_FirstHit_P0-P2_阶段报告.md`） | 本轮处置 |
|---|---|
| MPC 预测矩阵误差 1.7e-16，MPC 已修好 | 预测矩阵仍正确，但 Δu 代价、执行时序、碰撞检测、统计仍有错；旧 MPC 数值不可引用 |
| N=4 p=0 仅 1/6 碰撞、0.343 m | 基于旧执行器与 4 点采样；本轮正式集重跑后以新表为准 |
| N=4 W2+ p=0 碰撞 25% | 主要是制动振荡/采样错误；修复后不再成立 |
| M1 均值修时间更差（0.124→0.157） | 旧 M1 用了测试任务的实测时长（泄漏）且含暂停重复计入；结论作废，见阶段二 |
| M2 略优于 M0（0.135→0.119） | 旧 M2 抽样固定取 robot 0 且时长含暂停；结论作废，见阶段二 |
| 跟踪 RMSE 0.012–0.020 m | 旧值来自裁剪/旧执行器；本轮约 0.008 m |
| 固定长度任务/地图跨方法一致 | 本轮显式记录任务总数、事件样本数与区间；未再假设 |

---

## 5. 限制与遗留

- MPC 仍是**平面点质量双积分器**，不含航向/差速约束；不能称“完整移动机器人动力学验证”。
- 碰撞只做机器人间圆盘连续检测；静态障碍仅在能按 `map_id` 载入地图时评估。
- D2 的 `MPC 失败` 若出现，会走可行制动回退并计入 `d2_mpc_failures`，不会被静默删除；
  正式集的失败状态在 `per_run.csv` 的 `d2_solve_statuses` 中可追溯。
- `infeasible_stops`、超时、右删失均单独报告；不能把截断时域当完成时间。
- 两两碰撞风险相加不是全队碰撞概率；全队结论只能来自联合 MC 或并集上界。

---

## 6. 文件与复现

**修改/新增（阶段一）**：

| 文件 | 作用 |
|---|---|
| `src/fhd/control/linear_mpc.py` | Δu 代价、求解统计、缓存、安全回退 |
| `src/fhd/dynamics/trajectory_timing.py` | 到达/离开时刻、速度上限语义、单调投影 |
| `src/fhd/execution/mpc_executor.py` | 状态机、制动/爬行、事件账本、航点跨越 |
| `src/fhd/execution/mpc_simulator.py` | 连续碰撞、静态障碍、统计口径 |
| `src/fhd/geometry/continuous_collision.py` | 首次越界解析解 |
| `src/fhd/geometry/static_maps.py` | map_id→障碍物 |
| `scripts/mpc_execution_benchmark.py` | 正式基准与产物 |
| `tests/test_mpc_qp_fixes.py`、`tests/test_execution_timing.py`、`tests/test_execution_stats.py`、`tests/test_mpc_continuous_collision.py` | 定向回归 |
| `configs/online_mpc_v2.yaml` | 阶段一固定参数 |

**复现命令**：

```powershell
# 验收小样本
.\.venv\Scripts\python.exe scripts\mpc_execution_benchmark.py `
    --n2-tasks 8 --n4-tasks 8 --n-mc 6 --ps "0,0.1,0.2" `
    --out results\Online_FirstHit_v2\01_mpc_benchmark_smoke

# 正式固定集
.\.venv\Scripts\python.exe scripts\mpc_execution_benchmark.py `
    --n2-tasks 24 --n4-tasks 20 --n-mc 30 --ps "0,0.05,0.1,0.2" `
    --out results\Online_FirstHit_v2\02_mpc_benchmark

# 回归
.\.venv\Scripts\python.exe -m pytest tests\test_mpc_qp_fixes.py `
    tests\test_mpc_fixes.py tests\test_mpc_execution.py `
    tests\test_execution_timing.py tests\test_execution_stats.py `
    tests\test_mpc_continuous_collision.py -q
```

**提交**：`1fb1f2f`、`16de71b`、`8f8e55d`（短 SHA；完整 SHA 见 `git log`）。
