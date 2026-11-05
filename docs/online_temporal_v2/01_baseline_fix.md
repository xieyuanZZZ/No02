# 01 简单在线冲突预测与固定优先级让行：修复说明

**日期**：2026-11-05  
**分支**：`experiment/online-temporal-evaluation-v2`  
**提交**：`156a6fd`（本轮简单在线修复）  
**旧结果**：`results/Online_Temporal/03_ab_comparison/` 只读保留，不覆盖。

---

## 1. 旧实现的实际错误

旧 `SimpleConflictCoordinator`：

```python
mid = 0.5 * (pos[t_idx, i] + pos[t_idx, j])
di = np.linalg.norm(pos[t_idx, i] - mid)
dj = np.linalg.norm(pos[t_idx, j] - mid)
```

`mid` 是两点中点，`di` 与 `dj` 理论上恒相等；用它们判断“谁更远”没有信息量，
让行对象基本退化为机器人编号。旧版简单在线在 p=0 的碰撞率 0.500、p=0.2 的 0.796，
不能作为可信基线。

---

## 2. 新实现（`src/fhd/optim/online_baseline.py`）

### 2.1 冲突定位

每个 `dt_slot` 读真实状态，按名义模型外推 3 s，记录每对的：

- 首次进入 `d_safe + margin` 的时刻；
- 预测最接近时刻；
- 在该时刻**各自路径上的冲突弧长** `s_conf_i / s_conf_j`（不是几何中点）；
- 预计到达时间 `ETA_i / ETA_j = 沿路径剩余距离 / 名义速度`。

### 2.2 让行规则

- 预计**较晚**到达冲突区者让行；
- 只有 `|ETA_i-ETA_j| < 0.10 s` 时才用机器人编号作**唯一平局规则**
  （编号大者让行）；
- 首选让行方没有可行停车点时，检查另一方；双方都不可停车则记录
  `no_feasible_stop`，不发命令。

### 2.3 停车点

- 必须 `< s_conf - 0.10 m`，即留在冲突区外；
- 必须满足 `feasible_stop_candidates()` 的制动距离/控制延迟/余量；
- 选择冲突区前最靠后的可行点。

### 2.4 动作前全队前瞻

对“该机器人停在候选点 1 个时隙”再做一次名义预测：

- 当前 pair 必须消失、首次冲突时刻延后 ≥0.10 s，或最小间距增大 ≥0.02 m；
- 其它 pair 不得从“无冲突”变成“冲突”，也不得显著恶化；
- 否则拒绝该动作并尝试另一方。

**不使用 First-Hit DP**，避免简单基线变成拟研究方法。

### 2.5 计划等待恢复

`Coordinator.future_progress()` 旧逻辑到下一计划等待点就永远静止。新逻辑：

- NORMAL/RESTART：到达 `wait_slots>0` 的航点后停 `wait_slots*dt_slot`，
  等待结束后继续沿路径前进；
- HOLD：先保持 `hold_remaining`，之后以 0.6 m/s 名义速度恢复；
- BRAKE/CREEP：窗口内按静止处理（保守）；
- `extra_wait` 参数可在指定航点追加一次等待，用于动作前瞻。

### 2.6 p=0 边界

p=0 没有随机暂停，在线协调没有需要修正的随机扰动；首版明确**不做在线动作**，
避免把“接近但安全”的确定性计划改成新碰撞。这样 p=0 的 simple 方法严格退化为原始计划。

### 2.7 防抖

保留 `cooldown=1.2 s`、`hold_slots=1`、`max_online_waits=6`；
新增已处理冲突记忆 `conflict_memory_s=3.0 s`，同一预测冲突不每周期反复停车。

---

## 3. 测试

新增/更新：

- `tests/test_online_baseline.py`：ETA 让行、ID 平局、per-path 冲突弧长、
  不必要停车、计划等待恢复、候选点在冲突区外、双方不可停、p=0 不动作、确定性；
- `tests/test_online_coordinator.py`：接口/运行时/异常；
- `tests/test_online_safety.py`：死锁/超时、静态障碍分离；
- `tests/test_online_hybrid.py`：离线等待保留、在线/计划等待分开计数。

实际执行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_online_baseline.py `
    tests\test_online_coordinator.py tests\test_online_safety.py `
    tests\test_online_hybrid.py -q
```

该子集 22 passed；全量测试 **236 passed, 0 failed**（python -m pytest tests -q）。

---

## 4. 本轮未做的事

- 没有用旧 8 任务结果冒充新基线；
- 没有修改路径、MPC 参数、`d_safe`、事件表；
- 没有把简单在线调成 First-Hit 的变体；
- 没有宣称“简单在线已经有效”；修复后的性能必须在同事件种子的 8 任务复测中比较。
