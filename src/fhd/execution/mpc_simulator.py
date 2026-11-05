"""多机器人 MPC 动力学执行模拟器（D2 模型，修复版）。

与理想离散模拟器（D1）的区别：
  * 机器人有速度/加速度/制动能力，暂停需要减速过程；
  * 位置连续，碰撞按**连续相对运动**判据（含子步插值）计算；
  * 暂停事件在时隙边界以概率 p 触发，暂停期间不重复触发（冷却）。

修复要点（对应实施安排 §3.3 / §3.4 / §3.5）：
  1. **每对机器人分别记录**是否碰撞、首次碰撞时刻、最小距离（N=4 共 6 对）；
  2. 即使 `record=False` 也**在线累计**跟踪误差、速度、加速度、进度；
  3. 暂停事件序列**预先按种子生成**，与调度方法无关，保证公平；
  4. 记录请求 / 接受 / 制动开始 / 停稳 / 恢复的时间。

D1 与 D2 是不同的随机执行模型，二者差异不能解释为某个调度算法更好。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhd.control.linear_mpc import MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams
from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams
from fhd.execution.mpc_executor import SingleRobotExecutor


@dataclass
class MPCSimConfig:
    dt_ctrl: float = 0.1
    dt_dyn: float = 0.05
    dt_slot: float = 0.4
    horizon_steps: int = 96
    v_max: float = 1.5
    a_max: float = 2.0
    a_brake: float = 2.0
    d_safe: float = 0.35
    pause_mode: str = "fixed"        # fixed | exponential
    pause_mean_s: float = 0.4
    pause_fixed_s: float = 0.4
    n_sub_collision: int = 4         # 每个控制步的碰撞子采样数


@dataclass
class SimResult:
    # 全队
    collision: bool
    first_collision_time: float | None
    first_collision_pair: tuple | None
    all_arrived: bool
    completion_time: float
    # 每对
    pair_collision: np.ndarray = None
    pair_first_time: np.ndarray = None
    pair_min_distance: np.ndarray = None
    pairs: list = None
    # 事件
    pause_requests: int = 0
    pause_accepted: int = 0
    pause_events: list = field(default_factory=list)
    # 时间分解
    extra_brake_time: float = 0.0
    actual_wait_time: float = 0.0
    infeasible_stops: int = 0
    # 跟踪
    tracking_rmse: float = 0.0
    tracking_max: float = 0.0
    max_speed: float = 0.0
    max_accel: float = 0.0
    mean_solve_ms: float = 0.0
    mpc_failures: int = 0
    constraint_violations: int = 0
    # 可选全轨迹
    trajectories: np.ndarray | None = None
    modes: np.ndarray | None = None
    segment_times: list | None = None          # 每台机器人每段的实测通行时间


def make_pause_events(seed: int, pause_p: float, n_robots: int, n_slots: int,
                      cfg: MPCSimConfig) -> list:
    """**预先**生成暂停事件序列（与调度方法无关，保证公平对照）。

    返回 [(slot, robot, duration_s), ...]。
    """
    if pause_p <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    dur_rng = np.random.default_rng(int(seed) + 7919)
    events = []
    for k in range(n_slots):
        for i in range(n_robots):
            if rng.random() < pause_p:
                d = (cfg.pause_fixed_s if cfg.pause_mode == "fixed"
                     else float(dur_rng.exponential(cfg.pause_mean_s)))
                events.append((k, i, d))
    return events


def simulate_mpc(paths_m: np.ndarray, wait_slots: np.ndarray, pause_p: float,
                 cfg: MPCSimConfig, seed: int = 0, record: bool = True,
                 events: list | None = None) -> SimResult:
    """paths_m: [N,H,2]（米制）；wait_slots: [N,H-1]（时隙）。"""
    n, H, _ = paths_m.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs)
    di = DoubleIntegrator(DoubleIntegratorParams(
        dt=cfg.dt_dyn, dt_ctrl=cfg.dt_ctrl, dt_slot=cfg.dt_slot,
        v_max=cfg.v_max, a_max=cfg.a_max, a_brake=cfg.a_brake))
    mpc_params = MPCParams(horizon=18, v_max=cfg.v_max, a_max=cfg.a_max)
    tp = TimingParams(v_max=cfg.v_max, a_acc=cfg.a_max, a_brake=cfg.a_brake,
                      dt_slot=cfg.dt_slot)

    wait_full = np.zeros((n, H))
    wait_full[:, :H - 1] = wait_slots
    execs = []
    for i in range(n):
        path = TimeParameterizedPath(paths_m[i], wait_full[i], tp)
        ex = SingleRobotExecutor(path, di, mpc_params, cfg.dt_ctrl)
        ex.reset(np.array([paths_m[i, 0, 0], paths_m[i, 0, 1], 0.0, 0.0]))
        execs.append(ex)

    if events is None:
        events = make_pause_events(seed, pause_p, n, cfg.horizon_steps, cfg)
    by_slot: dict[int, list] = {}
    for (k, i, d) in events:
        by_slot.setdefault(k, []).append((i, d))

    T_total = cfg.horizon_steps * cfg.dt_slot
    n_ctrl = int(np.ceil(T_total / cfg.dt_ctrl))
    traj = np.zeros((n_ctrl, n, 2)) if record else None
    mode_rec = np.zeros((n_ctrl, n), dtype=np.int8) if record else None

    pair_coll = np.zeros(n_pairs, dtype=bool)
    pair_first = np.full(n_pairs, np.nan)
    pair_min = np.full(n_pairs, np.inf)

    collision = False
    first_t, first_pair = None, None
    prev_pos = np.stack([e.x[:2] for e in execs])
    k_end = 0
    accepted_events = []

    for k in range(n_ctrl):
        t = k * cfg.dt_ctrl
        slot = int(round(t / cfg.dt_slot))
        reqs = {}
        if abs(t / cfg.dt_slot - slot) < 1e-9 and slot in by_slot:
            for (i, d) in by_slot[slot]:
                reqs[i] = d
        for i in range(n):
            if i in reqs:
                ok = execs[i].request_pause(reqs[i])
                accepted_events.append((t, i, reqs[i], bool(ok)))
            execs[i].step()

        cur_pos = np.stack([e.x[:2] for e in execs])
        if record:
            traj[k] = cur_pos
            mode_rec[k] = np.array([{"NORMAL": 0, "BRAKE_DWELL": 1, "BRAKE_PAUSE": 2,
                                     "HOLD": 3}[e.mode] for e in execs])
        for sub in range(cfg.n_sub_collision):
            a = prev_pos + (cur_pos - prev_pos) * (sub / cfg.n_sub_collision)
            ts = t - cfg.dt_ctrl + sub * cfg.dt_ctrl / cfg.n_sub_collision
            for pi, (i, j) in enumerate(pairs):
                dij = float(np.linalg.norm(a[i] - a[j]))
                if dij < pair_min[pi]:
                    pair_min[pi] = dij
                if dij < cfg.d_safe:
                    if not pair_coll[pi]:
                        pair_coll[pi] = True
                        pair_first[pi] = ts
                    if not collision:
                        collision = True
                        first_t = ts
                        first_pair = (i, j)
        prev_pos = cur_pos
        k_end = k + 1
        if all(e.done for e in execs):
            break

    if record and k_end > 0:
        traj = traj[:k_end]
        mode_rec = mode_rec[:k_end]
    comp = max(e.t for e in execs)
    return SimResult(
        collision=collision, first_collision_time=first_t, first_collision_pair=first_pair,
        all_arrived=bool(all(e.done for e in execs)), completion_time=float(comp),
        pair_collision=pair_coll, pair_first_time=pair_first,
        pair_min_distance=pair_min, pairs=pairs,
        pause_requests=int(sum(e.pause_requests for e in execs)),
        pause_accepted=int(sum(e.pause_count for e in execs)),
        pause_events=accepted_events,
        extra_brake_time=float(sum(e.extra_brake_time for e in execs)),
        actual_wait_time=float(sum(e.actual_wait_time for e in execs)),
        infeasible_stops=int(sum(e.infeasible_stops for e in execs)),
        tracking_rmse=float(np.mean([e.tracking_rmse for e in execs])),
        tracking_max=float(np.max([e.err_max for e in execs])),
        max_speed=float(np.max([e.speed_max for e in execs])),
        max_accel=float(np.max([e.accel_max for e in execs])),
        mean_solve_ms=float(np.mean([1000 * e.mpc.last_solve_time for e in execs])),
        mpc_failures=int(sum(e.mpc.fail_count for e in execs)),
        constraint_violations=int(sum(e.constraint_violations for e in execs)),
        trajectories=traj, modes=mode_rec,
        segment_times=[e.segment_stats() for e in execs])
