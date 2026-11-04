"""多机器人 MPC 动力学执行模拟器（D2 模型）。

与理想离散模拟器（D1）的区别：
  * 机器人有速度/加速度/制动能力，暂停需要减速过程；
  * 位置是连续的，碰撞按**连续相对运动**判据（含子步插值）计算；
  * 暂停事件在时隙边界以概率 p 触发，暂停期间不重复触发（冷却）。

D1 与 D2 是**不同的随机执行模型**，它们的碰撞率差异不能解释为某个调度算法更好。
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
    response_delay_s: float = 0.0    # 可选控制响应延迟


@dataclass
class SimResult:
    collision: bool
    first_collision_time: float | None
    first_collision_pair: tuple | None
    completion_time: float
    all_arrived: bool
    pause_count: int
    extra_brake_time: float
    actual_wait_time: float
    tracking_rmse: float
    max_speed: float
    max_accel: float
    constraint_violations: int
    mpc_failures: int
    trajectories: np.ndarray | None = None
    modes: np.ndarray | None = None


def _pair_min_distance(traj: np.ndarray, n: int) -> tuple[float, int | None, int | None]:
    """traj: [K,N,2]，返回最小间距与时隙。"""
    d = np.linalg.norm(traj[:, :, None, :] - traj[:, None, :, :], axis=-1)  # [K,N,N]
    best, bi, bj = np.inf, None, None
    for i in range(n):
        for j in range(i + 1, n):
            k = int(np.argmin(d[:, i, j]))
            if d[k, i, j] < best:
                best, bi, bj = float(d[k, i, j]), i, j
    return best, bi, bj


def simulate_mpc(paths_m: np.ndarray, wait_slots: np.ndarray, pause_p: float,
                 cfg: MPCSimConfig, seed: int = 0,
                 record: bool = True) -> SimResult:
    """paths_m: [N,H,2]（米制）；wait_slots: [N,H-1]（时隙）。"""
    rng = np.random.default_rng(int(seed))
    n, H, _ = paths_m.shape
    di = DoubleIntegrator(DoubleIntegratorParams(
        dt=cfg.dt_dyn, dt_ctrl=cfg.dt_ctrl, dt_slot=cfg.dt_slot,
        v_max=cfg.v_max, a_max=cfg.a_max, a_brake=cfg.a_brake))
    mpc_params = MPCParams(horizon=18, v_max=cfg.v_max, a_max=cfg.a_max)
    tp = TimingParams(v_max=cfg.v_max, a_acc=cfg.a_max, a_brake=cfg.a_brake,
                      dt_slot=cfg.dt_slot)

    # wait 是 [N,H-1]，最后一列对应终点，终点不需要等待；停车点取前 H-1 个航点
    wait_full = np.zeros((n, H))
    wait_full[:, :H - 1] = wait_slots
    execs = []
    for i in range(n):
        path = TimeParameterizedPath(paths_m[i], wait_full[i], tp)
        ex = SingleRobotExecutor(path, di, mpc_params, cfg.dt_ctrl)
        ex.reset(np.array([paths_m[i, 0, 0], paths_m[i, 0, 1], 0.0, 0.0]))
        execs.append(ex)

    T_total = cfg.horizon_steps * cfg.dt_slot
    n_ctrl = int(np.ceil(T_total / cfg.dt_ctrl))
    traj = np.zeros((n_ctrl, n, 2)) if record else None
    mode_rec = np.zeros((n_ctrl, n), dtype=np.int8) if record else None
    pause_flags = []

    pending = [None] * n            # 响应延迟队列
    collision = False
    first_t, first_pair = None, None
    prev_pos = np.stack([e.x[:2] for e in execs])
    k_end = 0

    for k in range(n_ctrl):
        t = k * cfg.dt_ctrl
        # 时隙边界触发暂停事件
        at_slot = abs((t / cfg.dt_slot) - round(t / cfg.dt_slot)) < 1e-9
        reqs = [False] * n
        durs = [0.0] * n
        if at_slot and pause_p > 0:
            for i in range(n):
                if rng.random() < pause_p:
                    if cfg.pause_mode == "fixed":
                        durs[i] = cfg.pause_fixed_s
                    else:
                        durs[i] = float(rng.exponential(cfg.pause_mean_s))
                    reqs[i] = True
                    pause_flags.append((t, i, durs[i]))
        for i in range(n):
            execs[i].step(pause_request=reqs[i], pause_duration=durs[i])

        cur_pos = np.stack([e.x[:2] for e in execs])
        if record:
            traj[k] = cur_pos
            mode_rec[k] = np.array([{"NORMAL": 0, "BRAKING": 1, "HOLDING": 2,
                                     "RESUMING": 3}[e.mode] for e in execs])
        # 连续碰撞检查（子步插值）
        for sub in range(4):
            a = prev_pos + (cur_pos - prev_pos) * (sub / 4.0)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(a[i] - a[j]) < cfg.d_safe:
                        if not collision:
                            collision = True
                            first_t = t + sub * cfg.dt_ctrl / 4.0
                            first_pair = (i, j)
        prev_pos = cur_pos
        k_end = k + 1
        if all(e.done for e in execs):
            break

    if record and k_end > 0:
        traj = traj[:k_end]
        mode_rec = mode_rec[:k_end]
    comp = max(e.t for e in execs)
    all_arr = all(e.done for e in execs)
    # 跟踪误差：实际位置 vs 路径最近点
    rmses = []
    for i, e in enumerate(execs):
        if traj is None:
            continue
        d = np.array([np.linalg.norm(e.path.position_at(e.path.project(traj[k2, i]))
                                     - traj[k2, i]) for k2 in range(len(traj))])
        rmses.append(float(np.sqrt(np.mean(d ** 2))))
    speeds = [float(np.linalg.norm(v)) for e in execs for v in e.log.v] or [0.0]
    accs = [float(np.linalg.norm(u)) for e in execs for u in e.log.a] or [0.0]
    return SimResult(
        collision=collision, first_collision_time=first_t, first_collision_pair=first_pair,
        completion_time=comp, all_arrived=all_arr,
        pause_count=sum(e.pause_count for e in execs),
        extra_brake_time=float(sum(e.extra_brake_time for e in execs)),
        actual_wait_time=float(sum(e.path.dwell.sum() for e in execs)),
        tracking_rmse=float(np.mean(rmses)) if rmses else 0.0,
        max_speed=float(np.max(speeds)), max_accel=float(np.max(accs)),
        constraint_violations=int(sum(e.constraint_violations for e in execs)),
        mpc_failures=int(sum(e.mpc.fail_count for e in execs)),
        trajectories=traj, modes=mode_rec)
