"""多机器人 MPC 动力学执行模拟器（D2 模型，阶段一 v2）。

与理想离散模拟器（D1）的区别：
  * 机器人有速度/加速度/制动能力，暂停需要减速过程；
  * 位置连续，碰撞按**每个动力学子步区间的连续相对运动**判据计算
    （`geometry.continuous_collision`），不是 4 点离散采样；
  * 暂停事件在时隙边界以概率 p 触发，暂停期间不重复触发（冷却）。

统计口径（阶段一 §1.4）：
  * `mean_solve_ms` 由**全程所有 MPC 调用**汇总，并给 P50/P95/P99、
    失败率、不可行次数；`record=False` 只影响轨迹是否保存，不改变统计；
  * `completion_time` 区分真实完成时间与超时截断的观测时域；
  * `pair_collision`（逐对）、`team_collision`（全队任意一对）、
    `safe_completion`（无碰撞且全部到达）、`timeout`、`infeasible_stop`、
    `constraint_violation` 分开报告；两两风险之和不是全队碰撞概率；
  * 静态障碍：能按 map_id 载入地图时，用机器人圆盘沿**实际轨迹扫掠**
    检查障碍碰撞与最小净空；载入不到时显式标 `not_evaluated`。

D1 与 D2 是不同的随机执行模型，二者差异不能解释为某个调度算法更好。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from fhd.control.linear_mpc import MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams
from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams
from fhd.execution.coordinator import build_observation
from fhd.execution.mpc_executor import SingleRobotExecutor
from fhd.geometry.continuous_collision import first_breach_alpha, relative_min_distance
from fhd.geometry.primitives import Obstacles, segment_obstacle_distance


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
    n_sub_collision: int | None = None   # 兼容旧参数；None = 连续检测（正式评测）
    robot_radius: float = 0.15
    clearance_margin: float = 0.03


@dataclass
class SimResult:
    # 全队
    collision: bool
    first_collision_time: float | None
    first_collision_pair: tuple | None
    all_arrived: bool
    completion_time: float
    time_censored: bool = False
    timeout: bool = False
    safe_completion: bool = False
    # 每对
    pair_collision: np.ndarray = None
    pair_first_time: np.ndarray = None
    pair_min_distance: np.ndarray = None
    pairs: list = None
    # 事件
    pause_requests: int = 0
    pause_accepted: int = 0
    pause_rejected: int = 0
    pause_events: list = field(default_factory=list)
    # 时间分解
    extra_brake_time: float = 0.0
    actual_wait_time: float = 0.0
    random_hold_time: float = 0.0
    infeasible_stops: int = 0
    # 跟踪
    tracking_rmse: float = 0.0
    tracking_max: float = 0.0
    lateral_error_max: float = 0.0
    max_speed: float = 0.0
    max_accel: float = 0.0
    mean_solve_ms: float = 0.0
    solve_stats: dict = field(default_factory=dict)
    mpc_failures: int = 0
    constraint_violations: int = 0
    # 航点/段账本
    waypoint_stats: list | None = None
    segment_times: list | None = None
    n_steps_executed: int = 0
    # 静态障碍
    static_obstacle_status: str = "not_evaluated"
    static_obstacle_meta: dict = field(default_factory=dict)
    obstacle_collision: bool | None = None
    min_obstacle_clearance: float | None = None
    # 在线协调
    coordinator_name: str | None = None
    coordinator_stats: dict = field(default_factory=dict)
    coordinator_log: list | None = None
    online_requests: int = 0
    online_accepted: int = 0
    online_rejected: int = 0
    online_wait_time: float = 0.0
    controller_failed: bool = False
    degraded_control: int = 0
    # 可选全轨迹
    trajectories: np.ndarray | None = None
    modes: np.ndarray | None = None
    s_trace: np.ndarray | None = None
    v_trace: np.ndarray | None = None
    robot_events: list | None = None
    event_log: list | None = None


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


def _percentiles(times: list[float]) -> dict:
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return {"n_solves": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                "p99_ms": 0.0, "max_ms": 0.0}
    return {
        "n_solves": int(t.size),
        "mean_ms": float(t.mean() * 1e3),
        "p50_ms": float(np.percentile(t, 50) * 1e3),
        "p95_ms": float(np.percentile(t, 95) * 1e3),
        "p99_ms": float(np.percentile(t, 99) * 1e3),
        "max_ms": float(t.max() * 1e3),
    }


def simulate_mpc(paths_m: np.ndarray, wait_slots: np.ndarray, pause_p: float,
                 cfg: MPCSimConfig, seed: int = 0, record: bool = True,
                 events: list | None = None,
                 obstacles: Obstacles | None = None,
                 obstacle_meta: dict | None = None,
                 coordinator=None) -> SimResult:
    """paths_m: [N,H,2]（米制）；wait_slots: [N,H-1]（时隙）。"""
    paths_m = np.asarray(paths_m, dtype=np.float64)
    wait_slots = np.asarray(wait_slots, dtype=np.float64)
    n, H, _ = paths_m.shape
    if wait_slots.shape != (n, H - 1):
        raise ValueError(f"wait_slots 形状应为 {(n, H - 1)}，实际 {wait_slots.shape}")
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
    s_rec = np.zeros((n_ctrl, n)) if record else None
    v_rec = np.zeros((n_ctrl, n)) if record else None

    pair_coll = np.zeros(n_pairs, dtype=bool)
    pair_first = np.full(n_pairs, np.nan)
    pair_min = np.full(n_pairs, np.inf)
    collision = False

    # 初始时刻 t=0 的重叠也必须检测
    pos0 = np.stack([e.x[:2] for e in execs])
    for pi, (i, j) in enumerate(pairs):
        d0 = float(np.linalg.norm(pos0[i] - pos0[j]))
        pair_min[pi] = min(pair_min[pi], d0)
        if d0 < cfg.d_safe:
            pair_coll[pi] = True
            pair_first[pi] = 0.0

    prev_pos = pos0
    k_end = 0
    accepted_events = []
    event_log: list = []
    sub_dt = float(cfg.dt_dyn)
    n_sub = max(int(round(cfg.dt_ctrl / sub_dt)), 1)
    min_clear = np.inf
    decision_times_ms = []
    coordinator_log = []
    coordinator_errors = 0
    online_accepted_n = 0
    online_rejected_n = 0

    for k in range(n_ctrl):
        t = k * cfg.dt_ctrl
        slot = int(round(t / cfg.dt_slot))
        reqs = {}
        if abs(t / cfg.dt_slot - slot) < 1e-9 and slot in by_slot:
            for (i, d) in by_slot[slot]:
                reqs[i] = d
        # 1) 外部随机暂停事件（先注入；协调器只能看到注入后的真实状态）
        for i in range(n):
            if i in reqs:
                ok = execs[i].request_pause(reqs[i])
                accepted_events.append((t, i, reqs[i], bool(ok)))
        # 2) 在线协调决策（看不到随机事件表，只拿 Observation）
        if coordinator is not None and abs(t / cfg.dt_slot - slot) < 1e-9:
            try:
                obs = build_observation(execs, t, cfg.d_safe, cfg.dt_slot,
                                        T_total, float(pause_p),
                                        {"horizon_steps": cfg.horizon_steps})
                coordinator.observe(obs)
                td = time.perf_counter()
                actions = coordinator.decide(obs)
                dt_ms = (time.perf_counter() - td) * 1e3
                decision_times_ms.append(dt_ms)
                act_log = []
                for a in actions:
                    if a.kind != "stop":
                        online_rejected_n += 1
                        act_log.append({"robot": a.robot, "kind": a.kind,
                                        "accepted": False, "reason": "unsupported_kind"})
                        continue
                    ok = execs[a.robot].request_stop_at(
                        a.s_target, a.duration_s, source="online",
                        reason=a.reason)
                    online_accepted_n += int(ok)
                    online_rejected_n += int(not ok)
                    if hasattr(coordinator, "on_result"):
                        coordinator.on_result(a, bool(ok))
                    act_log.append({
                        "robot": int(a.robot), "kind": a.kind,
                        "s_target": float(a.s_target), "wp": a.candidate_wp,
                        "duration_s": float(a.duration_s), "accepted": bool(ok),
                        "reject_reason": execs[a.robot].last_stop_reject_reason
                        if not ok else None,
                        "reason": a.reason,
                        "risk_before": a.risk_before, "risk_after": a.risk_after,
                    })
                if record:
                    coordinator_log.append({
                        "t": float(t), "runtime_ms": float(dt_ms),
                        "actions": act_log,
                        "diagnostics": dict(getattr(coordinator, "last_diagnostics", {})),
                    })
            except Exception as e:  # 协调器失败不能静默改变物理时间
                coordinator_errors += 1
                if record:
                    coordinator_log.append({"t": float(t), "error": f"{type(e).__name__}:{e}"})
        # 3) 所有机器人推进一个控制周期
        for i in range(n):
            execs[i].step()

        # 逐步收集子步位置（用于连续碰撞与静态障碍扫掠）
        cur_pos = np.stack([e.x[:2] for e in execs])
        if record:
            traj[k] = cur_pos
            mode_rec[k] = np.array([{"NORMAL": 0, "BRAKE_DWELL": 1, "BRAKE_PAUSE": 2,
                                     "HOLD": 3, "CREEP_DWELL": 4}[e.mode] for e in execs])
            s_rec[k] = [e.s_act for e in execs]
            v_rec[k] = [float(np.linalg.norm(e.x[2:])) for e in execs]
        # 每个执行器的子步位置必须同长且时间一致
        for j in range(n_sub):
            t_edge = t + (j + 1) * sub_dt
            cur_sub = np.stack([e.sub_positions[j] for e in execs])
            for pi, (i, jj) in enumerate(pairs):
                r0 = prev_pos[i] - prev_pos[jj]
                dr = (cur_sub[i] - prev_pos[i]) - (cur_sub[jj] - prev_pos[jj])
                d, _ = relative_min_distance(r0, dr)
                if d < pair_min[pi]:
                    pair_min[pi] = d
                if not pair_coll[pi]:
                    a_hit = first_breach_alpha(r0, dr, cfg.d_safe)
                    if a_hit is not None:
                        th = t_edge - sub_dt + a_hit * sub_dt
                        pair_coll[pi] = True
                        pair_first[pi] = float(th)
            if obstacles is not None:
                a = torch.tensor(prev_pos, dtype=torch.float64)
                b = torch.tensor(cur_sub, dtype=torch.float64)
                dist = segment_obstacle_distance(a, b, obstacles).numpy()
                clear = float(np.min(dist)) - (cfg.robot_radius + cfg.clearance_margin)
                if clear < min_clear:
                    min_clear = clear
            prev_pos = cur_sub

        k_end = k + 1
        if all(e.done for e in execs):
            break

    if record and k_end > 0:
        traj = traj[:k_end]
        mode_rec = mode_rec[:k_end]
        s_rec = s_rec[:k_end]
        v_rec = v_rec[:k_end]

    for e in execs:
        if not e.done:
            e.mark_censored(T_total)

    # 全队首次碰撞 = 各对首次事件中的最小值（含 t=0 初始重叠）
    collision = bool(pair_coll.any())
    first_t, first_pair = None, None
    if collision:
        finite = np.isfinite(pair_first)
        if finite.any():
            idx = int(np.argmin(np.where(finite, pair_first, np.inf)))
            first_t = float(pair_first[idx])
            first_pair = pairs[idx]

    all_done = all(e.done for e in execs)
    if all_done:
        finishes = [e.finish_time for e in execs if e.finish_time is not None]
        comp = float(max(finishes)) if finishes else float(k_end * cfg.dt_ctrl)
        censored = False
    else:
        comp = float(T_total)
        censored = True

    # 求解时间：全程所有 MPC 调用，而不是每台机器人最后一次
    all_times = []
    for e in execs:
        all_times.extend(e.mpc.stats.solve_times)
    solve_pct = _percentiles(all_times)
    status_counts: dict[str, int] = {}
    for e in execs:
        for status, count in e.mpc.stats.status_counts.items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
    solve_stats = {
        **solve_pct,
        "statuses": status_counts,
        "failed": int(sum(e.mpc.stats.n_failed for e in execs)),
        "infeasible": int(sum(e.mpc.stats.n_infeasible for e in execs)),
        "inaccurate": int(sum(e.mpc.stats.n_inaccurate for e in execs)),
        "fallback": int(sum(e.mpc.stats.n_fallback for e in execs)),
        "fallback_infeasible": int(sum(e.mpc.stats.n_fallback_infeasible for e in execs)),
        "projections": int(sum(e.mpc.stats.n_projection for e in execs)),
        "calls": int(sum(e.mpc.solve_count for e in execs)),
        "failure_rate": float(sum(e.mpc.stats.n_failed for e in execs)
                              / max(sum(e.mpc.solve_count for e in execs), 1)),
    }

    obstacle_collision = None
    min_clear_value = None
    if obstacles is not None:
        obstacle_collision = bool(min_clear < 0.0)
        min_clear_value = float(min_clear)
        status = "evaluated"
    else:
        obstacle_collision = None
        status = "not_evaluated"
    meta = dict(obstacle_meta or {})
    meta.setdefault("status", status)

    if decision_times_ms:
        arr = np.asarray(decision_times_ms, dtype=np.float64)
        coord_stats = {
            "n_decisions": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
            "over_budget": int((arr > 400.0).sum()),
            "errors": int(coordinator_errors),
        }
    else:
        coord_stats = {"n_decisions": 0, "mean_ms": 0.0, "p50_ms": 0.0,
                       "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0,
                       "over_budget": 0, "errors": int(coordinator_errors)}
    controller_failed = any(
        ev.get("type") == "unsafe_state" for e in execs for ev in e.rec.events)
    degraded_control = int(sum(e.solver_failures for e in execs))

    return SimResult(
        collision=bool(collision), first_collision_time=first_t,
        first_collision_pair=first_pair,
        all_arrived=bool(all_done), completion_time=comp,
        time_censored=bool(censored),
        timeout=bool(not all_done),
        safe_completion=bool(all_done and not collision),
        pair_collision=pair_coll, pair_first_time=pair_first,
        pair_min_distance=pair_min, pairs=pairs,
        pause_requests=int(sum(e.pause_requests for e in execs)),
        pause_accepted=int(sum(e.pause_count for e in execs)),
        pause_rejected=int(sum(e.pause_rejected for e in execs)),
        pause_events=accepted_events,
        extra_brake_time=float(sum(e.extra_brake_time for e in execs)),
        actual_wait_time=float(sum(e.actual_wait_time for e in execs)),
        random_hold_time=float(sum(e.random_hold_time for e in execs)),
        infeasible_stops=int(sum(e.infeasible_stops for e in execs)),
        tracking_rmse=float(np.mean([e.tracking_rmse for e in execs])),
        tracking_max=float(np.max([e.err_max for e in execs])),
        lateral_error_max=float(np.max([e.lateral_error_max for e in execs])),
        max_speed=float(np.max([e.speed_max for e in execs])),
        max_accel=float(np.max([e.accel_max for e in execs])),
        mean_solve_ms=float(solve_pct["mean_ms"]),
        solve_stats=solve_stats,
        mpc_failures=int(sum(e.solver_failures for e in execs)),
        constraint_violations=int(sum(e.constraint_violations for e in execs)),
        waypoint_stats=[e.waypoint_stats() for e in execs],
        segment_times=[e.segment_stats() for e in execs],
        n_steps_executed=int(k_end),
        static_obstacle_status=status,
        static_obstacle_meta=meta,
        obstacle_collision=obstacle_collision,
        min_obstacle_clearance=min_clear_value,
        trajectories=traj, modes=mode_rec, s_trace=s_rec, v_trace=v_rec,
        robot_events=[e.rec.events for e in execs] if record else None,
        event_log=[{"t": float(a[0]), "robot": int(a[1]), "requested_s": float(a[2]),
                    "accepted": bool(a[3])} for a in accepted_events] if record else None,
        coordinator_name=(getattr(coordinator, "name", None)
                          if coordinator is not None else None),
        coordinator_stats=coord_stats,
        coordinator_log=coordinator_log if record else None,
        online_requests=int(online_accepted_n + online_rejected_n),
        online_accepted=int(online_accepted_n),
        online_rejected=int(online_rejected_n),
        online_wait_time=float(sum(e.online_wait_time for e in execs)),
        controller_failed=bool(controller_failed),
        degraded_control=int(degraded_control))
