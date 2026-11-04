"""单机器人 MPC 执行器：参考管理 + 模式机 + 暂停处理。

模式：
  NORMAL    正常跟踪（s_ref 按路径速度剖面推进）
  BRAKING   收到暂停请求，减速到停
  HOLDING   原地保持（计划等待或随机暂停）
  RESUMING  重新加速（本质仍走 NORMAL，只是从静止开始）

关键约定（实施安排 §五、§六）：
  * **计划到达时刻与实际到达时刻分开记录**；
  * 机器人落后时**不允许**加速追赶原绝对时间表——s_ref 只能缓慢前移，
    且被"实际进度 + 前视距离"夹住；
  * 暂停事件在**时隙边界**（0.4 s）以概率 p 触发，暂停期间不再触发（冷却）；
  * 制动过程使用 a_brake 上限，制动的额外时间单独统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhd.control.linear_mpc import LinearMPC, MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator
from fhd.dynamics.trajectory_timing import TimeParameterizedPath


@dataclass
class ExecutionLog:
    t: list = field(default_factory=list)
    p: list = field(default_factory=list)
    v: list = field(default_factory=list)
    s: list = field(default_factory=list)
    mode: list = field(default_factory=list)
    a: list = field(default_factory=list)
    s_ref: list = field(default_factory=list)

    def arrays(self):
        return {k: np.asarray(getattr(self, k)) for k in
                ("t", "p", "v", "s", "mode", "a", "s_ref")}


class SingleRobotExecutor:
    def __init__(self, path: TimeParameterizedPath, di: DoubleIntegrator,
                 mpc_params: MPCParams, dt_ctrl: float,
                 lookahead_m: float = 0.6, arrive_tol: float = 0.05):
        self.path = path
        self.di = di
        self.mpc = LinearMPC(di, mpc_params)
        self.dt = float(dt_ctrl)
        self.lookahead = float(lookahead_m)
        self.arrive_tol = float(arrive_tol)

        self.x = np.zeros(4)
        self.s_ref = 0.0
        self.t = 0.0
        self.mode = "NORMAL"
        self.hold_until = 0.0
        self.brake_time = 0.0
        self.dwell_done = np.zeros(path.n_wp, dtype=bool)
        self.wp_arrival = np.full(path.n_wp, np.nan)
        self.log = ExecutionLog()
        self.done = False
        self.pause_count = 0
        self.extra_brake_time = 0.0
        self.extra_accel_time = 0.0
        self._brake_start_t = 0.0
        self._accel_start_t = 0.0
        self.constraint_violations = 0

    # ------------------------------------------------------------------ #
    def reset(self, x0: np.ndarray | None = None):
        self.x = np.zeros(4) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        self.s_ref = 0.0
        self.t = 0.0
        self.mode = "NORMAL"
        self.hold_until = 0.0
        self.dwell_done[:] = False
        self.wp_arrival[:] = np.nan
        self.log = ExecutionLog()
        self.done = False
        self.pause_count = 0
        self.extra_brake_time = 0.0
        self.extra_accel_time = 0.0
        self.constraint_violations = 0
        self.mpc.reset()

    # ------------------------------------------------------------------ #
    def request_pause(self, duration_s: float):
        """外部（随机事件）请求暂停。已停或临近终点则忽略。"""
        if self.done or self.mode in ("BRAKING", "HOLDING"):
            return False
        self.mode = "BRAKING"
        self._brake_start_t = self.t
        self.hold_until = self.t + duration_s        # 从此刻起算（含制动时间）
        self.pause_count += 1
        return True

    # ------------------------------------------------------------------ #
    def _ref_window(self):
        """生成 MPC 参考：位置沿路径，速度按剖面。"""
        Np = self.mpc.Np
        s = self.s_ref
        ref = np.zeros((Np + 1, 4))
        # 用时间表推参考：s_k = s(t_ref + k*dt)，这样起点能从静止正确加速
        t_ref = float(np.interp(s, self.path.s_grid, self.path.t_grid))
        for k in range(Np + 1):
            sk = float(np.interp(t_ref + k * self.dt, self.path.t_grid, self.path.s_grid))
            sk = min(sk, self.path.length)
            pos = self.path.position_at(sk)
            spd = self.path.speed_at(sk) if self.mode == "NORMAL" else 0.0
            tan = self.path.tangent_at(sk)
            ref[k, :2] = pos
            ref[k, 2:] = spd * tan
        if self.mode != "NORMAL":
            # 制动/保持：参考点固定在当前参考位置，速度 0
            pos = self.path.position_at(self.s_ref)
            ref[:, :2] = pos
            ref[:, 2:] = 0.0
        return ref

    # ------------------------------------------------------------------ #
    def step(self, pause_request: bool = False, pause_duration: float = 0.4):
        if self.done:
            return
        # 计划等待：到达停车点时进入 HOLDING
        if self.mode == "NORMAL":
            for k in range(self.path.n_wp):
                if (self.path.dwell[k] > 1e-9 and not self.dwell_done[k]
                        and self.s_ref >= self.path.s_wp[k] - 1e-6
                        and np.linalg.norm(self.x[:2] - self.path.waypoints[k]) < 0.25):
                    self.dwell_done[k] = True
                    self.mode = "HOLDING"
                    self.hold_until = self.t + float(self.path.dwell[k])
                    break
        if pause_request:
            self.request_pause(pause_duration)

        if self.mode == "BRAKING":
            if float(np.linalg.norm(self.x[2:])) < 0.03:
                self.mode = "HOLDING"
                self.extra_brake_time += self.t - self._brake_start_t
        elif self.mode == "HOLDING":
            if self.t >= self.hold_until:
                self.mode = "NORMAL"
                self._accel_start_t = self.t
        elif self.mode == "NORMAL":
            if self._accel_start_t > 0 and float(np.linalg.norm(self.x[2:])) < 0.1:
                pass

        ref = self._ref_window()
        U, info = self.mpc.solve(self.x, ref)
        u = info["u"]
        # 约束检查
        from fhd.dynamics.double_integrator import l1_polynomial_violation
        if l1_polynomial_violation(u, self.di.p.a_max) > 1.0 + 1e-6:
            self.constraint_violations += 1

        # 子步积分（dt_ctrl / dt_dyn）
        n_sub = max(int(round(self.dt / self.di.p.dt)), 1)
        for _ in range(n_sub):
            self.x = self.di.step(self.x, u)
        # 数值兜底：速度不可超过保守多边形
        v = self.x[2:]
        if np.abs(v).sum() > self.di.p.v_max:
            v = v * (self.di.p.v_max / np.abs(v).sum())
            self.x[2:] = v

        s_act = self.path.project(self.x[:2])
        if self.mode == "NORMAL":
            v_nom = self.path.speed_at(s_act)
            self.s_ref = min(self.s_ref + v_nom * self.dt, s_act + self.lookahead,
                             self.path.length)

        for k in range(self.path.n_wp):
            if np.isnan(self.wp_arrival[k]) and \
                    np.linalg.norm(self.x[:2] - self.path.waypoints[k]) < self.arrive_tol:
                self.wp_arrival[k] = self.t

        self.log.t.append(self.t)
        self.log.p.append(self.x[:2].copy())
        self.log.v.append(self.x[2:].copy())
        self.log.s.append(s_act)
        self.log.mode.append(self.mode)
        self.log.a.append(u.copy())
        self.log.s_ref.append(self.s_ref)

        self.t += self.dt
        if s_act >= self.path.length - 0.05 and np.linalg.norm(self.x[2:]) < 0.05:
            self.done = True
