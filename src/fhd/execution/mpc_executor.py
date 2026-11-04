"""单机器人 MPC 执行器（修复版）：参考管理 + 模式机 + 等待/暂停处理。

修复要点（对应实施安排 §3.2）：
  1. **计划等待在真正停稳之后才开始计时**，不是进入 HOLDING 就开始；
  2. 制动时间单独记录，不计入等待时长，也不重复计入；
  3. 停车点必须在物理上刹得住：刹不住时标记 `infeasible_stops`，不悄悄跳过；
  4. 制动目标就是该航点的弧长位置，不允许提前经过停车点；
  5. 暂停时长从**停稳时刻**起算（不是从请求时刻起算）；
  6. 名义到达时间与实测到达时间分开记录；
  7. 连续运动下在线累计跟踪误差（即使不保存全轨迹）。

模式：
  NORMAL        正常跟踪
  BRAKE_DWELL   为计划等待而减速（目标 = 指定航点）
  BRAKE_PAUSE   收到暂停请求而减速（目标 = 当前位置）
  HOLD          已停稳，等待中（dwell 或 pause）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhd.control.linear_mpc import LinearMPC, MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, l1_polynomial_violation
from fhd.dynamics.trajectory_timing import TimeParameterizedPath

V_STOP = 0.03           # 视为停稳的速度阈值


@dataclass
class ExecRecords:
    pause_requested: list = field(default_factory=list)
    pause_accepted: list = field(default_factory=list)
    brake_start: list = field(default_factory=list)
    brake_stop: list = field(default_factory=list)
    hold_start: list = field(default_factory=list)
    resume: list = field(default_factory=list)
    dwell_done: list = field(default_factory=list)


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
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, x0: np.ndarray | None = None):
        self.x = np.zeros(4) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        self.s_ref = float(self.path.project(self.x[:2]))
        self.t = 0.0
        self.mode = "NORMAL"
        self.target_s = self.s_ref
        self.hold_until = 0.0
        self.pause_remaining = 0.0
        self._brake_start_t = 0.0
        self._hold_start_t = 0.0
        self._dwell_idx = None
        self._hold_kind = None
        self.dwell_done = np.zeros(self.path.n_wp, dtype=bool)
        self.wp_arrival = np.full(self.path.n_wp, np.nan)
        self.wp_arrival_speed = np.full(self.path.n_wp, np.nan)
        self.done = False
        self.pause_count = 0
        self.pause_requests = 0
        self.extra_brake_time = 0.0
        self.actual_wait_time = 0.0
        self.infeasible_stops = 0
        self.constraint_violations = 0
        self.err_sum_sq = 0.0
        self.err_max = 0.0
        self.n_steps = 0
        self.speed_max = 0.0
        self.accel_max = 0.0
        self.trace = {"t": [], "p": [], "v": [], "s": [], "mode": [], "a": [], "s_ref": []}
        self.mpc.reset()
        self.rec = ExecRecords()

    # ------------------------------------------------------------------ #
    def _next_dwell(self, s_act: float):
        for k in range(self.path.n_wp):
            if self.path.dwell[k] > 1e-9 and not self.dwell_done[k] \
                    and self.path.s_wp[k] >= s_act - 1e-6:
                return k
        return None

    def request_pause(self, duration_s: float):
        """外部请求暂停：只在 NORMAL 且未结束时接受。"""
        self.pause_requests += 1
        if self.done or self.mode != "NORMAL":
            return False
        self.mode = "BRAKE_PAUSE"
        self._dwell_idx = None
        self.target_s = float(self.path.project(self.x[:2]))
        self.pause_remaining = float(duration_s)
        self._brake_start_t = self.t
        self.rec.pause_requested.append((self.t, self.target_s, float(duration_s)))
        self.rec.pause_accepted.append((self.t, self.target_s))
        self.pause_count += 1
        return True

    # ------------------------------------------------------------------ #
    def _ref_window(self) -> np.ndarray:
        Np = self.mpc.Np
        ref = np.zeros((Np + 1, 4))
        if self.mode == "NORMAL":
            t_ref = float(np.interp(self.s_ref, self.path.s_grid, self.path.t_grid))
            for k in range(Np + 1):
                sk = float(np.interp(t_ref + k * self.dt, self.path.t_grid, self.path.s_grid))
                sk = min(sk, self.path.length)
                ref[k, :2] = self.path.position_at(sk)
                ref[k, 2:] = self.path.speed_at(sk) * self.path.tangent_at(sk)
        else:
            pos = self.path.position_at(self.target_s)
            ref[:, :2] = pos
            ref[:, 2:] = 0.0
        return ref

    # ------------------------------------------------------------------ #
    def step(self, pause_request: bool = False, pause_duration: float = 0.4):
        if self.done:
            return
        v_now = float(np.linalg.norm(self.x[2:]))
        s_act = float(self.path.project(self.x[:2]))

        # ---------------- 状态机 ----------------
        if self.mode == "NORMAL":
            k = self._next_dwell(s_act)
            if k is not None:
                dist = float(self.path.s_wp[k] - s_act)
                brake = DoubleIntegrator.brake_distance(v_now, self.di.p.a_brake)
                if dist <= brake + 0.02:
                    self.mode = "BRAKE_DWELL"
                    self._hold_kind = "dwell"
                    self._dwell_idx = k
                    self.target_s = float(self.path.s_wp[k])
                    self._brake_start_t = self.t
                    if dist < brake - 0.05:
                        self.infeasible_stops += 1
                    self.s_ref = min(self.s_ref, self.target_s)
            if self.mode == "NORMAL" and pause_request:
                self.request_pause(pause_duration)

        elif self.mode in ("BRAKE_DWELL", "BRAKE_PAUSE"):
            if v_now < V_STOP:
                self.extra_brake_time += self.t - self._brake_start_t
                self.rec.brake_stop.append((self.t, self.target_s))
                self.mode = "HOLD"
                self._hold_start_t = self.t
                self.rec.hold_start.append((self.t, self.target_s))
                if self._hold_kind == "dwell" and self._dwell_idx is not None:
                    self.hold_until = self.t + float(self.path.dwell[self._dwell_idx])
                else:
                    self.hold_until = self.t + float(self.pause_remaining)

        elif self.mode == "HOLD":
            if self.t >= self.hold_until:
                held = self.t - self._hold_start_t
                if self._hold_kind == "dwell":
                    self.actual_wait_time += held
                    k = self._dwell_idx
                    self.dwell_done[k] = True
                    self.rec.dwell_done.append((k, self.t, held))
                    self._dwell_idx = None
                self.rec.resume.append((self.t, self.target_s))
                self.mode = "NORMAL"

        # ---------------- 求解与积分 ----------------
        ref = self._ref_window()
        U, info = self.mpc.solve(self.x, ref)
        u = info["u"]
        if l1_polynomial_violation(u, self.di.p.a_max) > 1.0 + 1e-6:
            self.constraint_violations += 1
        n_sub = max(int(round(self.dt / self.di.p.dt)), 1)
        for _ in range(n_sub):
            self.x = self.di.step(self.x, u)
        v = self.x[2:]
        if np.abs(v).sum() > self.di.p.v_max:
            self.x[2:] = v * (self.di.p.v_max / np.abs(v).sum())

        s_act = float(self.path.project(self.x[:2]))
        if self.mode == "NORMAL":
            v_nom = self.path.speed_at(s_act)
            self.s_ref = min(self.s_ref + v_nom * self.dt, s_act + self.lookahead,
                             self.path.length)

        q = self.path.position_at(s_act)
        err = float(np.linalg.norm(q - self.x[:2]))
        self.err_sum_sq += err * err
        self.err_max = max(self.err_max, err)
        self.n_steps += 1

        for k in range(self.path.n_wp):
            if np.isnan(self.wp_arrival[k]) and \
                    np.linalg.norm(self.x[:2] - self.path.waypoints[k]) < self.arrive_tol:
                self.wp_arrival[k] = self.t
                self.wp_arrival_speed[k] = float(np.linalg.norm(self.x[2:]))

        self.trace["t"].append(self.t)
        self.trace["p"].append(self.x[:2].copy())
        self.trace["v"].append(self.x[2:].copy())
        self.trace["s"].append(s_act)
        self.trace["mode"].append(self.mode)
        self.trace["a"].append(u.copy())
        self.trace["s_ref"].append(self.s_ref)
        self.speed_max = max(self.speed_max, float(np.linalg.norm(self.x[2:])))
        self.accel_max = max(self.accel_max, float(np.linalg.norm(u)))

        self.t += self.dt
        if s_act >= self.path.length - 0.05 and float(np.linalg.norm(self.x[2:])) < 0.05:
            self.done = True

    # ------------------------------------------------------------------ #
    def segment_stats(self) -> list:
        """每段的 (序号, 长度, 实际通行时间, 进入速度, 驶出速度, 期间是否有停车)。"""
        out = []
        paused_k = [int(d[0]) for d in self.rec.dwell_done]
        for k in range(self.path.n_wp - 1):
            t0, t1 = self.wp_arrival[k], self.wp_arrival[k + 1]
            if np.isnan(t0) or np.isnan(t1):
                out.append({"seg": k, "length": float(self.path.s_wp[k + 1] - self.path.s_wp[k]),
                            "dt": None, "v_in": None, "v_out": None,
                            "paused": bool(k in paused_k)})
                continue
            out.append({"seg": k, "length": float(self.path.s_wp[k + 1] - self.path.s_wp[k]),
                        "dt": float(t1 - t0),
                        "v_in": float(self.wp_arrival_speed[k]),
                        "v_out": float(self.wp_arrival_speed[k + 1]),
                        "paused": bool(k in paused_k)})
        return out

    @property
    def tracking_rmse(self) -> float:
        return float(np.sqrt(self.err_sum_sq / max(self.n_steps, 1)))
