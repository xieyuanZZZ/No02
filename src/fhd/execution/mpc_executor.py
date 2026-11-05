"""单机器人 MPC 执行器：参考管理 + 模式机 + 等待/暂停处理（阶段一 v2）。

语义约定（全项目统一，见阶段一报告 §1.2）：
  * `dwell[k]` 表示**到达航点 k 后、离开航点 k 前**的等待时长（秒）；
    起点 k=0 的等待即"出发前等待"；
  * 计划等待：brake_start → 真正停稳 hold_start → 保持 dwell[k] → resume；
    制动时间、等待时间分别记账，不重算、不重叠；
  * 随机暂停：从**停稳时刻**起算 pause_remaining，`request_pause()` 只在
    NORMAL 接受，接受时清除旧 `_dwell_idx`、显式设置 `_hold_kind='pause'`；
  * 停车可行性：进入制动前检查剩余距离是否够刹；不够则记
    `infeasible_stops` 并跳过该等待点，绝不"越过航点当作正确执行"；
  * 航点到达：基于**实际轨迹弧长跨越**检测，一次控制步跨越多个航点时
    按跨越比例插值记录全部；制动停稳时也补记到达；
  * 进度投影：`MonotoneProgress` 在历史邻域内搜索最近点，避免折线拐角/
    平行段导致进度倒跳，并保存横向误差；
  * 统计：每个航点、每段都有 arrival/depart/move/planned_hold/random_hold/
    brake/resume/valid 账本；无法在时域内到达的段标记为右删失（dt=None）。

模式：
  NORMAL        正常跟踪
  BRAKE_DWELL   为计划等待而减速（目标 = 指定航点）
  BRAKE_PAUSE   收到暂停请求而减速（目标 = 当前制动距离处）
  HOLD          已停稳，等待中（dwell 或 pause）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhd.control.linear_mpc import LinearMPC, MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator
from fhd.dynamics.trajectory_timing import MonotoneProgress, TimeParameterizedPath

V_STOP = 0.03           # 视为停稳的速度阈值
STOP_DIST_TOL = 0.02    # 停车点位置容差（米）
CROSS_EPS = 1e-9


@dataclass
class ExecRecords:
    pause_requested: list = field(default_factory=list)
    pause_rejected: list = field(default_factory=list)
    pause_accepted: list = field(default_factory=list)
    brake_start: list = field(default_factory=list)
    brake_stop: list = field(default_factory=list)
    hold_start: list = field(default_factory=list)
    resume: list = field(default_factory=list)
    dwell_done: list = field(default_factory=list)
    infeasible: list = field(default_factory=list)
    events: list = field(default_factory=list)       # 统一事件账本

    def log(self, t: float, etype: str, **kw) -> None:
        item = {"t": float(t), "type": etype}
        item.update(kw)
        self.events.append(item)


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
        self.progress = MonotoneProgress(self.path)
        self.s_ref = float(self.progress.reset(self.x[:2]))
        self.s_act = float(self.s_ref)
        self.lateral_error_max = 0.0
        self.t = 0.0
        self.mode = "NORMAL"
        self.target_s = self.s_ref
        self.hold_until = 0.0
        self.pause_remaining = 0.0
        self._brake_start_t = 0.0
        self._brake_start_s = self.s_ref
        self._brake_start_v = 0.0
        self._hold_start_t = 0.0
        self._dwell_idx = None
        self._hold_kind = None
        self._hold_source = None
        self.last_stop_reject_reason = None
        self.pause_source_counts = {"random": 0, "online": 0}
        self.pause_reject_reasons: dict[str, int] = {}
        self.random_pause_count = 0
        self.online_pause_count = 0
        self.online_wait_time = 0.0
        self.last_online_stop_t = -1e9
        self._s_prev = self.s_ref
        self._step_index = 0
        self.dwell_done = np.zeros(self.path.n_wp, dtype=bool)
        self.dwell_skipped = np.zeros(self.path.n_wp, dtype=bool)
        self.wp_arrival = np.full(self.path.n_wp, np.nan)
        self.wp_depart = np.full(self.path.n_wp, np.nan)
        self.wp_arrival_speed = np.full(self.path.n_wp, np.nan)
        self.wp_arrival_shortfall = np.zeros(self.path.n_wp)
        self.wp_resume_speed = np.full(self.path.n_wp, np.nan)
        self.wp_brake_time = np.zeros(self.path.n_wp)
        self.wp_random_hold = np.zeros(self.path.n_wp)
        self.wp_resume_time = np.full(self.path.n_wp, np.nan)
        self.wp_valid = np.ones(self.path.n_wp, dtype=bool)
        self.done = False
        self.finish_time = None
        self.time_censored = False
        self.pause_count = 0
        self.pause_requests = 0
        self.pause_rejected = 0
        self.extra_brake_time = 0.0
        self.actual_wait_time = 0.0
        self.random_hold_time = 0.0
        self.creep_time = 0.0
        self._creep_start_t = 0.0
        self._creep_wp = None
        self.infeasible_stops = 0
        self.constraint_violations = 0
        self.solver_failures = 0
        self.err_sum_sq = 0.0
        self.err_max = 0.0
        self.n_steps = 0
        self.speed_max = 0.0
        self.accel_max = 0.0
        self.trace = {"t": [], "p": [], "v": [], "s": [], "mode": [], "a": [], "s_ref": []}
        self.sub_positions: list[np.ndarray] = []
        self.sub_times: list[float] = []
        self.mpc.reset()
        self.rec = ExecRecords()

        # 起点：t=0 即到达航点 0；若起点有等待，直接进入 HOLD
        self.wp_arrival[0] = 0.0
        self.wp_arrival_speed[0] = float(np.linalg.norm(self.x[2:]))
        if self.path.dwell[0] > 1e-9 and self.s_ref <= 1e-6:
            self.mode = "HOLD"
            self._hold_kind = "dwell"
            self._dwell_idx = 0
            self._hold_start_t = 0.0
            self.hold_until = float(self.path.dwell[0])
            self.rec.hold_start.append((0.0, 0.0))
            self.rec.log(0.0, "hold_start", kind="dwell", wp=0, s=0.0,
                         planned_hold=float(self.path.dwell[0]))
        else:
            self.wp_depart[0] = 0.0

    # ------------------------------------------------------------------ #
    def _next_dwell(self, s_act: float):
        for k in range(self.path.n_wp):
            if self.path.dwell[k] > 1e-9 and not self.dwell_done[k] \
                    and not self.dwell_skipped[k] and self.path.s_wp[k] >= s_act - 1e-6:
                return k
        return None

    # ------------------------------------------------------------------ #
    def _reject_stop(self, source: str, reason: str, s_now: float,
                     duration_s: float, **details) -> bool:
        self.pause_rejected += 1
        self.pause_reject_reasons[reason] = self.pause_reject_reasons.get(reason, 0) + 1
        self.last_stop_reject_reason = reason
        self.rec.pause_rejected.append((self.t, s_now, float(duration_s), reason))
        self.rec.log(self.t, "pause_rejected", s=s_now, duration=float(duration_s),
                     mode=self.mode, source=source, reason=reason, **details)
        return False

    def _request_stop(self, s_target: float, duration_s: float, source: str,
                      reason: str, margin: float = 0.05) -> bool:
        """统一停车命令：检查目标点是否在前方且确实刹得住。

        合同：`available = (s_target-s_act) - v²/(2a_brake) - v·dt_ctrl`
        必须 >= margin；理论刹车距离不作为闭环保证，接受后仍由仿真检验。
        """
        self.pause_requests += 1
        s_now = float(self.s_act)
        v_now = float(np.linalg.norm(self.x[2:]))
        if self.done:
            return self._reject_stop(source, "already_done", s_now, duration_s,
                                     target_s=float(s_target))
        if self.mode != "NORMAL":
            return self._reject_stop(source, f"mode_{self.mode}", s_now, duration_s,
                                     target_s=float(s_target))
        if s_target < s_now - 1e-6:
            return self._reject_stop(source, "target_behind", s_now, duration_s,
                                     target_s=float(s_target))
        if s_target > self.path.length + 1e-9:
            return self._reject_stop(source, "target_beyond_path", s_now, duration_s,
                                     target_s=float(s_target))
        brake = DoubleIntegrator.brake_distance(v_now, self.di.p.a_brake)
        lead = v_now * self.dt
        available = float(s_target) - s_now - brake - lead
        if available < margin:
            return self._reject_stop(source, "cannot_stop", s_now, duration_s,
                                     target_s=float(s_target), brake_distance=brake,
                                     lead_distance=lead, available=available)
        self.mode = "BRAKE_PAUSE"
        self._hold_kind = "pause"
        self._hold_source = source
        self._dwell_idx = None
        self._brake_start_t = self.t
        self._brake_start_s = s_now
        self._brake_start_v = v_now
        self.target_s = float(s_target)
        self.pause_remaining = float(max(duration_s, 0.0))
        self.rec.pause_requested.append((self.t, s_now, float(duration_s)))
        self.rec.pause_accepted.append((self.t, s_now, float(duration_s)))
        self._pause_wp = self._last_arrived_wp()
        self.pause_source_counts[source] = self.pause_source_counts.get(source, 0) + 1
        self.pause_count += 1
        if source == "online":
            self.online_pause_count += 1
            self.last_online_stop_t = self.t
        else:
            self.random_pause_count += 1
        self.rec.log(self.t, "pause_accepted", s=s_now, duration=float(duration_s),
                     brake_distance=brake, lead_distance=lead, target_s=float(s_target),
                     source=source, reason=reason, wp=self._pause_wp)
        return True

    def request_pause(self, duration_s: float) -> bool:
        """随机暂停请求（仿真环境专用）：按预测停车点自动取目标。"""
        s_now = float(self.s_act)
        v_now = float(np.linalg.norm(self.x[2:]))
        if self.mode == "NORMAL":
            # 已经逼近计划停车点：随机暂停与计划等待重叠，明确拒绝
            k_next = self._next_dwell(s_now)
            if k_next is not None:
                dist_next = float(self.path.s_wp[k_next] - s_now)
                if -1e-6 <= dist_next < (DoubleIntegrator.brake_distance(
                        v_now, self.di.p.a_brake) + 0.15):
                    self.pause_requests += 1
                    return self._reject_stop("random", "planned_stop_imminent",
                                             s_now, duration_s, wp=int(k_next))
        brake = DoubleIntegrator.brake_distance(v_now, self.di.p.a_brake)
        target = min(s_now + brake + v_now * self.dt + 0.05, self.path.length)
        return self._request_stop(target, duration_s, source="random",
                                  reason="random_pause", margin=0.0)

    def request_stop_at(self, s_target: float, duration_s: float,
                        source: str = "online", reason: str = "online_stop",
                        margin: float = 0.05) -> bool:
        """在线协调的指定目标停车命令（目标必须是未越过的未来位置）。"""
        return self._request_stop(float(s_target), float(duration_s), source,
                                  reason, margin=margin)

    def feasible_stop_candidates(self, lookahead_m: float = 2.0,
                                 margin: float = 0.05) -> list[dict]:
        """返回当前可安全停下的未来航点（尚未越过、非计划等待点、非终点）。"""
        if self.done or self.mode != "NORMAL":
            return []
        s_now = float(self.s_act)
        v_now = float(np.linalg.norm(self.x[2:]))
        brake = DoubleIntegrator.brake_distance(v_now, self.di.p.a_brake)
        lead = v_now * self.dt
        next_dwell = self._next_dwell(s_now)
        out = []
        for k in range(self.path.n_wp - 1):
            if self.path.dwell[k] > 1e-9:
                continue
            if next_dwell is not None and k >= int(next_dwell):
                break
            dist = float(self.path.s_wp[k] - s_now)
            if dist <= 0.01:
                continue
            if dist > lookahead_m:
                break
            available = dist - brake - lead
            if available >= margin:
                out.append({"wp": int(k), "s": float(self.path.s_wp[k]),
                            "dist": dist, "brake_distance": brake,
                            "lead_distance": lead, "available": available})
        return out

    def observation(self) -> dict:
        v = self.x[2:]
        hold_remaining = (max(self.hold_until - self.t, 0.0)
                          if self.mode == "HOLD" else 0.0)
        return {
            "x": float(self.x[0]), "y": float(self.x[1]),
            "vx": float(v[0]), "vy": float(v[1]),
            "speed": float(np.linalg.norm(v)),
            "s": float(self.s_act), "s_ref": float(self.s_ref),
            "mode": str(self.mode), "hold_remaining": float(hold_remaining),
            "pause_remaining": float(self.pause_remaining)
            if self.mode == "HOLD" else 0.0,
            "hold_kind": self._hold_kind, "hold_source": self._hold_source,
            "online_wait_time": float(self.online_wait_time),
            "last_online_stop_t": float(self.last_online_stop_t),
            "length": float(self.path.length),
            "stop_candidates": self.feasible_stop_candidates(),
        }

    def _last_arrived_wp(self) -> int:
        idx = [k for k in range(self.path.n_wp) if np.isfinite(self.wp_arrival[k])]
        return int(max(idx)) if idx else -1

    # ------------------------------------------------------------------ #
    def _normal_ref(self) -> np.ndarray:
        Np = self.mpc.Np
        ref = np.zeros((Np + 1, 4))
        t_ref = float(np.interp(self.s_ref, self.path.s_grid, self.path.t_grid))
        for k in range(Np + 1):
            sk = float(np.interp(t_ref + k * self.dt, self.path.t_grid, self.path.s_grid))
            sk = min(sk, self.path.length)
            ref[k, :2] = self.path.position_at(sk)
            ref[k, 2:] = self.path.speed_at(sk) * self.path.tangent_at(sk)
        return ref

    def _brake_ref(self) -> np.ndarray:
        """制动参考：按 v(t)=max(v0-a t,0) 前进，停在 min(预测停车点, target)。"""
        Np = self.mpc.Np
        a = max(self.di.p.a_brake, 1e-9)
        v0 = float(self._brake_start_v)
        s0 = float(self._brake_start_s)
        target = float(self.target_s)
        s_stop = min(s0 + v0 * v0 / (2.0 * a), target)
        s_stop = min(max(s_stop, s0), self.path.length)
        ref = np.zeros((Np + 1, 4))
        # 若实际位置已越过预测停车点，参考必须钉在当前位置、零速度；
        # 否则"参考在身后"会让 MPC 反向加速，造成来回振荡和二次制动。
        if self.s_act > s_stop + 1e-3:
            ref[:, :2] = self.path.position_at(self.s_act)
            return ref
        elapsed = max(self.t - self._brake_start_t, 0.0)
        for k in range(Np + 1):
            tau = elapsed + k * self.dt
            move = v0 * tau - 0.5 * a * tau * tau
            sk = min(s0 + max(move, 0.0), s_stop)
            vk = max(v0 - a * tau, 0.0)
            if sk >= s_stop - 1e-12:
                vk = 0.0
            ref[k, :2] = self.path.position_at(sk)
            ref[k, 2:] = vk * self.path.tangent_at(sk)
        return ref

    def _ref_window(self) -> np.ndarray:
        if self.mode == "NORMAL":
            return self._normal_ref()
        if self.mode in ("BRAKE_DWELL", "BRAKE_PAUSE"):
            return self._brake_ref()
        # HOLD：停在当前位置（不是 target，避免越过停车点后反向加速）
        ref = np.zeros((self.mpc.Np + 1, 4))
        ref[:, :2] = self.path.position_at(self.s_act)
        return ref

    # ------------------------------------------------------------------ #
    def _velocity_servo(self, v_des: float) -> np.ndarray:
        """把**速度向量**驱动到 v_des·tangent（而不是只看切向标量）。

        若机器人因过冲/投影误差短暂反向运动，标量式 `tangent * a_cmd`
        会在错误方向上加速；向量式 `u=(v_target-v)/dt` 永远朝目标速度收敛。
        """
        v_target = float(v_des) * self.path.tangent_at(self.s_act)
        u = (v_target - self.x[2:]) / self.dt
        u = np.clip(u, -self.di.p.a_max, self.di.p.a_max)
        return self.mpc._project(u)

    # ------------------------------------------------------------------ #
    def _record_arrivals_crossing(self, t0: float, s0: float, s1: float) -> None:
        """按弧长跨越记录航点到达（一次控制步可跨越多个航点）。"""
        if s1 <= s0 + CROSS_EPS:
            return
        for k in range(self.path.n_wp):
            if np.isnan(self.wp_arrival[k]) and s0 < self.path.s_wp[k] <= s1 + 1e-12:
                frac = (self.path.s_wp[k] - s0) / (s1 - s0)
                tt = t0 + frac * self.dt
                self.wp_arrival[k] = float(tt)
                self.wp_arrival_speed[k] = float(np.linalg.norm(self.x[2:]))
                self.rec.log(tt, "waypoint_arrival", wp=k, s=float(self.path.s_wp[k]),
                             how="crossing")

    def _enter_hold(self, t: float) -> None:
        self.extra_brake_time += t - self._brake_start_t
        self.rec.brake_stop.append((t, self.target_s, self._hold_kind))
        self.rec.log(t, "brake_stop", kind=self._hold_kind, s=self.s_act,
                     brake_time=t - self._brake_start_t)
        if self._hold_kind == "dwell" and self._dwell_idx is not None:
            k = int(self._dwell_idx)
            self.wp_brake_time[k] += t - self._brake_start_t
            shortfall = float(self.path.s_wp[k] - self.s_act)
            if shortfall > STOP_DIST_TOL:
                # 停短了：先低速爬行到精确停车点，等待时钟仍从真正到位后开始
                self.mode = "CREEP_DWELL"
                self._creep_start_t = t
                self._creep_wp = k
                self.rec.log(t, "creep_start", wp=k, s=self.s_act, shortfall=shortfall)
                return
            if np.isnan(self.wp_arrival[k]):
                self.wp_arrival[k] = t
                self.wp_arrival_speed[k] = float(np.linalg.norm(self.x[2:]))
                self.rec.log(t, "waypoint_arrival", wp=k, s=self.s_act, how="stop",
                             shortfall=abs(shortfall))
            self.wp_arrival_shortfall[k] = abs(shortfall)
            hold = float(self.path.dwell[k])
            self.hold_until = t + hold
            self.rec.log(t, "hold_start", kind="dwell", wp=k, s=self.s_act,
                         planned_hold=hold)
        else:
            self.hold_until = t + float(self.pause_remaining)
            self.rec.log(t, "hold_start", kind="pause", s=self.s_act,
                         duration=float(self.pause_remaining),
                         source=self._hold_source or "random")
        self.mode = "HOLD"
        self._hold_start_t = t
        self.rec.hold_start.append((t, self.s_act))

    # ------------------------------------------------------------------ #
    def step(self, pause_request: bool = False, pause_duration: float = 0.4):
        if self.done:
            return
        t0 = float(self.t)
        v_now = float(np.linalg.norm(self.x[2:]))
        s_act = float(self.progress.s)

        # ---------------- 1) 状态机（基于当前状态） ----------------
        if self.mode in ("BRAKE_DWELL", "BRAKE_PAUSE"):
            if v_now < V_STOP:
                self._enter_hold(t0)
        elif self.mode == "CREEP_DWELL":
            k = int(self._creep_wp)
            dist = float(self.path.s_wp[k] - s_act)
            if dist <= 0.04 and v_now < 0.06:
                self.creep_time += t0 - self._creep_start_t
                self.mode = "HOLD"
                self._hold_kind = "dwell"
                self._dwell_idx = k
                self._hold_start_t = t0
                self.hold_until = t0 + float(self.path.dwell[k])
                if np.isnan(self.wp_arrival[k]):
                    self.wp_arrival[k] = t0
                    self.wp_arrival_speed[k] = v_now
                self.wp_arrival_shortfall[k] = max(dist, 0.0)
                self.rec.hold_start.append((t0, self.s_act))
                self.rec.log(t0, "hold_start", kind="dwell", wp=k, s=self.s_act,
                             planned_hold=float(self.path.dwell[k]))
        elif self.mode == "HOLD":
            if t0 >= self.hold_until - 1e-9:
                held = t0 - self._hold_start_t
                self.rec.resume.append((t0, self.s_act, self._hold_kind))
                self.rec.log(t0, "resume", kind=self._hold_kind, s=self.s_act, held=held)
                if self._hold_kind == "dwell" and self._dwell_idx is not None:
                    k = int(self._dwell_idx)
                    self.actual_wait_time += held
                    self.dwell_done[k] = True
                    self.wp_resume_time[k] = t0
                    self.wp_resume_speed[k] = v_now
                    self.wp_depart[k] = t0
                    self.rec.dwell_done.append((k, t0, held))
                else:
                    source = self._hold_source or "random"
                    if source == "online":
                        self.online_wait_time += held
                    else:
                        self.random_hold_time += held
                        k = int(getattr(self, "_pause_wp", -1))
                        if 0 <= k < self.path.n_wp:
                            self.wp_random_hold[k] += held
                    self.rec.log(t0, "pause_hold_done", source=source, held=held)
                self._hold_kind = None
                self._hold_source = None
                self._dwell_idx = None
                self.mode = "NORMAL"
                self.s_ref = max(self.s_ref, self.s_act)

        if self.mode == "NORMAL":
            k = self._next_dwell(s_act)
            if k is not None:
                dist = float(self.path.s_wp[k] - s_act)
                brake = DoubleIntegrator.brake_distance(v_now, self.di.p.a_brake)
                # 一步预瞄：按"下一步最坏加速后的速度/距离"判断是否必须现在制动。
                # 只看 dist<=brake 会在 0.1 s 控制步下错过停车窗口，导致
                # 本可停住的等待点被误判为 infeasible_stop。
                v_next = min(v_now + self.di.p.a_max * self.dt, self.di.p.v_max)
                dist_next = max(dist - v_now * self.dt, 0.0)
                brake_next = DoubleIntegrator.brake_distance(v_next, self.di.p.a_brake)
                if dist < brake - STOP_DIST_TOL:
                    # 停不住：记不可行并跳过该等待点，不能假装停过
                    self.infeasible_stops += 1
                    self.dwell_skipped[k] = True
                    self.rec.infeasible.append((t0, k, dist, brake))
                    self.rec.log(t0, "infeasible_stop", wp=k, dist=dist, brake=brake)
                elif v_now < V_STOP and abs(dist) <= STOP_DIST_TOL:
                    self._hold_kind = "dwell"
                    self._dwell_idx = int(k)
                    self.target_s = float(self.path.s_wp[k])
                    self._brake_start_t = t0
                    self._brake_start_s = s_act
                    self._brake_start_v = v_now
                    self._enter_hold(t0)
                elif dist_next <= brake_next + STOP_DIST_TOL:
                    self.mode = "BRAKE_DWELL"
                    self._hold_kind = "dwell"
                    self._dwell_idx = int(k)
                    self.target_s = float(self.path.s_wp[k])
                    self._brake_start_t = t0
                    self._brake_start_s = s_act
                    self._brake_start_v = v_now
                    self.rec.brake_start.append((t0, "dwell", self.target_s))
                    self.rec.log(t0, "brake_start", kind="dwell", wp=int(k),
                                 dist=dist, brake_distance=brake)
            if self.mode == "NORMAL" and pause_request:
                self.request_pause(pause_duration)

        # ---------------- 2) 求解与积分 ----------------
        # 低速末端阻尼：MPC 在"参考=当前位置、速度=0"附近会以 ~0.04 m/s 长时间
        # 爬行，导致制动时长被放大到 1.5 s 以上。进入制动末段后直接用受
        # 加速度限幅的阻尼控制把速度在一拍内压到 0（不是后处理裁剪状态）。
        if self.mode == "CREEP_DWELL":
            k = int(self._creep_wp)
            dist = float(self.path.s_wp[k] - self.s_act)
            # 到 1 cm 以内才真正给零速，避免在"差 2 cm"处死锁
            v_des = 0.12 if dist > 0.03 else (0.06 if dist > 0.01 else 0.0)
            u = self._velocity_servo(v_des)
            self.mpc._last_u = np.asarray(u, dtype=np.float64)
            info = {"ok": True, "status": "creep", "u": u, "usable": True}
            U = np.tile(u, (self.mpc.Np, 1))
        elif self.mode in ("BRAKE_DWELL", "BRAKE_PAUSE"):
            # 制动段用显式减速剖面 v_des = sqrt(2 a d)，而不是让 MPC 慢慢
            # 追踪"身后/前方"的位置参考。MPC 的跟踪权重会让实际减速度
            # 只有 ~0.6 m/s²，把 0.5 s 的制动拖成 1.5 s，破坏时序。
            dist = float(self.target_s - self.s_act)
            # 扣掉"这一拍还会往前走"的距离，得到离散时间下的可行减速剖面；
            # 否则会先滑行一拍再制动，越过停车点 0.1~0.2 m。
            dist_eff = max(dist - v_now * self.dt - 0.01, 0.0)
            v_des = (min(v_now, float(np.sqrt(2.0 * self.di.p.a_brake * dist_eff)))
                     if dist > 0.01 else 0.0)
            u = self._velocity_servo(v_des)
            self.mpc._last_u = np.asarray(u, dtype=np.float64)
            info = {"ok": True, "status": "braking_profile", "u": u, "usable": True}
            U = np.tile(u, (self.mpc.Np, 1))
        else:
            ref = self._ref_window()
            U, info = self.mpc.solve(self.x, ref)
            u = info["u"]
        if not info["ok"]:
            self.solver_failures += 1
            if not info.get("usable", True):
                # 没有可行的安全动作：显式失败，不允许静默修改状态
                self.time_censored = True
                self.rec.log(t0, "unsafe_state", status=info.get("status"),
                             fallback=info.get("fallback"))
                return
        if np.abs(u).sum() > self.di.p.a_max * (1.0 + 1e-6):
            self.constraint_violations += 1

        self.sub_positions = []
        self.sub_times = []
        n_sub = max(int(round(self.dt / self.di.p.dt)), 1)
        for j in range(n_sub):
            self.x = self.di.step(self.x, u)
            self.sub_positions.append(self.x[:2].copy())
            self.sub_times.append(t0 + (j + 1) * self.di.p.dt)
        v = self.x[2:]
        # 不做后处理速度裁剪。物理速度界是欧氏 ||v||<=v_max；QP 内部用更保守的
        # L1 球，但直接制动伺服可让 L1 超过 v_max 而 ||v|| 仍低于 v_max。
        # 约束违反只按物理欧氏界计数，避免把保守多边形的松弛误报为越界。
        if float(np.linalg.norm(v)) > self.di.p.v_max * (1.0 + 1e-3):
            self.constraint_violations += 1

        # ---------------- 3) 进度与航点事件 ----------------
        s_new, lateral = self.progress.update(self.x[:2])
        self.lateral_error_max = max(self.lateral_error_max, lateral)
        self.s_act = s_new
        self._record_arrivals_crossing(t0, self._s_prev, s_new)

        q = self.path.position_at(s_new)
        err = float(np.linalg.norm(q - self.x[:2]))
        self.err_sum_sq += err * err
        self.err_max = max(self.err_max, err)
        self.n_steps += 1

        if self.mode == "NORMAL":
            v_nom = self.path.speed_at(s_new)
            self.s_ref = min(self.s_ref + v_nom * self.dt, s_new + self.lookahead,
                             self.path.length)

        # 无等待航点：离开时间 = 到达时间；被跳过/停不住的等待点也按通过处理
        for k in range(self.path.n_wp):
            if np.isnan(self.wp_depart[k]) and not np.isnan(self.wp_arrival[k]) \
                    and (self.path.dwell[k] <= 1e-9 or self.dwell_skipped[k]):
                self.wp_depart[k] = self.wp_arrival[k]
        if np.isnan(self.wp_arrival[-1]) and \
                np.linalg.norm(self.x[:2] - self.path.waypoints[-1]) < self.arrive_tol \
                and float(np.linalg.norm(v)) < self.arrive_tol:
            self.wp_arrival[-1] = t0 + self.dt
            self.wp_arrival_speed[-1] = float(np.linalg.norm(v))
            self.rec.log(t0 + self.dt, "waypoint_arrival", wp=self.path.n_wp - 1,
                         how="proximity")

        self.trace["t"].append(t0)
        self.trace["p"].append(self.x[:2].copy())
        self.trace["v"].append(self.x[2:].copy())
        self.trace["s"].append(s_new)
        self.trace["mode"].append(self.mode)
        self.trace["a"].append(u.copy())
        self.trace["s_ref"].append(self.s_ref)
        self.speed_max = max(self.speed_max, float(np.linalg.norm(v)))
        self.accel_max = max(self.accel_max, float(np.linalg.norm(u)))

        self.t += self.dt
        self._s_prev = max(self._s_prev, s_new)
        self._step_index += 1
        if s_new >= self.path.length - 0.05 and float(np.linalg.norm(v)) < 0.05:
            self.done = True
            if np.isnan(self.wp_arrival[-1]):
                self.wp_arrival[-1] = t0 + self.dt
                self.wp_arrival_speed[-1] = float(np.linalg.norm(v))
            if np.isnan(self.wp_depart[-1]):
                self.wp_depart[-1] = self.wp_arrival[-1]
            self.finish_time = float(self.wp_arrival[-1])
            self.rec.log(self.finish_time, "finish", s=s_new)

    # ------------------------------------------------------------------ #
    def mark_censored(self, horizon_s: float) -> None:
        """时域结束仍未完成：右删失，不能把截断时长当作完成时间。"""
        if self.done:
            return
        self.time_censored = True
        missing = np.isnan(self.wp_arrival)
        self.wp_valid[missing] = False
        self.rec.log(horizon_s, "censored", mode=self.mode, s=self.s_act,
                     n_unreached=int(missing.sum()))

    # ------------------------------------------------------------------ #
    def waypoint_stats(self) -> list:
        out = []
        for k in range(self.path.n_wp):
            out.append({
                "waypoint": k,
                "s": float(self.path.s_wp[k]),
                "arrival_time": _f(self.wp_arrival[k]),
                "depart_time": _f(self.wp_depart[k]),
                "arrival_speed": _f(self.wp_arrival_speed[k]),
                "arrival_shortfall_m": float(self.wp_arrival_shortfall[k]),
                "resume_time": _f(self.wp_resume_time[k]),
                "planned_hold": float(self.path.dwell[k]),
                "random_hold": float(self.wp_random_hold[k]),
                "brake_time": float(self.wp_brake_time[k]),
                "valid": bool(self.wp_valid[k] and np.isfinite(self.wp_arrival[k])),
                "hold_valid": bool(self.path.dwell[k] <= 1e-9 or self.dwell_done[k]),
                "skipped": bool(self.dwell_skipped[k]),
            })
        return out

    def segment_stats(self) -> list:
        """每段账本：move_time = departure(k) → arrival(k+1)，不含航点等待。"""
        paused_k = [int(d[0]) for d in self.rec.dwell_done if d[0] is not None]
        out = []
        for k in range(self.path.n_wp - 1):
            t0, t1 = self.wp_depart[k], self.wp_arrival[k + 1]
            valid = bool(np.isfinite(t0) and np.isfinite(t1) and t1 >= t0 - 1e-9)
            out.append({
                "seg": k,
                "length": float(self.path.s_wp[k + 1] - self.path.s_wp[k]),
                "dt": float(t1 - t0) if valid else None,
                "arrival_time": _f(t1),
                "depart_time": _f(t0),
                "planned_hold": float(self.path.dwell[k]),
                "hold_executed": bool(self.path.dwell[k] <= 1e-9 or self.dwell_done[k]),
                "random_hold": float(self.wp_random_hold[k]),
                "brake_time": float(self.wp_brake_time[k]),
                "resume_time": _f(self.wp_resume_time[k]),
                "v_in": _f(self.wp_arrival_speed[k]),
                "v_out": _f(self.wp_arrival_speed[k + 1]),
                "valid": valid,
                "censored": not valid,
                "paused": bool(k in paused_k),
            })
        return out

    @property
    def tracking_rmse(self) -> float:
        return float(np.sqrt(self.err_sum_sq / max(self.n_steps, 1)))


def _f(v) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)
