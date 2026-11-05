"""在线时序协调的统一观测/动作接口（A/B/C 共用）。

固定语义：
  * 集中式：每个 `dt_slot` 读取所有机器人的实际 `[x,y,vx,vy]`、弧长进度、模式、
    已有等待；首版不研究通信延迟。
  * 不知道未来：Coordinator 只能拿到 `Observation`，看不到随机暂停事件表、
    真实未来轨迹或测试 seed 派生信息。
  * 只允许“在未越过的可行停车点增加等待”；首版不改变几何、不瞬时改速度、
    不修改外部事件。恢复由执行器在等待时长结束后自动完成，避免状态机竞争。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobotObservation:
    robot_id: int
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    s: float
    mode: str
    hold_remaining: float
    hold_source: str | None
    online_wait_time: float
    last_online_stop_t: float
    length: float
    stop_candidates: list[dict] = field(default_factory=list)


@dataclass
class Observation:
    t: float
    robots: list[RobotObservation]
    d_safe: float
    dt_slot: float
    horizon_s: float
    pause_p: float
    config: dict = field(default_factory=dict)


@dataclass
class Action:
    robot: int
    kind: str = "stop"             # 首版只有 stop；resume 由执行器自动完成
    s_target: float | None = None
    duration_s: float = 0.4
    candidate_wp: int | None = None
    reason: str = ""
    risk_before: float | None = None
    risk_after: float | None = None
    extra_time_s: float | None = None


class Coordinator:
    """所有在线方法的基类；子类只需实现 `decide()`。"""

    name = "base"

    def __init__(self, paths: np.ndarray, wait_slots: np.ndarray,
                 d_safe: float, dt_slot: float, horizon_s: float,
                 pause_p: float, config: dict | None = None):
        self.paths = np.asarray(paths, dtype=np.float64)
        self.wait_slots = np.asarray(wait_slots, dtype=np.float64)
        self.d_safe = float(d_safe)
        self.dt_slot = float(dt_slot)
        self.horizon_s = float(horizon_s)
        self.pause_p = float(pause_p)
        self.config = dict(config or {})
        self.last_observation: Observation | None = None
        self.last_decisions: list[Action] = []
        self.last_diagnostics: dict = {}
        seg = np.linalg.norm(np.diff(self.paths, axis=1), axis=-1)
        self.s_wp = np.concatenate([np.zeros((self.paths.shape[0], 1)),
                                    np.cumsum(seg, axis=1)], axis=1)

    def observe(self, obs: Observation) -> None:
        self.last_observation = obs

    def decide(self, obs: Observation) -> list[Action]:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def future_progress(self, obs: Observation, robot_id: int, times: np.ndarray,
                        extra_wait: dict | None = None) -> np.ndarray:
        """名义短时预测：当前速度沿路径外推，**按已知计划等待恢复**。

        语义（v2 修正）：
          * NORMAL/RESTART：按当前速度走，遇到 `wait_slots>0` 的航点停车
            `wait_slots*dt_slot`，等待结束后继续沿路径走；
          * HOLD：先保持 `hold_remaining`，之后以 0.6 m/s 的名义速度恢复；
          * BRAKE/CREEP：窗口内按静止处理（保守，不预测制动后的恢复）；
          * `extra_wait={robot:(waypoint, slots)}` 用于动作前瞻：在指定航点
            追加一次等待，然后继续按计划走。
        不读取未来随机暂停事件或真实轨迹。
        """
        rb = obs.robots[robot_id]
        times = np.asarray(times, dtype=np.float64)
        s_now = float(rb.s)
        out = np.full(times.shape, s_now, dtype=np.float64)
        if rb.mode in ("BRAKE_DWELL", "BRAKE_PAUSE", "CREEP_DWELL"):
            return out

        t_start = 0.0
        if rb.mode == "HOLD":
            t_start = max(float(rb.hold_remaining), 0.0)
            v = 0.6
            out[times <= t_start] = s_now
        else:
            v = min(float(rb.speed), 1.5)
            if v < 0.02:
                return out

        extra = (extra_wait or {}).get(robot_id)
        stops = []
        for k in range(self.paths.shape[1] - 1):
            if self.s_wp[robot_id, k] <= s_now + 0.01:
                continue
            hold = 0.0
            if self.wait_slots[robot_id, k] > 1e-9:
                hold = float(self.wait_slots[robot_id, k]) * self.dt_slot
            if extra is not None and int(extra[0]) == k:
                hold += float(extra[1]) * self.dt_slot
            if hold > 1e-9:
                stops.append((float(self.s_wp[robot_id, k]), hold))
        stops.append((float(rb.length), 0.0))
        stops.sort(key=lambda x: x[0])

        s = s_now
        cursor = t_start
        for stop_s, hold in stops:
            if stop_s <= s + 1e-9:
                continue
            travel = (stop_s - s) / max(v, 1e-9)
            t_arr = cursor + travel
            mask = (times >= cursor) & (times < t_arr)
            out[mask] = s + v * (times[mask] - cursor)
            mask = (times >= t_arr) & (times < t_arr + hold)
            out[mask] = stop_s
            cursor = t_arr + hold
            s = stop_s
            if cursor > times[-1] + 1e-9:
                break
        out[times >= cursor] = s
        return np.clip(out, 0.0, rb.length)

    def future_positions(self, obs: Observation, times: np.ndarray,
                         extra_wait: dict | None = None) -> np.ndarray:
        """[T,N,2] 名义未来位置（可带一台机器人的额外等待做动作前瞻）。"""
        out = np.zeros((len(times), len(obs.robots), 2))
        for i in range(len(obs.robots)):
            s = self.future_progress(obs, i, times, extra_wait)
            for k, sk in enumerate(s):
                out[k, i] = self.s_to_position(self.paths[i], sk)
        return out

    @staticmethod
    def s_to_position(path: np.ndarray, s: float) -> np.ndarray:
        """按折线弧长取位置（与执行器同一路径几何）。"""
        s = float(np.clip(s, 0.0, _polyline_length(path)))
        seg = np.linalg.norm(np.diff(path, axis=0), axis=-1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
        L = max(seg[k], 1e-12)
        frac = (s - cum[k]) / L
        return path[k] + frac * (path[k + 1] - path[k])


def _polyline_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum())


def build_observation(execs: list, t: float, d_safe: float, dt_slot: float,
                      horizon_s: float, pause_p: float,
                      config: dict | None = None) -> Observation:
    robots = []
    for i, ex in enumerate(execs):
        st = ex.observation()
        robots.append(RobotObservation(
            robot_id=i, x=st["x"], y=st["y"], vx=st["vx"], vy=st["vy"],
            speed=st["speed"], s=st["s"], mode=st["mode"],
            hold_remaining=st["hold_remaining"], hold_source=st["hold_source"],
            online_wait_time=st["online_wait_time"],
            last_online_stop_t=st["last_online_stop_t"], length=st["length"],
            stop_candidates=st["stop_candidates"]))
    return Observation(t=float(t), robots=robots, d_safe=float(d_safe),
                       dt_slot=float(dt_slot), horizon_s=float(horizon_s),
                       pause_p=float(pause_p), config=dict(config or {}))
