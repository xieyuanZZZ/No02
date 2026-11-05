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
    def future_progress(self, obs: Observation, robot_id: int, times: np.ndarray) -> np.ndarray:
        """名义短时预测：当前速度沿路径外推，并不得越过下一计划停车点。

        明确近似：不预测未来随机暂停、不用真实未来轨迹；制动/等待中的机器人
        视为停在当前进度；到下一计划等待点后保持静止（首个窗口内不预测恢复）。
        """
        rb = obs.robots[robot_id]
        s_now = rb.s
        if rb.mode not in ("NORMAL", "RESTART") or rb.speed < 0.02:
            return np.full_like(times, s_now, dtype=np.float64)
        stops = np.where(self.wait_slots[robot_id] > 1e-9)[0]
        stop_s = rb.length
        for k in stops:
            if self.s_wp[robot_id, k] > s_now + 0.01:
                stop_s = min(stop_s, float(self.s_wp[robot_id, k]))
                break
        v = min(rb.speed, 1.5)
        return np.minimum(s_now + v * times, stop_s)

    def future_positions(self, obs: Observation, times: np.ndarray) -> np.ndarray:
        """[T,N,2] 名义未来位置。"""
        out = np.zeros((len(times), len(obs.robots), 2))
        for i in range(len(obs.robots)):
            s = self.future_progress(obs, i, times)
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
