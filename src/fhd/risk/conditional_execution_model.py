"""M3：状态相关的轻量执行模型（阶段二 §2.3）。

状态 z_t = (s_t, v_t, μ_t)，μ ∈ {NORMAL, BRAKE, HOLD, RESTART}：
  * `s_t` 沿路径的真实弧长；`v_t` 沿路径速度；μ 由显式状态机推进；
  * NORMAL：按计划速度剖面加速/巡航，接近**计划停车点**时切入 BRAKE；
  * BRAKE：按 v_des = sqrt(2 a_brake d) 减速，停稳后进入 HOLD；
  * HOLD：计划等待或随机暂停时长；结束后 RESTART；
  * RESTART：用较保守的再加速度，之后回到 NORMAL；
  * 随机暂停在 `dt_slot` 边界按 p **显式生成**，只对 NORMAL 生效，
    且靠近计划停车点时拒绝（与阶段一执行器一致）。

参数不是拍脑袋：`fit_execution_model()` 从**校准集**的 p=0 / p>0 MPC
轨迹里估计 a_acc / a_brake / v_scale / restart / 暂停时长分布。
模型只读取静态规划与 p，向前滚动自己预测的状态，**不读取测试轨迹**。

注意：这是模型预测/采样估计，不是对真实 MPC 风险的严格安全保证；
M0 的 hard+hard DP 在其理想转移模型下才是精确概率。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-9
V_STOP = 0.03


def _clip_quantile(values, q: float, lo: float, hi: float, default: float) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size < 3:
        return float(default)
    return float(np.clip(np.quantile(arr, q), lo, hi))


@dataclass
class ExecutionModelParams:
    a_acc: float = 1.8
    a_brake: float = 2.0
    a_restart: float = 1.5
    restart_time: float = 0.3
    v_scale: float = 0.9
    v_max: float = 1.5
    pause_durations: list = field(default_factory=lambda: [0.4])
    dt: float = 0.1
    dt_slot: float = 0.4
    n_calib_runs: int = 0
    calib_notes: dict = field(default_factory=dict)

    @property
    def slot_steps(self) -> int:
        return max(int(round(self.dt_slot / self.dt)), 1)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["pause_durations"] = [float(x) for x in self.pause_durations]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionModelParams":
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExecutionModelParams":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# 从 MPC 轨迹校准参数
# --------------------------------------------------------------------------- #
def fit_execution_model(traces: list[dict], events: list[list[dict]],
                        dt: float = 0.1, dt_slot: float = 0.4,
                        v_max: float = 1.5) -> ExecutionModelParams:
    """traces: 每项 {s:[T,N], v:[T,N], v_plan:[T,N], modes:[T,N]}。

    events: 与 traces 对应的每台机器人事件账本（list of lists）。
    """
    accels, brakes, v_ratios, restarts, restart_accels = [], [], [], [], []
    pause_holds = []
    n_runs = 0
    for tr, evs in zip(traces, events):
        s = np.asarray(tr["s"], dtype=np.float64)
        v = np.asarray(tr["v"], dtype=np.float64)
        vp = np.asarray(tr["v_plan"], dtype=np.float64)
        modes = np.asarray(tr["modes"], dtype=np.int64)
        tv = int(tr.get("T_valid", s.shape[0]))
        s, v, vp, modes = s[:tv], v[:tv], vp[:tv], modes[:tv]
        if s.ndim != 2 or s.shape[0] < 3:
            continue
        n_runs += 1
        dv = np.diff(v, axis=0) / dt
        # 执行器原始模式：0 NORMAL, 1 BRAKE_DWELL, 2 BRAKE_PAUSE, 3 HOLD, 4 CREEP
        m0 = modes[:-1]
        moving = m0 == 0
        accels.extend(dv[moving & (dv > 0.05)].tolist())
        braking = (m0 == 1) | (m0 == 2) | (m0 == 4)
        brakes.extend((-dv[braking & (dv < -0.05)]).tolist())
        ratio = v / np.maximum(vp, 0.05)
        ok = (m0 == 0) & (vp[:-1] > 0.1)
        v_ratios.extend(ratio[:-1][ok].tolist())
        # 事件：暂停时长、重启时间/加速度
        for i, ev in enumerate(evs):
            holds = {}
            for e in ev:
                if e.get("type") == "hold_start":
                    holds[(e.get("kind"), round(float(e["t"]), 6))] = e
                elif e.get("type") == "resume":
                    t_res = float(e["t"])
                    held = float(e.get("held", 0.0))
                    if e.get("kind") == "pause":
                        pause_holds.append(held)
                    # 重启：resume 后到 v 达到计划速度 80% 的时间
                    t0 = int(round(t_res / dt))
                    if 0 <= t0 < s.shape[0] - 1:
                        tail = v[t0:, i]
                        ref = np.maximum(vp[t0:, i], 0.2)
                        reach = np.where(tail >= 0.8 * ref)[0]
                        if reach.size:
                            dur = reach[0] * dt
                            restarts.append(dur)
                            seg = dv[t0:min(t0 + reach[0] + 1, dv.shape[0]), i]
                            pos = seg[seg > 0.05]
                            if pos.size:
                                restart_accels.append(float(pos.max()))
    params = ExecutionModelParams(dt=dt, dt_slot=dt_slot, v_max=v_max)
    params.n_calib_runs = n_runs
    params.a_acc = _clip_quantile(accels, 0.95, 0.3, 2.4, 1.8)
    params.a_brake = _clip_quantile(brakes, 0.95, 0.5, 2.4, 2.0)
    params.a_restart = _clip_quantile(restart_accels, 0.6, 0.2, params.a_acc, 0.8 * params.a_acc)
    params.restart_time = _clip_quantile(restarts, 0.6, 0.0, 1.5, 0.3)
    # 实测速度可以超过计划速度上限（MPC 为追位置误差会超调）；旧实现把
    # ratio 上限截到 1.0，使 M3 系统性偏慢。用 0.9 分位并允许到 1.5。
    params.v_scale = _clip_quantile(v_ratios, 0.9, 0.3, 1.5, 0.9)
    if pause_holds:
        params.pause_durations = [float(x) for x in np.asarray(pause_holds)]
    else:
        params.pause_durations = [0.4]
    params.calib_notes = {
        "n_accel_samples": len(accels), "n_brake_samples": len(brakes),
        "n_v_ratio_samples": len(v_ratios), "n_pause_samples": len(pause_holds),
        "n_restart_samples": len(restarts),
        "pause_mean_s": float(np.mean(params.pause_durations)),
    }
    return params


# --------------------------------------------------------------------------- #
# 路径几何工具（全部按机器人循环、对 K 个样本向量化）
# --------------------------------------------------------------------------- #
class PathGeometry:
    def __init__(self, paths: np.ndarray, wait_slots: np.ndarray, dt_slot: float = 0.4,
                 v_max: float = 1.5):
        from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams

        self.paths = np.asarray(paths, dtype=np.float64)
        self.n, self.h, _ = self.paths.shape
        self.wait = np.asarray(wait_slots, dtype=np.float64)
        self.s_wp = np.zeros((self.n, self.h))
        self.length = np.zeros(self.n)
        self.dwell = np.zeros((self.n, self.h))
        self.s_grid, self.v_grid = [], []
        for i in range(self.n):
            w_full = np.zeros(self.h)
            w_full[:self.h - 1] = self.wait[i]
            tp = TimeParameterizedPath(self.paths[i], w_full,
                                       TimingParams(v_max=v_max, dt_slot=dt_slot))
            self.s_wp[i] = tp.s_wp
            self.length[i] = tp.length
            self.dwell[i] = tp.dwell
            self.s_grid.append(tp.s_grid)
            self.v_grid.append(tp.v_grid)

    def positions(self, s: np.ndarray) -> np.ndarray:
        """s [K,N] -> [K,N,2]。"""
        k = s.shape[0]
        out = np.zeros((k, self.n, 2))
        for i in range(self.n):
            si = np.clip(s[:, i], 0.0, self.length[i])
            idx = np.clip(np.searchsorted(self.s_wp[i], si, side="right") - 1,
                          0, self.h - 2)
            seg = np.maximum(self.s_wp[i, idx + 1] - self.s_wp[i, idx], EPS)
            frac = (si - self.s_wp[i, idx]) / seg
            out[:, i] = (self.paths[i, idx] +
                         frac[:, None] * (self.paths[i, idx + 1] - self.paths[i, idx]))
        return out

    def planned_speed(self, s: np.ndarray) -> np.ndarray:
        out = np.zeros_like(s)
        for i in range(self.n):
            out[:, i] = np.interp(np.clip(s[:, i], 0.0, self.length[i]),
                                  self.s_grid[i], self.v_grid[i])
        return out

    def next_stop(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (target_s [K,N], stop_idx [K,N])；无后续停车点则 target=length。"""
        k = s.shape[0]
        target = np.tile(self.length[None, :], (k, 1))
        idx_out = np.full((k, self.n), -1, dtype=np.int64)
        for i in range(self.n):
            stops = np.where(self.dwell[i] > 1e-9)[0]
            for j in stops:
                # 必须"明显在前方"才算下一个停车点；否则刚起步时 s 仍等于
                # 上一个等待航点，会立刻再次刹车、无限重复同一次等待。
                mask = (idx_out[:, i] < 0) & (s[:, i] + 0.01 < self.s_wp[i, j])
                idx_out[mask, i] = j
            has = idx_out[:, i] >= 0
            target[:, i] = np.where(has, self.s_wp[i, np.maximum(idx_out[:, i], 0)],
                                    self.length[i])
        return target, idx_out


# --------------------------------------------------------------------------- #
# 采样滚动 + 首次碰撞
# --------------------------------------------------------------------------- #
def _relative_first_hit(positions: np.ndarray, times: np.ndarray,
                        pair: tuple[int, int], d_safe: float) -> tuple[np.ndarray, np.ndarray]:
    """positions [T+1,K,N,2] -> (first_hit [K], min_distance [K])。"""
    i, j = pair
    # positions [T+1,K,N,2] -> [K,T+1,2]，时间维放在最后便于按样本向量化
    pi = np.transpose(positions[:, :, i, :], (1, 0, 2))
    pj = np.transpose(positions[:, :, j, :], (1, 0, 2))
    r0 = pi[:, :-1] - pj[:, :-1]
    dr = (pi[:, 1:] - pi[:, :-1]) - (pj[:, 1:] - pj[:, :-1])
    a = (dr * dr).sum(-1)
    b = 2.0 * (r0 * dr).sum(-1)
    c0 = (r0 * r0).sum(-1) - d_safe ** 2
    # 最小距离
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_min = np.where(a > EPS, np.clip(-b / (2.0 * a), 0.0, 1.0), 0.0)
    dmin = np.sqrt(np.maximum(((r0 + alpha_min[..., None] * dr) ** 2).sum(-1), 0.0))
    min_dist = dmin.min(axis=1)
    # 首次越界：起点已在内 -> 0；否则最小正根
    disc = b * b - 4.0 * a * c0
    with np.errstate(divide="ignore", invalid="ignore"):
        root = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * np.maximum(a, EPS))
    valid = (a > EPS) & (disc > 0.0) & (root >= 0.0) & (root <= 1.0)
    valid &= (c0 >= 0.0)
    inside = c0 < 0.0
    t0 = times[:-1][None, :]
    dt = (times[1:] - times[:-1])[None, :]
    hit_time = np.where(inside, t0, np.where(valid, t0 + root * dt, np.inf))
    first = hit_time.min(axis=1)
    first[~np.isfinite(first)] = np.inf
    return first, min_dist


def compute_pair_first_hits(positions: np.ndarray, times: np.ndarray,
                            pairs: list, d_safe: float) -> dict:
    """返回每对的样本级首次碰撞时间/最小距离，以及团队首次事件。"""
    n_samples = positions.shape[1]
    pair_first = np.full((n_samples, len(pairs)), np.inf)
    pair_min = np.full((n_samples, len(pairs)), np.inf)
    for k, pair in enumerate(pairs):
        f, d = _relative_first_hit(positions, times, pair, d_safe)
        pair_first[:, k] = f
        pair_min[:, k] = d
    team_first = pair_first.min(axis=1) if len(pairs) else np.full(n_samples, np.inf)
    team_hit = np.isfinite(team_first)
    return {"pair_first": pair_first, "pair_min": pair_min,
            "team_first": team_first, "team_hit": team_hit}


class ConditionalExecutionModel:
    def __init__(self, params: ExecutionModelParams):
        self.p = params

    # ------------------------------------------------------------------ #
    def rollout(self, paths: np.ndarray, wait_slots: np.ndarray, pause_p: float,
                horizon: float, n_samples: int = 200, seed: int = 0,
                use_braking: bool = True, use_modes: bool = True,
                use_speed: bool = True) -> dict:
        rng = np.random.default_rng(int(seed))
        geo = PathGeometry(paths, wait_slots, self.p.dt_slot, self.p.v_max)
        n, h = geo.n, geo.h
        T = max(int(np.ceil(horizon / self.p.dt)), 1)
        dt = self.p.dt
        k = int(n_samples)
        s = np.zeros((k, n))
        v = np.zeros((k, n))
        mode = np.zeros((k, n), dtype=np.int8)      # 0 NORMAL 1 BRAKE 2 HOLD 3 RESTART
        hold_left = np.zeros((k, n))
        target = np.tile(geo.length[None, :], (k, 1))
        hold_kind = np.zeros((k, n), dtype=np.int8)  # 0 dwell, 1 pause
        restart_left = np.zeros((k, n))
        pause_pool = np.asarray(self.p.pause_durations, dtype=np.float64)
        if pause_pool.size == 0:
            pause_pool = np.array([0.4])
        # 起点等待
        for i in range(n):
            if geo.dwell[i, 0] > 1e-9 and use_modes:
                mode[:, i] = 2
                hold_kind[:, i] = 0
                hold_left[:, i] = geo.dwell[i, 0]
        times = np.arange(T + 1) * dt
        positions = np.zeros((T + 1, k, n, 2))
        s_hist = np.zeros((T + 1, k, n))
        v_hist = np.zeros((T + 1, k, n))
        mode_hist = np.zeros((T + 1, k, n), dtype=np.int8)
        hold_hist = np.zeros((T + 1, k, n))
        positions[0] = geo.positions(s)
        s_hist[0], v_hist[0] = s, v
        mode_hist[0] = mode
        hold_hist[0] = hold_left
        pause_starts = 0
        braking_on = bool(use_braking)

        for t in range(T):
            if use_modes and pause_p > 0 and (t % self.p.slot_steps == 0) and t > 0:
                draw = rng.random((k, n)) < float(pause_p)
                next_target, _ = geo.next_stop(s)
                for i in range(n):
                    dist = next_target[:, i] - s[:, i]
                    near = dist < (v[:, i] ** 2 / (2.0 * max(self.p.a_brake, 1e-9))
                                   + 0.15)
                    start = draw[:, i] & ((mode[:, i] == 0) | (mode[:, i] == 3)) & ~near
                    if not start.any():
                        continue
                    brake_d = v[start, i] ** 2 / (2.0 * max(self.p.a_brake, 1e-9))
                    target[start, i] = s[start, i] + brake_d
                    mode[start, i] = 1
                    hold_kind[start, i] = 1
                    hold_left[start, i] = pause_pool[rng.integers(0, pause_pool.size,
                                                                  size=int(start.sum()))]
                    pause_starts += int(start.sum())

            for i in range(n):
                # NORMAL / RESTART
                moving = (mode[:, i] == 0) | (mode[:, i] == 3)
                if moving.any():
                    v_plan = (self.p.v_scale * geo.planned_speed(s)[:, i]
                              if use_speed else np.full(k, self.p.v_max))
                    a_cap = np.where(restart_left[:, i] > 0, self.p.a_restart, self.p.a_acc)
                    if use_modes:
                        tgt, sidx = geo.next_stop(s)
                        dist = tgt[:, i] - s[:, i]
                        trigger = dist <= (v[:, i] ** 2 / (2.0 * max(self.p.a_brake, 1e-9))
                                           + v[:, i] * dt + 0.01)
                        start = moving & trigger & (sidx[:, i] >= 0)
                        if start.any():
                            mode[start, i] = 1
                            target[start, i] = tgt[start, i]
                            hold_kind[start, i] = 0
                    acc = np.clip(v_plan - v[:, i], -a_cap * dt, a_cap * dt)
                    v[:, i] = np.where(moving, np.maximum(v[:, i] + acc, 0.0), v[:, i])
                    s[:, i] = np.where(moving, s[:, i] + v[:, i] * dt, s[:, i])
                    restart_left[:, i] = np.maximum(restart_left[:, i] - dt, 0.0)
                    done_restart = (mode[:, i] == 3) & (restart_left[:, i] <= 0.0)
                    mode[done_restart, i] = 0
                    # 到达路径末端：吸收为 HOLD
                    end = moving & (s[:, i] >= geo.length[i] - 1e-6)
                    if end.any():
                        mode[end, i] = 2
                        hold_left[end, i] = 1e9
                        v[end, i] = 0.0

                # BRAKE
                braking = mode[:, i] == 1
                if braking.any():
                    if not braking_on:
                        v[braking, i] = 0.0
                        mode[braking, i] = 2
                        hold_left[braking, i] = np.where(
                            hold_kind[braking, i] == 1, hold_left[braking, i],
                            _dwell_at_target(geo, i, target[braking, i]))
                    else:
                        dist = target[:, i] - s[:, i]
                        dist_eff = np.maximum(dist - v[:, i] * dt, 0.0)
                        v_des = np.minimum(v[:, i],
                                           np.sqrt(2.0 * max(self.p.a_brake, 1e-9) * dist_eff))
                        acc = np.clip(v_des - v[:, i], -self.p.a_brake * dt,
                                      self.p.a_acc * dt)
                        v[:, i] = np.where(braking, np.maximum(v[:, i] + acc, 0.0), v[:, i])
                        s[:, i] = np.where(braking, s[:, i] + v[:, i] * dt, s[:, i])
                        stop = braking & (v[:, i] <= V_STOP)
                        if stop.any():
                            mode[stop, i] = 2
                            v[stop, i] = 0.0
                            dwell = _dwell_at_target(geo, i, target[stop, i])
                            hold_left[stop, i] = np.where(hold_kind[stop, i] == 1,
                                                          hold_left[stop, i], dwell)
                # HOLD
                holding = mode[:, i] == 2
                if holding.any():
                    hold_left[:, i] = np.where(holding, hold_left[:, i] - dt, hold_left[:, i])
                    resume = holding & (hold_left[:, i] <= 0.0)
                    if resume.any():
                        mode[resume, i] = 3
                        restart_left[resume, i] = self.p.restart_time
                        v[resume, i] = 0.0
                s[:, i] = np.clip(s[:, i], 0.0, geo.length[i])

            positions[t + 1] = geo.positions(s)
            s_hist[t + 1], v_hist[t + 1], mode_hist[t + 1] = s, v, mode
            hold_hist[t + 1] = hold_left

        return {"positions": positions, "times": times, "mode": mode, "s": s, "v": v,
                "s_hist": s_hist, "v_hist": v_hist, "mode_hist": mode_hist,
                "hold_hist": hold_hist, "pause_starts": pause_starts,
                "n_samples": k, "dt": dt, "params": self.p.to_dict(),
                "flags": {"use_braking": braking_on, "use_modes": bool(use_modes),
                          "use_speed": bool(use_speed)}}

    # ------------------------------------------------------------------ #
    def predict(self, paths: np.ndarray, wait_slots: np.ndarray, pause_p: float,
                horizon: float, d_safe: float, n_samples: int = 200, seed: int = 0,
                **rollout_kwargs) -> dict:
        t0 = time.perf_counter()
        out = self.rollout(paths, wait_slots, pause_p, horizon, n_samples=n_samples,
                           seed=seed, **rollout_kwargs)
        pairs = _pair_index(out["positions"].shape[2])
        ev = compute_pair_first_hits(out["positions"], out["times"], pairs, d_safe)
        pair_risks = (np.isfinite(ev["pair_first"]).mean(axis=0) if len(pairs)
                      else np.zeros(0))
        team_est = float(np.mean(ev["team_hit"])) if n_samples else 0.0
        hist = {}
        for k, pair in enumerate(pairs):
            vals = ev["pair_first"][:, k]
            hist[f"{pair[0]}-{pair[1]}"] = np.sort(vals[np.isfinite(vals)]).tolist()
        team_vals = ev["team_first"][np.isfinite(ev["team_first"])]
        return {
            "pair_index": pairs,
            "pair_risks": np.asarray(pair_risks, dtype=np.float64),
            "pair_min_distance_mean": np.asarray(ev["pair_min"].mean(axis=0)),
            "first_hit_time_distribution": hist,
            "team_first_hit_times": np.sort(team_vals).tolist(),
            "team_collision_estimate": team_est,
            "n_samples": int(n_samples),
            "runtime_ms": float((time.perf_counter() - t0) * 1e3),
            "model_info": {
                "name": "M3 状态相关执行模型（采样式首次碰撞风险估计）",
                "is_exact_probability": False,
                "params": self.p.to_dict(),
                "flags": out["flags"],
            },
            "trace_arrays": {"positions": out["positions"], "times": out["times"],
                             "mode": out["mode"], "s": out["s"], "v": out["v"],
                             "s_hist": out["s_hist"], "v_hist": out["v_hist"],
                             "mode_hist": out["mode_hist"]},
        }


def _dwell_at_target(geo: PathGeometry, i: int, target: np.ndarray) -> np.ndarray:
    """按停车目标弧长找到该航点的计划等待时长（未命中则 0）。"""
    arr = np.asarray(target, dtype=np.float64)
    idx = np.clip(np.searchsorted(geo.s_wp[i], arr, side="right") - 1,
                  0, geo.h - 1)
    dwell = geo.dwell[i, idx]
    close = np.abs(geo.s_wp[i, idx] - arr) <= 1e-3
    return np.where(close, dwell, 0.0)


def _pair_index(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]
