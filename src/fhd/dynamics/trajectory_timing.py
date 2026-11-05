"""沿冻结几何路径做**时间参数化**，并支持在指定航点停车等待。

输入：H 个航点（米制）+ 每个航点的等待时隙 w[k]（0.4 s/时隙）。
输出：`TimeParameterizedPath`
  * `v_limit(s)`：位置相关的速度上限（含加速度/制动/停车点约束）
  * `speed_at(s)`：实际参考速度
  * `dwell_at(s)`：该位置是否需要停车等待多少秒
  * `nominal_arrival()`：无随机暂停时的计划到达时刻（**计划值**，不是实测）

方法：沿弧长做前向（加速）与后向（制动到下一个停车点）两次扫描，
得到位置相关的速度上限；再积分出时间表。这是标准的梯形速度规划，
只是把"停车点"从两端扩展到所有需要等待的航点。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimingParams:
    ds: float = 0.01          # 弧长离散
    v_max: float = 1.5
    a_acc: float = 2.0
    a_brake: float = 2.0
    dt_slot: float = 0.4
    v_min: float = 0.02       # 低于此速度视为静止


class TimeParameterizedPath:
    def __init__(self, waypoints: np.ndarray, wait_slots: np.ndarray,
                 params: TimingParams):
        self.p = params
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        assert self.waypoints.ndim == 2 and self.waypoints.shape[1] == 2
        self.n_wp = self.waypoints.shape[0]

        seg = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=-1)
        self.s_wp = np.concatenate([[0.0], np.cumsum(seg)])       # 各航点弧长
        self.length = float(self.s_wp[-1])

        # 停车点：等待 > 0 的航点（含起点）
        w = np.asarray(wait_slots, dtype=np.float64)
        self.dwell = np.zeros(self.n_wp)
        self.dwell[: len(w)] = np.maximum(w, 0.0) * params.dt_slot
        self.stop_s = [float(self.s_wp[k]) for k in range(self.n_wp) if self.dwell[k] > 1e-9]

        self._build_speed_profile()

    # ------------------------------------------------------------------ #
    def _build_speed_profile(self):
        p = self.p
        n = max(int(np.ceil(self.length / p.ds)) + 1, 2)
        s = np.linspace(0.0, self.length, n)
        self.s_grid = s

        # 1) 纯速度上限
        v = np.full(n, p.v_max)

        # 2) 从每个停车点向后传播"必须能刹住"
        stops = sorted(set(self.stop_s + [self.length]))
        for s_stop in stops:
            idx = np.searchsorted(s, s_stop)
            for i in range(idx, -1, -1):
                d = max(s_stop - s[i], 0.0)
                v_allow = np.sqrt(2.0 * p.a_brake * d)
                if v[i] <= v_allow:
                    break
                v[i] = v_allow

        # 3) 从起点/上一停车点向前传播"能加速到"
        starts = sorted(set([0.0] + self.stop_s))
        for s0 in starts:
            idx = np.searchsorted(s, s0)
            for i in range(idx, n):
                d = s[i] - s0
                v_allow = np.sqrt(2.0 * p.a_acc * max(d, 0.0))
                v[i] = min(v[i], max(v_allow, 1e-6))
                if v[i] >= p.v_max and d > 1e-9:
                    # 已达到上限，继续向后不需要再限制
                    v[i + 1:] = np.minimum(v[i + 1:], p.v_max)
                    break

        # 2.5) 服从原调度表：不得比计划到达时刻更快（防止短段机器人抢跑）
        sched_v = self._scheduled_speed_cap()
        if sched_v is not None:
            v = np.minimum(v, np.interp(s, self.s_grid, sched_v))
        v = np.maximum(v, 1e-6)
        self.v_grid = v

        # 4) 积分时间表
        ds = np.diff(s)
        v_mid = 0.5 * (v[:-1] + v[1:])
        dt = ds / np.maximum(v_mid, 1e-6)
        self.t_grid = np.concatenate([[0.0], np.cumsum(dt)])
        self.duration_motion = float(self.t_grid[-1])
        self.duration_total = float(self.duration_motion + self.dwell.sum())

    # ------------------------------------------------------------------ #

    def _scheduled_speed_cap(self):
        """由离散调度表推出的位置-速度上限。

        语义约定（阶段一 §1.2-7）：
          * `w[k] = dwell[k]` 表示"到达航点 k 后、离开航点 k 前"的等待；
          * 离散模型里两航点之间的**行驶**时间恒为 1 个时隙，
            等待发生在航点处、由执行状态机停车实现，**不是**把等待
            摊进相邻段的平均速度。
        旧实现用 Δt_k = Δt_slot + dwell[k+1] 作为第 k 段的可用时间，
        等于把"在 k+1 等待"提前摊到"k→k+1 的行驶"上，既改变了
        原始 W2+ 等待计划的含义，又让机器人提前减速。这里改为
        每段平均速度上限 = Δs_k / dt_slot；等待只由 `dwell` 与
        状态机负责。
        """
        if not getattr(self, "use_schedule", True):
            return None
        cap = np.full(self.s_grid.shape, self.p.v_max)
        for k in range(self.n_wp - 1):
            seg_len = max(self.s_wp[k + 1] - self.s_wp[k], 1e-9)
            vavg = seg_len / max(self.p.dt_slot, 1e-9)
            mask = (self.s_grid >= self.s_wp[k]) & (self.s_grid <= self.s_wp[k + 1])
            cap[mask] = np.minimum(cap[mask], vavg)
        return cap

    def speed_at(self, s: float) -> float:
        return float(np.interp(s, self.s_grid, self.v_grid))

    def position_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        k = int(np.searchsorted(self.s_wp, s, side="right") - 1)
        k = min(max(k, 0), self.n_wp - 2)
        seg_len = max(self.s_wp[k + 1] - self.s_wp[k], 1e-12)
        w = (s - self.s_wp[k]) / seg_len
        return self.waypoints[k] + w * (self.waypoints[k + 1] - self.waypoints[k])

    def tangent_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        k = int(np.searchsorted(self.s_wp, s, side="right") - 1)
        k = min(max(k, 0), self.n_wp - 2)
        d = self.waypoints[k + 1] - self.waypoints[k]
        nrm = np.linalg.norm(d)
        return d / nrm if nrm > 1e-12 else np.array([1.0, 0.0])

    def dwell_between(self, s0: float, s1: float) -> float:
        """(s0, s1] 区间内需要停车等待的总时长。"""
        tot = 0.0
        for k in range(self.n_wp):
            if self.dwell[k] > 1e-9 and s0 + 1e-9 < self.s_wp[k] <= s1 + 1e-9:
                tot += float(self.dwell[k])
        return tot

    def nominal_arrival_times(self) -> dict:
        """无随机暂停时的**计划**时刻（旧实现把累计时间重复累加，已修正）。

        定义（与离散执行模型 `tau_m = m + Σ_{h<=m} w_h` 对齐）：
          * `t_grid` 本身是**累计运动时间**，因此到航点 k 的运动时间为
            `t_grid(s_wp[k])`，不能再把之前航点的 t_grid 累加一次；
          * `arrival[k]  = t_grid(s_wp[k]) + Σ_{h<k} dwell[h]`
            （到达航点 k、尚未等待的时刻）；
          * `departure[k] = arrival[k] + dwell[k]`
            （在航点 k 等待结束、准备离开的时刻）；
          * `wait_before_arrival` / `wait_after_arrival` 两种定义都显式给出，
            调用方不得混用。

        返回字典包含：waypoint、arrival、departure、motion_time、
        wait_before_arrival、wait_after_arrival、time（= arrival，兼容旧键）。
        """
        out = {"waypoint": [], "arrival": [], "departure": [], "motion_time": [],
               "wait_before_arrival": [], "wait_after_arrival": [], "time": []}
        cum_wait = 0.0
        for k in range(self.n_wp):
            motion = float(np.interp(float(self.s_wp[k]), self.s_grid, self.t_grid))
            arrival = motion + cum_wait
            departure = arrival + float(self.dwell[k])
            out["waypoint"].append(k)
            out["motion_time"].append(motion)
            out["wait_before_arrival"].append(cum_wait)
            out["wait_after_arrival"].append(float(self.dwell[k]))
            out["arrival"].append(arrival)
            out["departure"].append(departure)
            out["time"].append(arrival)      # 兼容旧接口：time = 到达时刻
            cum_wait += float(self.dwell[k])
        return out

    def scheduled_segment_durations(self) -> np.ndarray:
        """离散名义调度下每段的**行驶**时间（单位：秒，不含航点等待）。"""
        return np.full(max(self.n_wp - 1, 0), float(self.p.dt_slot))

    def project(self, p: np.ndarray) -> float:
        """把任意点投影到整条折线上，返回弧长（无历史约束的全局投影）。"""
        s, _ = project_point_on_path(self.waypoints, self.s_wp, p, None, None)
        return s


def project_point_on_path(waypoints: np.ndarray, s_wp: np.ndarray, p: np.ndarray,
                          s_prev: float | None = None,
                          back_window: float | None = None,
                          forward_window: float | None = None) -> tuple[float, float]:
    """折线投影；给定 `s_prev` 时只在历史邻域内搜索，返回 (弧长, 横向误差)。

    折线拐角、近距离平行段或自交路径上，全局最近点可能让进度倒跳。
    这里把候选段限制在 `[s_prev - back_window, s_prev + forward_window]`，
    再选择最近点；若最近点仍略小于 s_prev（数值噪声），夹回 s_prev。
    真实倒车需调用方显式传入更大的 back_window。
    """
    p = np.asarray(p, dtype=np.float64)
    best, best_s = np.inf, 0.0
    lo = -np.inf if s_prev is None or back_window is None else float(s_prev) - float(back_window)
    hi = np.inf if s_prev is None or forward_window is None else float(s_prev) + float(forward_window)
    for k in range(len(waypoints) - 1):
        s0, s1 = float(s_wp[k]), float(s_wp[k + 1])
        if s1 < lo or s0 > hi:
            continue
        a, b = waypoints[k], waypoints[k + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        q = a + t * ab
        dist = float(np.linalg.norm(p - q))
        s_cand = s0 + t * np.sqrt(L2)
        if dist < best - 1e-12:
            best, best_s = dist, s_cand
    if not np.isfinite(best):
        # 没有任何段落入邻域：退化为全路径投影
        return project_point_on_path(waypoints, s_wp, p, None, None)
    if s_prev is not None:
        best_s = max(best_s, float(s_prev) - 1e-9)
    return float(best_s), float(best)


class MonotoneProgress:
    """带历史邻域约束的单调弧长进度跟踪器。

    * `back_window` 允许很小的数值回退；正数时夹回上一进度；
    * 需要真实倒车时显式设置 `allow_reverse=True` 并给出额外窗口；
    * 同时保存横向误差，避免把错误投影误判为到达。
    """

    def __init__(self, path: TimeParameterizedPath, back_window: float = 0.15,
                 forward_window: float = 1.2, allow_reverse: bool = False):
        self.path = path
        self.back_window = float(back_window)
        self.forward_window = float(forward_window)
        self.allow_reverse = bool(allow_reverse)
        self.s = 0.0
        self.lateral = 0.0

    def reset(self, p: np.ndarray | None = None) -> float:
        self.s, self.lateral = project_point_on_path(
            self.path.waypoints, self.path.s_wp, np.zeros(2) if p is None else p)
        return self.s

    def update(self, p: np.ndarray) -> tuple[float, float]:
        back = self.forward_window if self.allow_reverse else self.back_window
        s, lat = project_point_on_path(self.path.waypoints, self.path.s_wp, p,
                                       self.s, back, self.forward_window)
        if not self.allow_reverse:
            s = max(s, self.s)
        self.s, self.lateral = float(s), float(lat)
        return self.s, self.lateral
