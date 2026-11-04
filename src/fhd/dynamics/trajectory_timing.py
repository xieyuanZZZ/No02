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
        """由离散调度表 t_k = (k + Σw)·Δt 推出的位置-速度上限。

        离散执行模型假设每时隙推进一个航点，因此每段的平均速度是
        Δs_k / Δt_k。MPC 若不受此限制，短段机器人会以 v_max 抢跑，
        相对时序被破坏、p=0 的安全保证失效。这里把该速度作为**上限**，
        只允许更慢（物理受限时自然变慢），不允许更快。
        """
        if not getattr(self, "use_schedule", True):
            return None
        t_k = np.zeros(self.n_wp)
        acc = 0.0
        for k in range(self.n_wp):
            if k > 0:
                acc += float(self.dwell[k])
            t_k[k] = k * self.p.dt_slot + acc
        cap = np.full(self.s_grid.shape, self.p.v_max)
        for k in range(self.n_wp - 1):
            dt_seg = max(t_k[k + 1] - t_k[k], 1e-6)
            seg_len = max(self.s_wp[k + 1] - self.s_wp[k], 1e-9)
            vavg = seg_len / dt_seg
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

    def project(self, p: np.ndarray) -> float:
        """把任意点投影到路径上，返回弧长（用于从实际位置估计进度）。"""
        d = np.linalg.norm(self.waypoints[1:] - self.waypoints[:-1], axis=-1)
        best, best_s = np.inf, 0.0
        for k in range(self.n_wp - 1):
            a, b = self.waypoints[k], self.waypoints[k + 1]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-12:
                continue
            t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            q = a + t * ab
            dist = float(np.linalg.norm(p - q))
            if dist < best:
                best, best_s = dist, float(self.s_wp[k] + t * np.sqrt(L2))
        return best_s

    def nominal_arrival_times(self) -> dict:
        """无随机暂停时的**计划**到达时刻（含停车等待）。"""
        t = 0.0
        out = {"waypoint": [], "time": []}
        for k in range(self.n_wp):
            s = float(self.s_wp[k])
            t += float(np.interp(s, self.s_grid, self.t_grid))
            out["waypoint"].append(k)
            out["time"].append(t)
            t += float(self.dwell[k])
        return out
