"""B：简单在线冲突预测与固定优先级让行（不使用 First-Hit DP）。

决策规则（首版、可复现）：
  1. 每 `dt_slot` 用当前速度沿冻结路径外推 `horizon_s`，并截断在下一计划停车点；
  2. 对每对机器人求预测最小距离；`d_min < d_safe + margin` 视为冲突；
  3. 冲突中“离冲突点更远”的机器人让行（同等情况编号大者让行，编号小者优先）；
  4. 在**未越过、刹得住、冲突点之前**的候选航点中选最靠后的一个停车；
  5. 固定等待 `wait_slots` 个时隙，下一周期重新评估；有 cooldown/max-waits 防抖。
  6. 无可行停车点则不命令，记录 `no_feasible_stop`，不排除样本。
"""

from __future__ import annotations

import numpy as np

from fhd.execution.coordinator import Action, Coordinator, Observation


class SimpleConflictCoordinator(Coordinator):
    name = "B_simple_online"

    def __init__(self, paths, wait_slots, d_safe, dt_slot, horizon_s, pause_p,
                 config: dict | None = None):
        super().__init__(paths, wait_slots, d_safe, dt_slot, horizon_s, pause_p,
                         config)
        c = self.config
        self.margin = float(c.get("conflict_margin_m", 0.15))
        self.predict_dt = float(c.get("predict_dt", 0.1))
        self.hold_slots = int(c.get("hold_slots", 1))
        self.lookahead_m = float(c.get("lookahead_m", 2.0))
        self.cooldown_s = float(c.get("cooldown_s", 1.2))
        self.max_online_waits = int(c.get("max_online_waits", 6))
        self.stop_before_conflict_m = float(c.get("stop_before_conflict_m", 0.10))
        self._accepted_waits = {i: 0 for i in range(len(paths))}
        self._last_action_t = {i: -1e9 for i in range(len(paths))}

    # ------------------------------------------------------------------ #
    def on_result(self, action: Action, accepted: bool) -> None:
        """执行器接受后才更新防抖计数。"""
        if accepted and action.kind == "stop":
            self._accepted_waits[action.robot] = self._accepted_waits.get(action.robot, 0) + 1
            self._last_action_t[action.robot] = float(
                self.last_observation.t if self.last_observation else 0.0)

    # ------------------------------------------------------------------ #
    def decide(self, obs: Observation) -> list[Action]:
        self.observe(obs)
        self.last_decisions = []
        n = len(obs.robots)
        times = np.arange(0.0, self.horizon_s + 1e-9, self.predict_dt)
        pos = self.future_positions(obs, times)
        conflicts = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=-1)
                dmin = float(d.min())
                if dmin < self.d_safe + self.margin:
                    t_close = float(times[int(np.argmin(d))])
                    mid = 0.5 * (pos[int(np.argmin(d)), i] + pos[int(np.argmin(d)), j])
                    di = float(np.linalg.norm(pos[int(np.argmin(d)), i] - mid))
                    dj = float(np.linalg.norm(pos[int(np.argmin(d)), j] - mid))
                    conflicts.append({
                        "pair": (i, j), "dmin": dmin, "t_close": t_close,
                        "di": di, "dj": dj,
                    })
        conflicts.sort(key=lambda c: c["t_close"])

        used = set()
        no_feasible = 0
        for c in conflicts:
            i, j = c["pair"]
            # 远的让行；同等情况编号大的让行（编号小者优先）
            if abs(c["di"] - c["dj"]) < 0.10:
                yielder, keeper = (j, i) if j > i else (i, j)
            else:
                yielder, keeper = ((i, j) if c["di"] > c["dj"] else (j, i))
            if yielder in used:
                continue
            rb = obs.robots[yielder]
            cands = [c for c in rb.stop_candidates if c["dist"] <= self.lookahead_m]
            if rb.mode != "NORMAL" or not cands:
                no_feasible += 1
                continue
            if obs.t - rb.last_online_stop_t < self.cooldown_s:
                continue
            if self._accepted_waits.get(yielder, 0) >= self.max_online_waits:
                no_feasible += 1
                continue
            # 冲突点在该机器人路径上的弧长（名义预测）
            t_close = c["t_close"]
            s_conf = float(self.future_progress(obs, yielder,
                                                np.array([t_close]))[0])
            before = [cand for cand in cands
                      if cand["s"] < s_conf - self.stop_before_conflict_m]
            if not before:
                no_feasible += 1
                continue
            cand = max(before, key=lambda x: x["s"])   # 最靠后但仍早于冲突点
            action = Action(
                robot=int(yielder), kind="stop", s_target=float(cand["s"]),
                duration_s=float(self.hold_slots * self.dt_slot),
                candidate_wp=int(cand["wp"]),
                reason=(f"B_conflict_{i}_{j}_d={c['dmin']:.2f}"
                        f"_keeper={keeper}"),
                extra_time_s=float(self.hold_slots * self.dt_slot))
            used.add(yielder)
            self.last_decisions.append(action)
        self.last_diagnostics = {
            "n_conflicts": len(conflicts), "no_feasible_stop": no_feasible,
            "decisions": len(self.last_decisions),
        }
        return list(self.last_decisions)
