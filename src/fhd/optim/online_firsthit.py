"""C：当前状态条件化的 First-Hit 在线等待（首版有限搜索）。

与 B 共用同一个 `Coordinator` 接口、候选停车点、动作约束、决策周期与执行器。
差异只在风险判据：

  1. 从**当前实际进度**截断出每台机器人的剩余路径与剩余等待；
  2. 用 `FirstHitRisk`（hard+hard）在短时域（默认 3.2 s = 8 个时隙）上
     计算 pair 风险；`pair_sum_risk` 仅作排序目标，不是全队概率；
  3. 对“某机器人停在某个可行候选点 1 个时隙”做有限枚举，选
     `risk_after + λ·extra_time` 最小且相对 baseline 有足够下降的动作；
  4. 每次只提交一个动作；下一周期重新观察。计算超时/失败时记录并退回
     简单冲突策略（B）的动作。

明确近似：这是**条件化近似风险**，不是校准的 MPC 真实碰撞概率；剩余路径
的 DP 从当前进度 0 开始，首段长度可能短于名义段长；不读取未来事件。
"""

from __future__ import annotations

import time

import numpy as np
import torch

from fhd.execution.coordinator import Action, Coordinator, Observation
from fhd.optim.online_baseline import SimpleConflictCoordinator
from fhd.risk.multi_pair import FirstHitRisk


class OnlineFirstHitCoordinator(Coordinator):
    name = "firsthit_online"

    def __init__(self, paths, wait_slots, d_safe, dt_slot, horizon_s, pause_p,
                 risk_model: FirstHitRisk, config: dict | None = None):
        super().__init__(paths, wait_slots, d_safe, dt_slot, horizon_s, pause_p,
                         config)
        c = self.config
        self.risk = risk_model
        self.hold_slots = int(c.get("hold_slots", 1))
        self.max_candidates_per_robot = int(c.get("max_candidates_per_robot", 2))
        self.lambda_time = float(c.get("lambda_time", 0.005))
        self.min_risk_reduction = float(c.get("min_risk_reduction", 0.02))
        self.min_baseline_risk = float(c.get("min_baseline_risk", 0.02))
        self.max_online_waits = int(c.get("max_online_waits", 6))
        self.cooldown_s = float(c.get("cooldown_s", 1.2))
        self.decision_budget_s = float(c.get("decision_budget_s", 0.4))
        self.fallback = SimpleConflictCoordinator(
            paths, wait_slots, d_safe, dt_slot, horizon_s, pause_p,
            {**c, "hold_slots": self.hold_slots})
        self._accepted_waits = {i: 0 for i in range(len(paths))}
        self._last_action_t = {i: -1e9 for i in range(len(paths))}
        self.last_runtime_s = 0.0
        self.used_fallback = False

    def on_result(self, action: Action, accepted: bool) -> None:
        if accepted and action.kind == "stop":
            self._accepted_waits[action.robot] = self._accepted_waits.get(action.robot, 0) + 1
            self._last_action_t[action.robot] = float(
                self.last_observation.t if self.last_observation else 0.0)

    # ------------------------------------------------------------------ #
    def _residual_model(self, obs: Observation, extra_wait: dict | None = None):
        """返回 (paths_res, wait_res, next_idx, T_steps)。"""
        n = len(obs.robots)
        h = self.paths.shape[1]
        next_idx, paths_list, waits_list = [], [], []
        for i in range(n):
            rb = obs.robots[i]
            k = int(np.searchsorted(self.s_wp[i], rb.s + 1e-6, side="right"))
            k = min(max(k, 1), h - 1)
            next_idx.append(k)
            pts = [self.s_to_position(self.paths[i], rb.s)] + \
                [self.paths[i, q] for q in range(k, h)]
            paths_list.append(np.asarray(pts, dtype=np.float64))
            # 剩余等待：索引 0=当前位置，索引 m>=1 对应原航点 k+m-1
            w = np.zeros(len(pts) - 1)
            if rb.mode == "HOLD":
                w[0] = max(float(rb.hold_remaining) / self.dt_slot, 1.0)
            elif rb.mode in ("BRAKE_DWELL", "BRAKE_PAUSE", "CREEP_DWELL"):
                w[0] = 1.0
            for m in range(1, len(pts) - 1):
                orig = k + m - 1
                if orig < h - 1:
                    w[m] = float(self.wait_slots[i, orig])
            if extra_wait and i in extra_wait:
                m, slots = extra_wait[i]
                if 0 <= m < len(w):
                    w[m] += float(slots)
            waits_list.append(w)
        hmax = max(len(p) for p in paths_list)
        paths_res = np.zeros((n, hmax, 2))
        wait_res = np.zeros((n, hmax - 1))
        for i, (p, w) in enumerate(zip(paths_list, waits_list)):
            paths_res[i, :len(p)] = p
            paths_res[i, len(p):] = p[-1]
            wait_res[i, :len(w)] = w
        return (torch.tensor(paths_res, dtype=torch.float64),
                torch.tensor(wait_res, dtype=torch.float64), next_idx)

    def _risk(self, paths_res, wait_res, n: int) -> float:
        with torch.no_grad():
            out = self.risk(
                paths_res, wait_res,
                torch.full((n,), self.pause_p, dtype=torch.float64),
                int(max(1, round(self.horizon_s / self.dt_slot))),
                collision_mode="hard", gate_mode="hard",
                return_diagnostics=False)
        return float(out["pair_sum_risk"])

    # ------------------------------------------------------------------ #
    def decide(self, obs: Observation) -> list[Action]:
        t0 = time.perf_counter()
        self.observe(obs)
        self.last_decisions = []
        self.used_fallback = False
        n = len(obs.robots)
        paths_res, wait_res, next_idx = self._residual_model(obs)
        base_risk = self._risk(paths_res, wait_res, n)
        self._base_risk = base_risk

        candidates = []
        for i, rb in enumerate(obs.robots):
            cands = [c for c in rb.stop_candidates
                     if c["dist"] <= float(self.config.get("lookahead_m", 2.0))]
            if rb.mode != "NORMAL" or not cands:
                continue
            if obs.t - rb.last_online_stop_t < self.cooldown_s:
                continue
            if self._accepted_waits.get(i, 0) >= self.max_online_waits:
                continue
            for cand in cands[:self.max_candidates_per_robot]:
                m = 1 + (int(cand["wp"]) - next_idx[i])
                if m < 0:
                    continue
                candidates.append((i, cand, m))

        best = None
        if base_risk >= self.min_baseline_risk:
            for (i, cand, m) in candidates:
                # 单动作风险：只改一台机器人的剩余等待
                pr2, wr2, _ = self._residual_model(obs, extra_wait={i: (m, self.hold_slots)})
                risk2 = self._risk(pr2, wr2, n)
                extra = self.hold_slots * self.dt_slot
                score = risk2 + self.lambda_time * extra
                reduction = base_risk - risk2
                if best is None or score < best["score"]:
                    best = {"score": score, "risk": risk2, "reduction": reduction,
                            "robot": i, "cand": cand, "m": m, "extra": extra}
        elapsed = time.perf_counter() - t0
        self.last_runtime_s = elapsed

        if best is not None and best["reduction"] >= self.min_risk_reduction:
            action = Action(
                robot=int(best["robot"]), kind="stop",
                s_target=float(best["cand"]["s"]),
                duration_s=float(self.hold_slots * self.dt_slot),
                candidate_wp=int(best["cand"]["wp"]),
                reason=f"C_firsthit_reduce_{best['reduction']:.3f}",
                risk_before=float(base_risk), risk_after=float(best["risk"]),
                extra_time_s=float(best["extra"]))
            self.last_decisions = [action]
            self.last_diagnostics = {
                "base_risk": base_risk, "best_risk": best["risk"],
                "reduction": best["reduction"], "n_candidates": len(candidates),
                "runtime_s": elapsed, "fallback": False,
            }
            return [action]

        # 超时或风险无法下降：首版退回 B 的动作，并标记
        if elapsed > self.decision_budget_s:
            fallback_actions = self.fallback.decide(obs)
            self.used_fallback = True
            self.last_decisions = list(fallback_actions)
            self.last_diagnostics = {
                "base_risk": base_risk, "runtime_s": elapsed,
                "fallback": True, "n_candidates": len(candidates),
                "reason": "decision_budget_exceeded",
            }
            return list(fallback_actions)

        self.last_diagnostics = {
            "base_risk": base_risk, "runtime_s": elapsed,
            "fallback": False, "n_candidates": len(candidates),
            "reason": "no_sufficient_risk_reduction",
        }
        return []
