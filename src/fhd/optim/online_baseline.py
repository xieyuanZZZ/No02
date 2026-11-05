"""修复后的简单在线冲突预测与固定优先级让行（不使用 First-Hit DP）。

关键修复（对应 v2 第一阶段）：
  * 冲突定位不再用"两者当前位置的中点"（该量对两台机器人恒相等，无法判断谁更远）；
  * 在预测危险时刻分别沿各自路径保留冲突弧长 `s_conf_i/s_conf_j`，用
    "沿路径剩余距离 / 名义速度" 计算预计到达时间（ETA）；
  * 让预计**较晚**到达者让行；只有 ETA 差小于 `eta_tie_s` 时才用机器人 ID 作
    唯一平局规则；
  * 停车点必须在冲突弧长之前且满足执行器制动距离；首选方不可行时检查另一方；
  * 动作前做一次轻量全队前瞻：新增等待后不得在其它 pair 上制造新的近距离冲突，
    也不能让当前 pair 更差；
  * 预测按已知计划等待**恢复**（等待结束后继续沿路径前进），不再到下一等待点
    就一直静止；
  * 用已处理冲突记忆 + cooldown 防止同一事件每周期反复停车。
"""

from __future__ import annotations

import numpy as np

from fhd.execution.coordinator import Action, Coordinator, Observation


class SimpleConflictCoordinator(Coordinator):
    name = "simple_conflict_online"

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
        self.eta_tie_s = float(c.get("eta_tie_s", 0.10))
        self.conflict_memory_s = float(c.get("conflict_memory_s", 3.0))
        self._accepted_waits = {i: 0 for i in range(len(paths))}
        self._last_action_t = {i: -1e9 for i in range(len(paths))}
        self._handled = {}

    # ------------------------------------------------------------------ #
    def on_result(self, action: Action, accepted: bool) -> None:
        if accepted and action.kind == "stop":
            self._accepted_waits[action.robot] = self._accepted_waits.get(action.robot, 0) + 1
            self._last_action_t[action.robot] = float(
                self.last_observation.t if self.last_observation else 0.0)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _eta(progress: np.ndarray, s_target: float, times: np.ndarray) -> float:
        hit = np.where(progress >= s_target - 0.02)[0]
        return float(times[int(hit[0])]) if hit.size else float("inf")

    def _conflict_scan(self, obs, extra_wait=None):
        n = len(obs.robots)
        times = np.arange(0.0, self.horizon_s + 1e-9, self.predict_dt)
        prog = np.stack([self.future_progress(obs, i, times, extra_wait)
                         for i in range(n)], axis=1)          # [T,N]
        pos = np.zeros((len(times), n, 2))
        for i in range(n):
            for k, sk in enumerate(prog[:, i]):
                pos[k, i] = self.s_to_position(self.paths[i], sk)
        out = []
        # p=0 没有随机暂停不确定性：用很小的数值余量，避免把"接近但安全"
        # 的原始计划误判为冲突并在安全场景制造新碰撞。
        # p=0 时只有"预测已明显进入安全距离以内"才允许干预，避免把
        # 接近但安全的原始计划当作冲突并在安全场景制造新碰撞。
        threshold = (self.d_safe + self.margin if obs.pause_p > 0.0
                     else self.d_safe - 0.05)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=-1)
                dmin = float(d.min())
                if dmin >= threshold:
                    continue
                idx = int(np.argmin(d))
                s_ci = float(prog[idx, i])
                s_cj = float(prog[idx, j])
                eta_i = self._eta(prog[:, i], s_ci, times)
                eta_j = self._eta(prog[:, j], s_cj, times)
                # 首次进入阈值（不是最小距离点）
                first = int(np.argmax(d < threshold))
                out.append({
                    "pair": (i, j), "dmin": dmin, "t_min": float(times[idx]),
                    "t_first": float(times[first]),
                    "s_conf_i": s_ci, "s_conf_j": s_cj,
                    "eta_i": eta_i, "eta_j": eta_j,
                    "pos_i": pos[idx, i].copy(), "pos_j": pos[idx, j].copy(),
                })
        return times, prog, pos, out

    # ------------------------------------------------------------------ #
    def _yielder(self, c) -> tuple[int, int, str]:
        i, j = c["pair"]
        ei, ej = c["eta_i"], c["eta_j"]
        if abs(ei - ej) < self.eta_tie_s:
            # 唯一平局规则：编号大者让行（编号小者优先）
            if i > j:
                return i, j, f"tie_eta_{ei:.2f}_{ej:.2f}"
            return j, i, f"tie_eta_{ei:.2f}_{ej:.2f}"
        if ei > ej:
            return i, j, f"later_eta_{ei:.2f}>{ej:.2f}"
        return j, i, f"later_eta_{ej:.2f}>{ei:.2f}"

    def _candidate_for(self, obs, yielder: int, s_conf: float):
        rb = obs.robots[yielder]
        cands = [c for c in rb.stop_candidates
                 if c["dist"] <= self.lookahead_m
                 and c["s"] < s_conf - self.stop_before_conflict_m]
        if not cands:
            return None, "no_candidate_before_conflict"
        return max(cands, key=lambda x: x["s"]), ""

    def _check_new_conflicts(self, obs, base_pairs, extra_wait, current_pair):
        """新增等待后：当前 pair 不得更差，其它 pair 不得新进入阈值。"""
        times, prog2, pos2, conflicts2 = self._conflict_scan(obs, extra_wait)
        new = {(c["pair"][0], c["pair"][1]): c for c in conflicts2}
        base = {(c["pair"][0], c["pair"][1]): c for c in base_pairs}
        cur = new.get((current_pair[0], current_pair[1]))
        bcur = base.get((current_pair[0], current_pair[1]))
        if bcur is not None:
            if cur is None:
                pass          # 冲突消失，动作有效
            else:
                # 停一次可能只是把冲突时刻推后，最小间距不一定变大；
                # 只要"首次冲突时刻明显延后"或"最小间距变大"即可继续。
                delayed = cur["t_first"] >= bcur["t_first"] + 0.10
                safer = cur["dmin"] >= bcur["dmin"] + 0.02
                if not (delayed or safer):
                    return False, (f"current_pair_not_improved("
                                   f"t {cur['t_first']:.2f}<={bcur['t_first']:.2f}, "
                                   f"d {cur['dmin']:.3f}<={bcur['dmin']:.3f})")
        for pair, c2 in new.items():
            if pair == (current_pair[0], current_pair[1]):
                continue
            if pair not in base:
                return False, f"new_conflict_{pair[0]}_{pair[1]}"
            if c2["dmin"] + 0.02 < base[pair]["dmin"]:
                return False, f"worse_pair_{pair[0]}_{pair[1]}"
        return True, ""

    # ------------------------------------------------------------------ #
    def decide(self, obs: Observation) -> list[Action]:
        self.observe(obs)
        self.last_decisions = []
        # p=0 没有随机暂停，在线协调没有可修正的扰动；直接保留离线计划，
        # 避免把"接近但安全"的确定性计划改成新碰撞。这是首版明确边界。
        if obs.pause_p <= 0.0:
            self.last_diagnostics = {"n_conflicts": 0, "conflicts": [],
                                     "no_feasible_stop": 0,
                                     "rejected_new_conflict": 0,
                                     "decisions": 0, "p0_no_action": True}
            return []
        times, prog, pos, conflicts = self._conflict_scan(obs)
        conflicts.sort(key=lambda c: c["t_first"])
        diagnostics = {"n_conflicts": len(conflicts), "conflicts": [],
                       "no_feasible_stop": 0, "rejected_new_conflict": 0,
                       "decisions": 0}

        for c in conflicts:
            key = (c["pair"][0], c["pair"][1], round(c["s_conf_i"], 1),
                   round(c["s_conf_j"], 1), round(c["t_first"], 1))
            if obs.t - self._handled.get(key, -1e9) < self.conflict_memory_s:
                continue
            yielder, keeper, rule = self._yielder(c)
            tried = []
            for cand_robot, s_conf in ((yielder, c["s_conf_i"] if yielder == c["pair"][0]
                                        else c["s_conf_j"]),
                                       (keeper, c["s_conf_j"] if keeper == c["pair"][1]
                                        else c["s_conf_i"])):
                rb = obs.robots[cand_robot]
                if rb.mode != "NORMAL":
                    tried.append(f"r{cand_robot}_mode_{rb.mode}")
                    continue
                if obs.t - rb.last_online_stop_t < self.cooldown_s:
                    tried.append(f"r{cand_robot}_cooldown")
                    continue
                if self._accepted_waits.get(cand_robot, 0) >= self.max_online_waits:
                    tried.append(f"r{cand_robot}_max_waits")
                    continue
                cand, why = self._candidate_for(obs, cand_robot, s_conf)
                if cand is None:
                    tried.append(f"r{cand_robot}_{why}")
                    continue
                extra = {cand_robot: (int(cand["wp"]), self.hold_slots)}
                ok, why2 = self._check_new_conflicts(obs, conflicts, extra,
                                                     c["pair"])
                if not ok:
                    diagnostics["rejected_new_conflict"] += 1
                    tried.append(f"r{cand_robot}_{why2}")
                    continue
                action = Action(
                    robot=int(cand_robot), kind="stop",
                    s_target=float(cand["s"]),
                    duration_s=float(self.hold_slots * self.dt_slot),
                    candidate_wp=int(cand["wp"]),
                    reason=(f"simple_conflict pair={c['pair']} {rule} "
                            f"eta_i={c['eta_i']:.2f} eta_j={c['eta_j']:.2f} "
                            f"s_conf=({c['s_conf_i']:.2f},{c['s_conf_j']:.2f})"),
                    extra_time_s=float(self.hold_slots * self.dt_slot))
                self.last_decisions = [action]
                self._handled[key] = obs.t
                diagnostics["conflicts"].append({
                    **{k: (list(v) if isinstance(v, np.ndarray) else v)
                       for k, v in c.items()},
                    "yielder": int(cand_robot), "keeper": int(keeper),
                    "rule": rule, "s_stop": float(cand["s"]),
                    "wp_stop": int(cand["wp"]), "tried": tried,
                })
                diagnostics["decisions"] = 1
                self.last_diagnostics = diagnostics
                return [action]
            diagnostics["no_feasible_stop"] += 1
            diagnostics["conflicts"].append({
                **{k: (list(v) if isinstance(v, np.ndarray) else v)
                   for k, v in c.items()},
                "yielder": None, "keeper": int(keeper), "rule": rule,
                "tried": tried,
            })

        if not conflicts:
            diagnostics["no_feasible_stop"] = 0
        self.last_diagnostics = diagnostics
        return []
