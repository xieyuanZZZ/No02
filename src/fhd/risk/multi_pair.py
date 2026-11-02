"""N 机器人两两首次碰撞风险汇总（文档 §4.4），对外主入口 FirstHitRisk。

重要说明：
  * R_hard 只在 collision_mode="hard" 且 gate_mode="hard" 时才是**精确首次碰撞概率**；
    其余组合一律是优化用的可动代理量，不得当作概率报告。
  * 两两风险之和 sum_{i<j} R_ij 不是全队“至少碰撞一次”的概率；
    全队真实碰撞率必须由联合 N 机器人 Monte Carlo 估计（文档 §4.4）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fhd.execution.progress import finish_cost, move_probabilities, propagate_progress, release_times
from fhd.risk.collision_kernel import build_pair_index, pair_collision_tensors
from fhd.risk.first_hit_dp import first_hit_dp


class FirstHitRisk(nn.Module):
    """首次碰撞风险模块（纯 PyTorch，不依赖 diffusion）。"""

    def __init__(
        self,
        d_safe: float = 0.35,
        beta: float = 0.25,
        offset: float = 0.5,
        temperature_m: float = 0.025,
        kernel: str = "logistic",
        support_m: float = 0.05,
        dt_s: float = 0.4,
        lambda_tail_s: float = 2.0,
    ):
        super().__init__()
        self.d_safe = float(d_safe)
        self.beta = float(beta)
        self.offset = float(offset)
        self.temperature_m = float(temperature_m)
        self.kernel = str(kernel)
        self.support_m = float(support_m)
        self.dt_s = float(dt_s)
        self.lambda_tail_s = float(lambda_tail_s)

    @classmethod
    def from_config(cls, cfg) -> "FirstHitRisk":
        world = cfg["world"]
        risk = cfg["risk"]
        traj = cfg["trajectory"]
        d_safe = 2.0 * float(world["robot_radius_m"]) + float(world["safety_margin_m"])
        return cls(
            d_safe=d_safe,
            beta=float(risk["gate_beta"]),
            offset=float(risk["gate_offset_c"]),
            temperature_m=float(risk["soft_collision_temperature_m"]),
            kernel=str(risk["soft_collision_kernel"]),
            support_m=float(risk["smoothstep_support_m"]),
            dt_s=float(traj["dt_s"]),
            lambda_tail_s=float(cfg["guidance"]["lambda_tail_s"]),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        paths: torch.Tensor,
        wait: torch.Tensor,
        pause_p: torch.Tensor,
        time_steps: int,
        collision_mode: str = "soft",
        gate_mode: str = "soft",
        dt_s: float | None = None,
        lambda_tail_s: float | None = None,
        return_diagnostics: bool = True,
    ) -> dict:
        """返回 pair_risks、pair_sum_risk、finish_cost、unfinished_prob、diagnostics。"""
        if paths.dim() != 3:
            raise ValueError("paths 必须是 [N,H,2]")
        n, h, _ = paths.shape
        if wait.shape != (n, h - 1):
            raise ValueError(f"wait 形状应为 {(n, h - 1)}，实际 {tuple(wait.shape)}")

        dt = self.dt_s if dt_s is None else float(dt_s)
        lam_tail = self.lambda_tail_s if lambda_tail_s is None else float(lambda_tail_s)

        # 1) 前进概率与单机进度分布（J_finish 使用，与 DP 共用同一门槛）
        tau = release_times(wait)
        a = move_probabilities(
            tau, pause_p, time_steps, beta=self.beta, offset=self.offset, gate_mode=gate_mode
        )
        pi = propagate_progress(a)
        j_finish, unfinished = finish_cost(pi, dt_s=dt, lambda_tail_s=lam_tail)

        # 2) 两两首次碰撞风险
        ii, jj = build_pair_index(n)
        d_min, collision = pair_collision_tensors(
            paths,
            ii,
            jj,
            d_safe=self.d_safe,
            temperature_m=self.temperature_m,
            kernel=self.kernel,
            support_m=self.support_m,
            mode=collision_mode,
        )
        a_i = a.index_select(0, ii) if ii.numel() else a[:0]
        a_j = a.index_select(0, jj) if jj.numel() else a[:0]
        pair_risks, survival = first_hit_dp(collision, a_i, a_j, time_steps)

        result = {
            "pair_risks": pair_risks,
            "pair_sum_risk": pair_risks.sum() if pair_risks.numel() else paths.new_zeros(()),
            "finish_cost": j_finish,
            "unfinished_prob": unfinished,
            "pair_index": (ii, jj),
        }
        if return_diagnostics:
            with torch.no_grad():
                result["diagnostics"] = {
                    "survival": survival.detach(),
                    "min_pair_distance": d_min.amin(dim=(1, 2, 3)).detach() if d_min.numel() else d_min,
                    "move_probability_mean": a.mean().detach(),
                    "progress_entropy": (
                        -(pi.clamp(min=1e-12) * pi.clamp(min=1e-12).log()).sum(-1).mean().detach()
                    ),
                    "collision_mode": collision_mode,
                    "gate_mode": gate_mode,
                    "is_exact_probability": collision_mode == "hard" and gate_mode == "hard",
                }
        return result

    # ------------------------------------------------------------------
    def pair_collision_tensor(
        self,
        paths: torch.Tensor,
        collision_mode: str = "soft",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (ii, jj, d_min, C)，供可视化图 2 使用。"""
        ii, jj = build_pair_index(paths.shape[0])
        d_min, c = pair_collision_tensors(
            paths,
            ii,
            jj,
            d_safe=self.d_safe,
            temperature_m=self.temperature_m,
            kernel=self.kernel,
            support_m=self.support_m,
            mode=collision_mode,
        )
        return ii, jj, d_min, c


def team_union_bound(pair_risks: torch.Tensor) -> torch.Tensor:
    """硬风险的并集上界：P(全队至少一次碰撞) <= min(1, sum R_ij)。"""
    if pair_risks.numel() == 0:
        return torch.zeros((), dtype=pair_risks.dtype, device=pair_risks.device)
    return torch.clamp(pair_risks.sum(), max=1.0)


__all__ = ["FirstHitRisk", "team_union_bound"]
