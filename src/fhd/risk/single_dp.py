"""单机器人进度分布传播（文档 §4.5）：用于 J_finish 与基线 B3。

与两机器人 DP 共用同一套前进概率 a_{i,m}(t)，但不含机器人间碰撞吸收。
"""

from __future__ import annotations

import torch

from fhd.execution.progress import finish_cost, move_probabilities, propagate_progress, release_times


def single_agent_distributions(
    wait: torch.Tensor,
    pause_p: torch.Tensor,
    time_steps: int,
    beta: float = 0.25,
    offset: float = 0.5,
    gate_mode: str = "soft",
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (a [N,T,H], pi [N,T+1,H])。"""
    tau = release_times(wait)
    a = move_probabilities(tau, pause_p, time_steps, beta=beta, offset=offset, gate_mode=gate_mode)
    pi = propagate_progress(a)
    return a, pi


def joint_independent_distribution(pi: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """暂停独立假设下的联合进度分布 P_t(m,n) = pi_i^t(m) pi_j^t(n)。

    pi [N,T+1,H] -> [T+1,H,H]
    """
    return pi[i].unsqueeze(-1) * pi[j].unsqueeze(-2)


def progress_entropy(pi: torch.Tensor) -> torch.Tensor:
    """进度分布的平均熵（诊断量，越大表示越不确定）。"""
    p = pi.clamp(min=1e-12)
    return -(p * p.log()).sum(-1).mean()


def completion_time_stats(pi: torch.Tensor, dt_s: float) -> dict:
    """由进度分布得到的完成时间统计（秒）。"""
    t_steps = pi.shape[1] - 1
    arrived = pi[:, :, -1]  # [N,T+1]
    # P(T_finish = t) = pi^t(H-1) - pi^{t-1}(H-1)
    pmf = torch.clamp(arrived[:, 1:] - arrived[:, :-1], min=0.0)  # [N,T]
    steps = torch.arange(1, t_steps + 1, dtype=pi.dtype, device=pi.device)
    mean_steps = (pmf * steps).sum(-1) / (pmf.sum(-1) + 1e-12)
    return {
        "mean_completion_s": (mean_steps * dt_s).tolist(),
        "unfinished_prob": (1.0 - arrived[:, -1]).tolist(),
        "arrival_probability_curve": arrived.detach().to(torch.float64).tolist(),
    }


__all__ = [
    "single_agent_distributions",
    "joint_independent_distribution",
    "progress_entropy",
    "completion_time_stats",
    "propagate_progress",
    "move_probabilities",
    "finish_cost",
]
