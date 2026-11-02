"""核心：两机器人首次碰撞概率动态规划（文档 §4.3）。

F_t(m,n) = P(0..t 未碰撞 且 z_i(t)=m, z_j(t)=n)

    G^u_t(m,n) = F_t(m,n) P_u^t(m,n) (1 - C_{mn,u})
    F_{t+1}(m',n') = sum_{(m,n),u->(m',n')} G^u_t(m,n)
    R = 1 - sum_{m,n} F_T(m,n)

碰撞后概率质量不再传播，因此同一条轨迹的多次碰撞只计一次“首次碰撞”。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F_pad


def _shift_mass(x: torch.Tensor) -> torch.Tensor:
    """沿进度维右移一格（m -> m+1），丢弃移出边界的质量。

    x [P,H,H] -> [P,H,H]，新张量，不做原地累加（文档 §5.2）。
    """
    h = x.shape[-2]
    padded = F_pad.pad(x, (0, 0, 1, 0))  # 倒数第二维前补 1
    return padded[:, :h, :]


def _shift_n(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1]
    padded = F_pad.pad(x, (1, 0))
    return padded[:, :, :h]


def _shift_mn(x: torch.Tensor) -> torch.Tensor:
    h0, h1 = x.shape[-2], x.shape[-1]
    padded = F_pad.pad(x, (1, 0, 1, 0))
    return padded[:, :h0, :h1]


def first_hit_dp(
    collision: torch.Tensor,
    a_i: torch.Tensor,
    a_j: torch.Tensor,
    time_steps: int,
    return_history: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """两机器人首次碰撞概率 DP。

    参数
    ----
    collision : [P,4,H,H]  四种动作的碰撞量（soft 为代理，hard 为 0/1）
    a_i, a_j  : [P,T,H]    各机器人前进概率（最后一列为 0，表示终点吸收）
    time_steps: T
    return_history: 额外返回每个时刻的 F 场（detach），供可视化

    返回
    ----
    risks    : [P]   1 - sum F_T
    survival : [P,T+1]  每个时刻的存活概率质量（detach 后的诊断量）
    history  : [P,T+1,H,H]（仅当 return_history=True）
    """
    p, _, h, _ = collision.shape
    dtype, device = collision.dtype, collision.device

    f = torch.zeros(p, h, h, dtype=dtype, device=device)
    f[:, 0, 0] = 1.0

    survival = [f.sum(dim=(-2, -1))]
    history = [f.detach().clone()] if return_history else None

    for t in range(time_steps):
        ai = a_i[:, t, :]  # [P,H]
        aj = a_j[:, t, :]
        pi = ai.unsqueeze(-1)  # [P,H,1]
        pj = aj.unsqueeze(-2)  # [P,1,H]
        p00 = (1.0 - pi) * (1.0 - pj)
        p10 = pi * (1.0 - pj)
        p01 = (1.0 - pi) * pj
        p11 = pi * pj

        keep = 1.0 - collision  # [P,4,H,H]
        g00 = f * p00 * keep[:, 0]
        g10 = f * p10 * keep[:, 1]
        g01 = f * p01 * keep[:, 2]
        g11 = f * p11 * keep[:, 3]

        f = g00 + _shift_mass(g10) + _shift_n(g01) + _shift_mn(g11)
        survival.append(f.sum(dim=(-2, -1)))
        if history is not None:
            history.append(f.detach().clone())

    risks = 1.0 - f.sum(dim=(-2, -1))
    survival_t = torch.stack(survival, dim=-1)
    if history is not None:
        return risks, survival_t, torch.stack(history, dim=1)
    return risks, survival_t


def first_hit_risk_single(
    collision: torch.Tensor,
    a_i: torch.Tensor,
    a_j: torch.Tensor,
    time_steps: int,
) -> torch.Tensor:
    """单对（无批维）版本，便于单元测试与手工枚举对照。

    collision [4,H,H]，a_i/a_j [T,H] -> 标量风险。
    """
    risks, _ = first_hit_dp(collision.unsqueeze(0), a_i.unsqueeze(0), a_j.unsqueeze(0), time_steps)
    return risks.squeeze(0)


def survival_at(f: torch.Tensor) -> torch.Tensor:
    """给定 F 张量返回存活总质量（诊断用）。"""
    return f.sum(dim=(-2, -1))


__all__ = ["first_hit_dp", "first_hit_risk_single", "survival_at"]
