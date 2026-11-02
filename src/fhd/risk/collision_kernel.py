"""碰撞核：为每一对机器人预先向量化计算四种动作下的连续最小距离与碰撞量。

输出形状 [P, 4, H, H]：
    动作维 0=00（双方停）、1=10（i 前进）、2=01（j 前进）、3=11（双方前进）
    后两维是联合进度 (m, n)。
硬碰撞 C = 1[d_min < d_safe]；软碰撞按 logistic 或 smoothstep（文档 §4.2，批复 §14）。
"""

from __future__ import annotations

import torch

from fhd.geometry.collisions import (
    ACTION_MOVE_BOTH,
    ACTION_MOVE_I,
    ACTION_MOVE_J,
    ACTION_STOP_STOP,
    pair_action_min_distances,
    soft_collision,
)


def build_pair_index(n_agents: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """枚举所有机器人对 (i<j)。返回两个 [P] 的索引张量。"""
    ii, jj = [], []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            ii.append(i)
            jj.append(j)
    return (
        torch.tensor(ii, dtype=torch.long, device=device),
        torch.tensor(jj, dtype=torch.long, device=device),
    )


def pair_collision_tensors(
    paths: torch.Tensor,
    index_i: torch.Tensor,
    index_j: torch.Tensor,
    d_safe: float,
    temperature_m: float,
    kernel: str = "logistic",
    support_m: float = 0.05,
    mode: str = "soft",
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (d_min [P,4,H,H], C [P,4,H,H])。

    mode="soft" 时 C 是**可微代理**，不是真实碰撞概率；
    mode="hard" 时 C 是 0/1 指示，DP 结果才是精确首次碰撞概率。
    """
    if index_i.numel() == 0:
        n, h = paths.shape[0], paths.shape[1]
        empty = paths.new_zeros(n, 4, h, h)
        return empty, empty
    pi = paths.index_select(0, index_i)
    pj = paths.index_select(0, index_j)
    d_min = pair_action_min_distances(pi, pj).d_min
    if mode == "soft":
        c = soft_collision(d_min, d_safe, temperature_m, kernel=kernel, support_m=support_m)
    elif mode == "hard":
        c = (d_min < d_safe).to(d_min.dtype)
    else:
        raise ValueError(f"未知 collision_mode: {mode}")
    return d_min, c


__all__ = [
    "build_pair_index",
    "pair_collision_tensors",
    "ACTION_STOP_STOP",
    "ACTION_MOVE_I",
    "ACTION_MOVE_J",
    "ACTION_MOVE_BOTH",
]
