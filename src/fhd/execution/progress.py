"""随机执行进度模型（文档 §3）：软/硬门槛、单机器人进度传播、完成时间代价。

全部时间为**执行时隙**单位；物理时间 = 时隙数 × Δt（默认 0.4 s）。

转移概率（软门槛，批复 §11 的偏移 c）：
    a_{i,m}(t) = (1 - p_i) * sigma((t + 0.5 + c - tau_{i,m}) / beta)
硬门槛：
    a_{i,m}(t) = (1 - p_i) * 1[t >= tau_{i,m}]
二者统一比较**时隙起点 t**（批复 §12）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def release_times(wait: torch.Tensor, h: int | None = None) -> torch.Tensor:
    """由等待预算得到最早放行时隙 tau_{i,m} = m + sum_{h<=m} w_{i,h}。

    wait [N, H-1]（非负，单位：时隙）-> tau [N, H-1]
    """
    if wait.dim() != 2:
        raise ValueError("wait 必须是 [N, H-1]")
    h1 = wait.shape[-1]
    if h is not None and h - 1 != h1:
        raise ValueError(f"wait 的列数应为 H-1={h - 1}，实际 {h1}")
    steps = torch.arange(h1, dtype=wait.dtype, device=wait.device).unsqueeze(0)
    return steps + torch.cumsum(wait, dim=-1)


def release_times_with_durations(wait: torch.Tensor,
                                 durations: torch.Tensor | None = None) -> torch.Tensor:
    """带**段通行时长**的释放时刻。

    durations [N, H-1]：第 j 段的实际通行时长（单位：时隙），默认全 1 = 原模型。
    tau_m = (到达航点 m 的累计通行时间) + sum_{h<=m} w_h

    这是 M1/M2 的接口：只改进度转移的**时间尺度**，不改首次碰撞 DP 的数学结构。
    """
    if wait.dim() != 2:
        raise ValueError("wait 必须是 [N, H-1]")
    n, h1 = wait.shape
    if durations is None:
        return release_times(wait)
    if durations.shape != wait.shape:
        raise ValueError(f"durations 形状应为 {tuple(wait.shape)}，实际 {tuple(durations.shape)}")
    zero = torch.zeros(n, 1, dtype=wait.dtype, device=wait.device)
    time_to_wp = torch.cat([zero, torch.cumsum(durations, dim=-1)], dim=-1)   # [N,H]
    return time_to_wp[:, :-1] + torch.cumsum(wait, dim=-1)


def move_probabilities(
    tau: torch.Tensor,
    pause_p: torch.Tensor,
    time_steps: int,
    beta: float = 0.25,
    offset: float = 0.5,
    gate_mode: str = "soft",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """前进概率 a_{i,m}(t)，形状 [N, T, H]，最后一列（终点前进）恒为 0。

    tau [N, H-1]，pause_p [N]。
    """
    dtype = dtype or tau.dtype
    tau = tau.to(dtype)
    pause_p = pause_p.to(dtype).reshape(-1, 1)
    n, h1 = tau.shape
    t = torch.arange(time_steps, dtype=dtype, device=tau.device).view(1, time_steps, 1)
    tau3 = tau.unsqueeze(1)  # [N,1,H-1]
    if gate_mode == "soft":
        gate = torch.sigmoid((t + 0.5 + offset - tau3) / beta)
    elif gate_mode == "hard":
        gate = (t >= tau3).to(dtype)
    else:
        raise ValueError(f"未知 gate_mode: {gate_mode}")
    a = (1.0 - pause_p).unsqueeze(1) * gate  # [N,T,H-1]
    zeros = a.new_zeros(n, time_steps, 1)
    return torch.cat([a, zeros], dim=-1)  # [N,T,H]


def propagate_progress(a: torch.Tensor) -> torch.Tensor:
    """单机器人进度分布传播：pi_{t+1}(m+1) += pi_t(m) a_{i,m}(t)，终点吸收。

    a [N,T,H] -> pi [N,T+1,H]，每个 t 上 pi[:,t,:] 之和恒为 1。
    """
    n, t_steps, h = a.shape
    dtype, device = a.dtype, a.device
    pi = torch.zeros(n, h, dtype=dtype, device=device)
    pi[:, 0] = 1.0
    out = [pi]
    for t in range(t_steps):
        at = a[:, t, :]  # [N,H]
        stay = pi * (1.0 - at)
        advance = F.pad(pi * at, (1, 0))[:, :h]  # 右移一格
        pi = stay + advance
        out.append(pi)
    return torch.stack(out, dim=1)  # [N,T+1,H]


def finish_cost(
    pi: torch.Tensor,
    dt_s: float,
    lambda_tail_s: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """完成时间代价 J_finish（文档 §4.5，单位：秒）。

    J = sum_i [ dt * sum_{t=0}^{T-1} (1 - pi_i^t(H-1)) + lambda_tail * (1 - pi_i^T(H-1)) ]
    返回 (J 标量, 每台机器人的未完成概率 [N])。
    """
    t_steps = pi.shape[1] - 1
    arrived = pi[:, :, -1]  # [N,T+1]
    expected = dt_s * (1.0 - arrived[:, :t_steps]).sum(dim=-1)  # [N]
    tail = lambda_tail_s * (1.0 - arrived[:, t_steps])
    per_agent = expected + tail
    return per_agent.sum(), 1.0 - arrived[:, t_steps]


def expected_completion_time_steps(pi: torch.Tensor, dt_s: float) -> torch.Tensor:
    """可微的期望完成时间（秒），仅作诊断量。"""
    t_steps = pi.shape[1] - 1
    return dt_s * (1.0 - pi[:, :t_steps, -1]).sum(dim=-1)


def hard_progress_trajectory(tau_row: torch.Tensor, time_steps: int) -> list[int]:
    """给定时隙门槛，返回无暂停时的确定性进度序列（用于测试与参考轨迹）。"""
    h1 = int(tau_row.numel())
    z = 0
    seq = [z]
    for t in range(time_steps):
        if z < h1 and float(tau_row[z]) <= t:
            z += 1
        seq.append(z)
    return seq
