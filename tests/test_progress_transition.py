"""S1 随机执行进度模型测试：门槛、概率守恒、单机分布与暴力枚举一致。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_dp import brute_force_progress_pmf  # noqa: E402

from fhd.execution.progress import (  # noqa: E402
    finish_cost,
    hard_progress_trajectory,
    move_probabilities,
    propagate_progress,
    release_times,
)


def test_release_times_semantics():
    """tau_{i,m} = m + sum_{h<=m} w_{i,h}，且非降。"""
    wait = torch.tensor([[0.0, 2.0, 0.0, 1.0]], dtype=torch.float64)
    tau = release_times(wait)
    # cumsum = [0,2,2,3] -> tau = [0, 3, 4, 6]
    expected = torch.tensor([[0.0, 3.0, 4.0, 6.0]], dtype=torch.float64)
    assert torch.allclose(tau, expected)
    assert torch.all(tau[0, 1:] >= tau[0, :-1])


def test_move_probability_in_range_and_terminal_zero():
    wait = torch.zeros(2, 4, dtype=torch.float64)
    tau = release_times(wait)
    p = torch.tensor([0.1, 0.3], dtype=torch.float64)
    for gate in ("soft", "hard"):
        a = move_probabilities(tau, p, time_steps=8, beta=0.25, offset=0.5, gate_mode=gate)
        assert a.shape == (2, 8, 5)
        assert a.min() >= 0.0 and a.max() <= 1.0
        assert torch.all(a[:, :, -1] == 0), "终点前进概率必须恒为 0"


def test_hard_gate_is_deterministic_ramp():
    """p=0、无等待时，硬门槛在 t=m 允许前进，进度恒等于 t。"""
    h, t_steps = 5, 8
    wait = torch.zeros(1, h - 1, dtype=torch.float64)
    tau = release_times(wait)
    a = move_probabilities(tau, torch.zeros(1, dtype=torch.float64), t_steps, gate_mode="hard")
    pi = propagate_progress(a)
    z = pi[0].argmax(dim=-1)
    assert z.tolist() == [min(t, h - 1) for t in range(t_steps + 1)]
    assert torch.allclose(pi[0, -1], torch.eye(h, dtype=torch.float64)[-1])


def test_progress_mass_conservation():
    torch.manual_seed(0)
    wait = torch.rand(3, 7, dtype=torch.float64) * 2
    p = torch.rand(3, dtype=torch.float64) * 0.3
    tau = release_times(wait)
    for gate in ("soft", "hard"):
        a = move_probabilities(tau, p, 20, gate_mode=gate)
        pi = propagate_progress(a)
        sums = pi.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12), f"{gate} 概率不守恒"
        assert (pi >= -1e-12).all()


def test_wait_delays_release():
    """在安全点等待会让该点及其后续进度的放行时间整体推迟，之前不受影响。"""
    h = 8
    base = torch.zeros(1, h - 1, dtype=torch.float64)
    delayed = base.clone()
    delayed[0, 3] = 2.0
    tau0 = release_times(base)
    tau1 = release_times(delayed)
    assert torch.allclose(tau0[0, :3], tau1[0, :3], atol=1e-12)  # 等待点之前不受影响
    assert torch.allclose(tau1[0, 3:] - tau0[0, 3:], torch.full((4,), 2.0, dtype=torch.float64))


def test_pmf_matches_bruteforce_hard():
    """单机进度分布与独立暴力枚举逐项一致（硬门槛）。"""
    h, t_steps = 4, 6
    wait = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    p = torch.tensor([0.25], dtype=torch.float64)
    a = move_probabilities(release_times(wait), p, t_steps, gate_mode="hard")
    pi = propagate_progress(a)[0]  # [T+1,H]
    ref = brute_force_progress_pmf(wait[0].tolist(), 0.25, t_steps, h, gate_mode="hard")
    ref_t = torch.tensor(ref, dtype=torch.float64)
    assert torch.allclose(pi, ref_t, atol=1e-12)


def test_pmf_matches_bruteforce_soft():
    """单机进度分布与暴力枚举一致（软门槛，默认偏移 c=0.5）。"""
    h, t_steps = 4, 6
    wait = torch.tensor([[0.5, 0.0, 1.5]], dtype=torch.float64)
    p = torch.tensor([0.2], dtype=torch.float64)
    beta, offset = 0.25, 0.5
    a = move_probabilities(release_times(wait), p, t_steps, beta=beta, offset=offset, gate_mode="soft")
    pi = propagate_progress(a)[0]
    ref = brute_force_progress_pmf(
        wait[0].tolist(), 0.2, t_steps, h, gate_mode="soft", beta=beta, offset=offset
    )
    assert torch.allclose(pi, torch.tensor(ref, dtype=torch.float64), atol=1e-10)


def test_soft_gate_offset_reduces_zero_wait_bias():
    """批复 §11：偏移 c=0.5 使 w=0 时前进概率接近硬门槛的 1-p，而不是 0.881(1-p)。"""
    h, t_steps = 4, 6
    wait = torch.zeros(2, h - 1, dtype=torch.float64)
    tau = release_times(wait)
    p = torch.tensor([0.1, 0.1], dtype=torch.float64)
    hard = move_probabilities(tau, p, t_steps, gate_mode="hard")
    soft_c0 = move_probabilities(tau, p, t_steps, beta=0.25, offset=0.0, gate_mode="soft")
    soft_c05 = move_probabilities(tau, p, t_steps, beta=0.25, offset=0.5, gate_mode="soft")
    # t = m 时刻：硬门槛正好放行
    idx = torch.arange(h - 1)
    hard_at_release = hard[0, idx, idx]
    c0 = soft_c0[0, idx, idx]
    c05 = soft_c05[0, idx, idx]
    assert torch.allclose(hard_at_release, torch.full_like(hard_at_release, 0.9), atol=1e-12)
    assert float(c0.min()) < 0.85, "c=0 时零等待偏差异常（应约为 0.881*(1-p)）"
    assert float(c05.min()) > 0.95 * 0.9, "c=0.5 时应接近硬门槛"


def test_finish_cost_monotone_in_pause_probability():
    """暂停概率越大，期望完成时间代价越高。"""
    h, t_steps = 8, 40
    wait = torch.zeros(1, h - 1, dtype=torch.float64)
    tau = release_times(wait)
    costs = []
    for p in (0.0, 0.1, 0.3, 0.6):
        a = move_probabilities(tau, torch.tensor([p], dtype=torch.float64), t_steps, gate_mode="hard")
        pi = propagate_progress(a)
        j, _ = finish_cost(pi, dt_s=0.4, lambda_tail_s=2.0)
        costs.append(float(j))
    assert costs[0] < costs[1] < costs[2] < costs[3], costs


def test_hard_progress_trajectory_helper():
    """tau 长度为 H-1=4（即 H=5），p=0 时每个时隙前进一格直到终点。"""
    tau = release_times(torch.zeros(1, 4, dtype=torch.float64))[0]
    seq = hard_progress_trajectory(tau, 8)
    assert seq == [0, 1, 2, 3, 4, 4, 4, 4, 4]
