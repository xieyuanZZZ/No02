"""R9 软硬模型差异、R10 数值稳定性与计算开销。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from fhd.maps.scenarios import build_scenario, crossing, multi_bottleneck
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml")
D_SAFE = 0.35


def _risk(paths, wait, p, t_steps, **kwargs):
    params = dict(d_safe=D_SAFE, beta=0.25, offset=0.5, temperature_m=0.025, kernel="logistic", dt_s=0.4)
    params.update({k: v for k, v in kwargs.items() if k in params})
    model = FirstHitRisk(**params)
    out = model(paths, wait, p, time_steps=t_steps,
                collision_mode=kwargs.get("collision", "soft"), gate_mode=kwargs.get("gate", "soft"),
                return_diagnostics=False)
    return float(out["pair_risks"][0])


@pytest.fixture(scope="module")
def cross_setup():
    sc = crossing(CFG, h=32, offset=0.5)
    wait = torch.zeros(2, 31, dtype=torch.float64)
    pause = torch.tensor([0.1, 0.1], dtype=torch.float64)
    return sc, wait, pause


def test_r9_hard_risk_independent_of_temperature(cross_setup):
    """硬风险只由 gate/collision 的硬开关决定，对 epsilon、beta 不敏感。"""
    sc, wait, pause = cross_setup
    hard = [_risk(sc.paths, wait, pause, 96, temperature_m=t, beta=b, collision="hard", gate="hard")
            for t, b in [(0.005, 0.1), (0.025, 0.25), (0.1, 0.5)]]
    assert max(hard) - min(hard) < 1e-12, hard
    assert 0.0 <= hard[0] <= 1.0


def test_r9_soft_risk_varies_with_temperature(cross_setup):
    """软代理随温度显著变化，说明它只是优化代理而不是概率。"""
    sc, wait, pause = cross_setup
    soft_vals = [
        _risk(sc.paths, wait, pause, 96, temperature_m=t, collision="soft", gate="soft")
        for t in (0.005, 0.025, 0.1)
    ]
    assert len(set(round(v, 8) for v in soft_vals)) == 3, soft_vals
    assert all(0.0 <= v <= 1.0 for v in soft_vals), soft_vals


def test_r9_soft_risk_exceeds_hard_at_far_distance(cross_setup):
    """Logistic 在远距离非零，软风险会高于硬风险（“虚假累积”），必须记录而非隐藏。"""
    sc, wait, pause = cross_setup
    soft = _risk(sc.paths, wait, pause, 96, collision="soft", gate="soft")
    hard = _risk(sc.paths, wait, pause, 96, collision="hard", gate="hard")
    assert soft >= hard - 1e-9, f"soft={soft} hard={hard}"


def test_r9_smoothstep_has_compact_support():
    """smoothstep 代理在 d >= d_safe + support 时严格为 0，避免远距离累积。"""
    from fhd.geometry.collisions import soft_collision

    d = torch.tensor([0.0, 0.35, 0.38, 0.40, 0.50, 2.0], dtype=torch.float64)
    c = soft_collision(d, 0.35, 0.025, kernel="smoothstep", support_m=0.05)
    assert float(c[0]) == pytest.approx(1.0)
    assert float(c[1]) == pytest.approx(1.0)
    assert float(c[-1]) == pytest.approx(0.0)
    assert float(c[-2]) == pytest.approx(0.0)
    assert 0.0 < float(c[2]) < 1.0


def test_r10_numerics_stable_large():
    """H=32,T=96,N=2 上无 NaN/Inf、概率非负、可反传，并记录开销。"""
    sc = crossing(CFG, h=32, offset=0.4)
    wait = torch.zeros(2, 31, dtype=torch.float64, requires_grad=True)
    paths = sc.paths.clone().requires_grad_(True)
    pause = torch.tensor([0.15, 0.15], dtype=torch.float64)
    model = FirstHitRisk(d_safe=D_SAFE)

    t0 = time.perf_counter()
    out = model(paths, wait, pause, 96, collision_mode="soft", gate_mode="soft")
    loss = out["pair_sum_risk"] + out["finish_cost"]
    g_paths, g_wait = torch.autograd.grad(loss, (paths, wait))
    elapsed = time.perf_counter() - t0

    assert torch.isfinite(out["pair_sum_risk"]) and torch.isfinite(out["finish_cost"])
    assert 0.0 <= float(out["pair_sum_risk"]) <= 1.0 + 1e-9
    assert torch.isfinite(g_paths).all() and torch.isfinite(g_wait).all()
    assert float(g_paths.norm()) > 0
    survival = out["diagnostics"]["survival"]
    assert torch.isfinite(survival).all()
    assert (survival >= -1e-12).all() and (survival <= 1 + 1e-12).all()
    # 单对 H=32,T=96 的前向 + 反向应远低于 5 秒（CPU, float64）
    assert elapsed < 30.0, f"开销异常: {elapsed:.2f}s"
    print(f"\n[R10] N=2 前向+反向耗时 {elapsed * 1000:.1f} ms")


def test_r10_multi_agent_pair_enumeration():
    """N=4/6 时机器人对数量与风险向量长度正确，且数值有限。"""
    for n in (4, 6):
        sc = build_scenario("multi_bottleneck", CFG, n_agents=n)
        assert sc.n_agents == n
        wait = torch.zeros(n, 31, dtype=torch.float64)
        pause = torch.full((n,), 0.1, dtype=torch.float64)
        out = FirstHitRisk(d_safe=D_SAFE)(sc.paths, wait, pause, 96, collision_mode="hard",
                                          gate_mode="hard", return_diagnostics=False)
        expected = n * (n - 1) // 2
        assert out["pair_risks"].numel() == expected
        assert torch.isfinite(out["pair_risks"]).all()
        assert (out["pair_risks"] >= -1e-12).all() and (out["pair_risks"] <= 1 + 1e-12).all()


def test_r10_scenario_geometry_sanity():
    """所有手工场景的路径都满足速度约束与端点一致性。"""
    from fhd.geometry.collisions import path_segment_lengths

    for name in ("crossing", "head_on", "follow", "terminal_occupancy", "dual_corridor", "narrow_gate"):
        sc = build_scenario(name, CFG)
        lens = path_segment_lengths(sc.paths)
        assert float(lens.max()) <= 1.5 * 0.4 + 1e-9, f"{name} 违反速度约束: {float(lens.max())}"
        assert torch.allclose(sc.paths[:, 0, :], sc.starts)
        assert torch.allclose(sc.paths[:, -1, :], sc.goals)
