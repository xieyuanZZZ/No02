"""R8 梯度正确性：与中心差分对照（float64、远离硬边界、梯度非零处）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fhd.maps.scenarios import crossing
from fhd.risk.multi_pair import FirstHitRisk, team_union_bound
from fhd.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml")
D_SAFE = 2.0 * float(CFG["world"]["robot_radius_m"]) + float(CFG["world"]["safety_margin_m"])


def _model(**kwargs) -> FirstHitRisk:
    params = dict(d_safe=D_SAFE, beta=0.25, offset=0.5, temperature_m=0.025, kernel="logistic", dt_s=0.4,
                  lambda_tail_s=2.0)
    params.update(kwargs)
    return FirstHitRisk(**params)


def _risk(paths, wait, p, t_steps, model):
    out = model(
        paths, wait, p, time_steps=t_steps, collision_mode="soft", gate_mode="soft",
        return_diagnostics=False,
    )
    return out["pair_sum_risk"]


@pytest.fixture(scope="module")
def setup():
    sc = crossing(CFG, h=24, offset=0.35)
    paths = sc.paths.clone().requires_grad_(True)
    wait = torch.zeros(2, 23, dtype=torch.float64)
    wait[0, 8:14] = torch.linspace(0.0, 0.8, 6, dtype=torch.float64)
    wait = wait.clone().requires_grad_(True)
    p = torch.tensor([0.1, 0.1], dtype=torch.float64)
    return paths, wait, p


def test_risk_is_nonzero_and_finite(setup):
    paths, wait, p = setup
    model = _model()
    value = _risk(paths, wait, p, 96, model)
    assert torch.isfinite(value)
    assert 1e-3 < float(value.detach()) < 0.999, f"梯度测试需要非平凡风险值，实际 {float(value.detach())}"


def test_gradient_wrt_paths_matches_finite_difference(setup):
    """非端点坐标的中心差分对照（相对误差 <= 1e-3）。"""
    paths, wait, p = setup
    model = _model()
    pi = paths.detach().clone().requires_grad_(True)
    value = _risk(pi, wait, p, 96, model)
    (g_paths,) = torch.autograd.grad(value, pi)
    assert torch.isfinite(g_paths).all()

    h = 1e-6
    coords = [(0, 12, 1), (0, 18, 1), (1, 10, 0), (1, 20, 0)]
    with torch.no_grad():
        for (agent, idx, axis) in coords:
            base = pi.detach().clone()
            plus = base.clone()
            minus = base.clone()
            plus[agent, idx, axis] += h
            minus[agent, idx, axis] -= h
            f_plus = float(_risk(plus, wait, p, 96, model))
            f_minus = float(_risk(minus, wait, p, 96, model))
            fd = (f_plus - f_minus) / (2 * h)
            ana = float(g_paths[agent, idx, axis])
            denom = max(abs(fd), abs(ana), 1e-8)
            rel = abs(fd - ana) / denom
            assert rel <= 1e-3 or abs(fd - ana) <= 1e-8, (
                f"坐标 {(agent, idx, axis)} 梯度不一致: 解析={ana:.8e} 差分={fd:.8e} 相对误差={rel:.2e}"
            )


def test_gradient_wrt_wait_matches_finite_difference(setup):
    paths, wait, p = setup
    model = _model()
    w = wait.detach().clone().requires_grad_(True)
    value = _risk(paths, w, p, 96, model)
    (g_wait,) = torch.autograd.grad(value, w)
    assert torch.isfinite(g_wait).all()
    assert g_wait.abs().sum() > 0, "等待变量必须收到非零梯度"

    h = 1e-6
    coords = [(0, 9), (0, 11), (1, 12)]
    with torch.no_grad():
        for (agent, idx) in coords:
            base = w.detach().clone()
            plus, minus = base.clone(), base.clone()
            plus[agent, idx] += h
            minus[agent, idx] -= h
            fd = (float(_risk(paths, plus, p, 96, model)) - float(_risk(paths, minus, p, 96, model))) / (2 * h)
            ana = float(g_wait[agent, idx])
            denom = max(abs(fd), abs(ana), 1e-8)
            rel = abs(fd - ana) / denom
            assert rel <= 1e-3 or abs(fd - ana) <= 1e-8, (
                f"等待 {(agent, idx)} 梯度不一致: 解析={ana:.8e} 差分={fd:.8e} 相对误差={rel:.2e}"
            )


def test_gradient_has_no_nan_across_modes(setup):
    """四类组合的梯度行为：
    * collision=soft -> 对路径有梯度；gate=soft -> 对等待有梯度；
    * 两者皆 hard -> 完全不可微（这正是精确概率与可微代理的分工）。
    """
    paths, wait, p = setup
    for collision in ("soft", "hard"):
        for gate in ("soft", "hard"):
            pi = paths.detach().clone().requires_grad_(True)
            w = wait.detach().clone().requires_grad_(True)
            out = _model()(pi, w, p, time_steps=96, collision_mode=collision, gate_mode=gate,
                           return_diagnostics=False)
            loss = out["pair_sum_risk"] + out["finish_cost"]
            assert torch.isfinite(loss), f"{collision}/{gate} 前向出现 NaN/Inf"
            if collision == "hard" and gate == "hard":
                assert not loss.requires_grad, "hard/hard 应完全不可微"
                continue
            g_paths, g_wait = torch.autograd.grad(loss, (pi, w), allow_unused=True)
            if collision == "soft":
                assert g_paths is not None and torch.isfinite(g_paths).all(), (
                    f"{collision}/{gate} 对路径的梯度异常"
                )
            else:
                assert g_paths is None, "collision=hard 不应产生路径梯度"
            if gate == "soft":
                assert g_wait is not None and torch.isfinite(g_wait).all(), (
                    f"{collision}/{gate} 对等待的梯度异常"
                )
            else:
                assert g_wait is None, "gate=hard 不应产生等待梯度"


def test_gradient_flows_to_endpoints_too(setup):
    """端点也参与碰撞（终点占据），因此风险图应当覆盖端点；屏蔽在采样器中做。"""
    paths, wait, p = setup
    pi = paths.detach().clone().requires_grad_(True)
    value = _risk(pi, wait, p, 96, _model())
    (g,) = torch.autograd.grad(value, pi)
    assert g[:, 0, :].abs().sum() > 0 or g[:, -1, :].abs().sum() > 0


def test_union_bound():
    risks = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
    assert float(team_union_bound(risks)) == pytest.approx(1.0)
    risks2 = torch.tensor([0.1, 0.2], dtype=torch.float64)
    assert float(team_union_bound(risks2)) == pytest.approx(0.3)
