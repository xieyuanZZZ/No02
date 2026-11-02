"""S3/S4：扩散调度、条件 U-Net 与 DDIM 的等价性/端点测试。"""

from __future__ import annotations

import torch
import pytest

from fhd.diffusion.ddim import ddim_sample, project_endpoints
from fhd.diffusion.model import ConditionalUNet1D, count_parameters
from fhd.diffusion.schedule import DDPMSchedule


def small_model(h: int = 16, device="cpu") -> ConditionalUNet1D:
    m = ConditionalUNet1D(base_channels=16, mults=(1, 2, 4), cond_dim=32,
                          map_feature_dim=32, grid_size=16).to(device)
    m.horizon = h
    return m


def test_schedule_monotone_and_roundtrip():
    sch = DDPMSchedule(200, kind="cosine")
    ab = sch.alphas_cumprod
    assert torch.all(ab[1:] < ab[:-1]), "alpha_bar 必须单调下降"
    assert 0.99 < float(ab[0]) <= 1.0
    assert float(ab[-1]) < 0.01
    x0 = torch.randn(4, 16, 2)
    noise = torch.randn_like(x0)
    t = torch.full((4,), 100, dtype=torch.long)
    x_t = sch.q_sample(x0, t, noise)
    eps = sch.predict_eps(x_t, x0, t)
    assert torch.allclose(eps, noise, atol=1e-5), "predict_eps 应能精确还原噪声"
    assert torch.allclose(sch.predict_x0(x_t, eps, t), x0, atol=1e-5)


def test_model_forward_backward():
    m = small_model()
    x = torch.randn(3, 16, 2)
    t = torch.tensor([0, 50, 199])
    start = torch.randn(3, 2)
    goal = torch.randn(3, 2)
    grid = torch.zeros(3, 1, 16, 16)
    out = m(x, t, start, goal, grid)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert grad_ok
    assert count_parameters(m) > 0


def test_conditional_dependence():
    """不同条件（起终点/地图）必须给出不同的预测，否则条件注入失效。"""
    m = small_model().eval()
    x = torch.randn(1, 16, 2)
    t = torch.tensor([10])
    start = torch.zeros(1, 2)
    goal = torch.ones(1, 2) * 0.5
    grid_a = torch.zeros(1, 1, 16, 16)
    grid_b = torch.zeros(1, 1, 16, 16)
    grid_b[:, :, 4:8, 4:8] = 1.0
    with torch.no_grad():
        out_a = m(x, t, start, goal, grid_a)
        out_b = m(x, t, start + 0.3, goal, grid_a)
        out_c = m(x, t, start, goal, grid_b)
    assert not torch.allclose(out_a, out_b, atol=1e-6), "起点条件未生效"
    assert not torch.allclose(out_a, out_c, atol=1e-6), "地图条件未生效"


def test_ddim_zero_guidance_equivalence():
    """引导系数为零时，带回调的 DDIM 必须与基础 DDIM 逐步一致（批复 §S4 必须首先验证）。"""
    torch.manual_seed(0)
    m = small_model().eval()
    sch = DDPMSchedule(100, kind="cosine")
    b, h = 4, 16
    start = torch.rand(b, 2)
    goal = torch.rand(b, 2)
    grid = torch.zeros(b, 1, 16, 16)
    x_init = torch.randn(b, h, 2)

    base = ddim_sample(m, sch, start, goal, grid, num_steps=10, x_init=x_init)
    called = {"n": 0}

    def zero_guide(step, t, x_t, x0):
        called["n"] += 1
        return x0

    guided = ddim_sample(m, sch, start, goal, grid, num_steps=10, x_init=x_init,
                         guide_fn=zero_guide)
    assert called["n"] == 10
    assert torch.allclose(base, guided, atol=1e-7), "零引导必须与基础 DDIM 完全一致"


def test_ddim_endpoints_fixed():
    m = small_model().eval()
    sch = DDPMSchedule(100)
    start = torch.rand(5, 2)
    goal = torch.rand(5, 2)
    grid = torch.zeros(5, 1, 16, 16)
    out = ddim_sample(m, sch, start, goal, grid, num_steps=8,
                      x_init=torch.randn(5, 16, 2))
    assert torch.allclose(out[:, 0, :], start, atol=1e-6)
    assert torch.allclose(out[:, -1, :], goal, atol=1e-6)


def test_project_endpoints():
    x = torch.zeros(2, 5, 2)
    s = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    g = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    out = project_endpoints(x, s, g)
    assert torch.allclose(out[:, 0, :], s) and torch.allclose(out[:, -1, :], g)
    assert torch.allclose(out[:, 1:-1, :], torch.zeros_like(out[:, 1:-1, :]))


def test_ddim_step_last_step_is_x0():
    """最后一步应直接输出 x0（避免 alpha_bar≈1 时的病态除法）。"""
    sch = DDPMSchedule(100)
    x_t = torch.randn(2, 8, 2)
    x0 = torch.randn(2, 8, 2)
    out = sch.ddim_step(x_t, x0, t=10, t_next=-1)
    assert torch.allclose(out, x0, atol=1e-5)
