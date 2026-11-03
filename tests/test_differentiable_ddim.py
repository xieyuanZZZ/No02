"""可微分 DDIM 的一致性、梯度与信赖域测试（S5 前置门槛）。

门槛规则（方案 §7.2）：
  1. 同一 checkpoint、同一初始噪声下，可微 DDIM 与普通 DDIM 输出必须一致；
  2. 对初始噪声的梯度必须有限、非零，且与有限差分一致；
  3. 网络权重不参与求导（梯度为空）。
不通过则不允许进入 S5 的噪声优化。
"""

from __future__ import annotations

import pytest
import torch

from fhd.diffusion.ddim import ddim_sample
from fhd.diffusion.differentiable_ddim import (
    ddim_sample_diff,
    noise_rms,
    project_endpoints_diff,
    project_trust_region,
)
from fhd.diffusion.model import ConditionalUNet1D
from fhd.diffusion.schedule import DDPMSchedule


def _model(h: int = 16, grid: int = 16) -> ConditionalUNet1D:
    torch.manual_seed(0)
    m = ConditionalUNet1D(base_channels=16, mults=(1, 2, 4), cond_dim=32,
                          map_feature_dim=32, grid_size=grid, map_spatial_channels=8,
                          map_res=8, map_conditioning="grid_sample")
    m.horizon = h
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _setup(h=16, grid=16, steps=60, double: bool = False):
    model = _model(h, grid)
    dtype = torch.float64 if double else torch.float32
    if double:
        model = model.double()
    sch = DDPMSchedule(steps, dtype=dtype)
    b = 2
    start = (torch.rand(b, 2, dtype=dtype) * 1.2 - 0.6)
    goal = (torch.rand(b, 2, dtype=dtype) * 1.2 - 0.6)
    g = torch.zeros(b, 1, grid, grid, dtype=dtype)
    z = torch.randn(b, h, 2, generator=torch.Generator().manual_seed(3), dtype=dtype)
    return model, sch, start, goal, g, z


def test_diff_matches_vanilla_ddim():
    model, sch, start, goal, grid, z = _setup()
    with torch.no_grad():
        vanilla = ddim_sample(model, sch, start, goal, grid, num_steps=8, x_init=z.clone())
        diff = ddim_sample_diff(model, sch, start, goal, grid, z.clone(), num_steps=8)
    assert torch.allclose(vanilla, diff.x0, atol=1e-5), \
        f"可微 DDIM 与普通 DDIM 不一致：max={float((vanilla - diff.x0).abs().max())}"


def test_gradient_flows_to_noise_only():
    model, sch, start, goal, grid, z = _setup()
    z = z.clone().requires_grad_(True)
    out = ddim_sample_diff(model, sch, start, goal, grid, z, num_steps=8)
    loss = out.x0.pow(2).sum()
    (grad,) = torch.autograd.grad(loss, z)
    assert torch.isfinite(grad).all(), "对初始噪声的梯度含 NaN/Inf"
    assert float(grad.abs().sum()) > 0, "梯度恒为零，说明计算图断了"
    assert all(p.grad is None for p in model.parameters()), "网络权重不应收到梯度"


def test_finite_difference_gradient():
    """中心差分校验 ∂loss/∂z：在 **float64** 下做，h=1e-6，要求相对误差 < 1e-3。"""
    model, sch, start, goal, grid, z0 = _setup(double=True)
    idx = [(0, 5, 0), (1, 10, 1), (0, 12, 1)]

    def loss_of(zz):
        out = ddim_sample_diff(model, sch, start, goal, grid, zz, num_steps=10)
        return out.x0.pow(2).sum()

    z = z0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(loss_of(z), z)
    h = 1e-6
    for (a, i, c) in idx:
        zp = z0.clone()
        zm = z0.clone()
        zp[a, i, c] += h
        zm[a, i, c] -= h
        fd = float((loss_of(zp) - loss_of(zm)) / (2 * h))
        ana = float(grad[a, i, c])
        rel = abs(fd - ana) / max(abs(fd), abs(ana), 1e-8)
        assert rel < 1e-3 or abs(fd - ana) < 1e-6, \
            f"z[{a},{i},{c}] 梯度不一致：解析 {ana:.6e} 差分 {fd:.6e} 相对误差 {rel:.2e}"


def test_endpoints_projection_diff():
    x = torch.randn(2, 6, 2)
    s = torch.rand(2, 2)
    g = torch.rand(2, 2)
    out = project_endpoints_diff(x, s, g)
    assert torch.allclose(out[:, 0, :], s) and torch.allclose(out[:, -1, :], g)
    assert torch.allclose(out[:, 1:-1, :], x[:, 1:-1, :])


def test_trust_region_projection():
    z0 = torch.zeros(2, 8, 2)
    z = torch.randn(2, 8, 2) * 5.0
    z = z.requires_grad_(True)
    proj = project_trust_region(z, z0, radius=0.1)
    assert noise_rms(proj, z0) <= 0.1 + 1e-5
    assert proj.requires_grad
