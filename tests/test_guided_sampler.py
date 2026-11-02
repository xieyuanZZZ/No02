"""S4 接口测试：引导采样器的行为与约束（用微型模型，不需要训练）。"""

from __future__ import annotations

from pathlib import Path

import torch

from fhd.diffusion.ddim import ddim_sample
from fhd.diffusion.guided_ddim import GuidanceConfig, guided_ddim_sample
from fhd.diffusion.model import ConditionalUNet1D
from fhd.diffusion.schedule import DDPMSchedule
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "guidance.yaml")


def _setup(h: int = 16, grid_size: int = 16):
    torch.manual_seed(0)
    model = ConditionalUNet1D(base_channels=16, mults=(1, 2, 4), cond_dim=32,
                              map_feature_dim=32, grid_size=grid_size).eval()
    model.horizon = h
    for p in model.parameters():
        p.requires_grad_(False)
    sch = DDPMSchedule(80)
    risk = FirstHitRisk(d_safe=0.35)
    n = 2
    start = torch.tensor([[0.6, 4.5], [5.5, 0.6]], dtype=torch.float32)
    goal = torch.tensor([[9.4, 4.5], [5.5, 9.4]], dtype=torch.float32)
    grid = torch.zeros(n, 1, grid_size, grid_size)
    pause = torch.full((n,), 0.1, dtype=torch.float32)
    return model, sch, risk, start, goal, grid, pause


def test_guided_sampler_shapes_and_constraints():
    model, sch, risk, start, goal, grid, pause = _setup()
    cfg = CFG
    cfg["trajectory"]["horizon"] = 16
    cfg["execution"]["horizon_steps"] = 12
    guide = GuidanceConfig(enabled=True, start_fraction=0.0, step_size_x=0.05,
                           step_size_w=0.1, max_wait_per_point=4.0, max_total_wait=8.0,
                           wait_mask_mode="all", hard_eval_every=2)
    out = guided_ddim_sample(model, sch, risk, cfg, start, goal, grid, pause,
                             obstacles=None, num_steps=6, guide=guide,
                             time_steps=12, dt_s=0.4, d_safe=0.35,
                             x_init=torch.randn(2, 16, 2))
    assert out.paths_m.shape == (2, 16, 2)
    assert torch.isfinite(out.paths_m).all()
    assert out.wait.shape == (2, 15)
    assert (out.wait >= -1e-6).all()
    assert float(out.wait.max()) <= 4.0 + 1e-6
    assert float(out.wait.sum(-1).max()) <= 8.0 + 1e-3
    assert torch.allclose(out.paths_m[:, 0, :], start, atol=1e-4)
    assert torch.allclose(out.paths_m[:, -1, :], goal, atol=1e-4)
    assert len(out.history) == len(sch.ddim_timesteps(6))
    assert out.final["hard_risk_sum"] >= 0.0


def test_guided_disabled_equals_base_ddim():
    model, sch, risk, start, goal, grid, pause = _setup()
    cfg = CFG
    cfg["trajectory"]["horizon"] = 16
    x_init = torch.randn(2, 16, 2)
    scale = torch.tensor([5.0, 5.0])
    start_n = start / scale - 1.0
    goal_n = goal / scale - 1.0
    base = ddim_sample(model, sch, start_n, goal_n, grid, num_steps=5, x_init=x_init)
    guide = GuidanceConfig(enabled=False, max_wait_per_point=4.0, max_total_wait=8.0)
    out = guided_ddim_sample(model, sch, risk, cfg, start, goal, grid, pause, obstacles=None,
                             num_steps=5, guide=guide, time_steps=12, dt_s=0.4,
                             x_init=x_init)
    guided_n = out.paths_m / scale - 1.0
    assert torch.allclose(base, guided_n, atol=1e-4)


def test_wait_mask_respected():
    model, sch, risk, start, goal, grid, pause = _setup()
    cfg = CFG
    cfg["trajectory"]["horizon"] = 16
    mask = torch.zeros(2, 15, dtype=torch.bool)
    mask[0, 5:8] = True
    mask[1, 2:4] = True
    guide = GuidanceConfig(enabled=True, start_fraction=0.0, step_size_w=1.0,
                           wait_mask_mode="manual", wait_mask=mask,
                           max_wait_per_point=4.0, max_total_wait=8.0)
    out = guided_ddim_sample(model, sch, risk, cfg, start, goal, grid, pause, obstacles=None,
                             num_steps=6, guide=guide, time_steps=12, dt_s=0.4,
                             x_init=torch.randn(2, 16, 2))
    assert float(out.wait.masked_fill(mask, 0.0).abs().max()) == 0.0, "掩码外不允许出现等待"


def test_auto_wait_mask_only_safe_points():
    from fhd.geometry.primitives import segment_point_distance
    from fhd.diffusion.guided_ddim import auto_wait_mask

    paths = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        [[0.0, 0.3], [1.0, 0.3], [2.0, 0.3], [3.0, 0.3], [4.0, 0.3]],
    ], dtype=torch.float32)
    clearance_ok = torch.ones(2, 5, dtype=torch.bool)
    mask = auto_wait_mask(paths, d_safe=0.35, clearance_ok=clearance_ok, n_candidates=2)
    # 两车路径相距 0.3 < 0.35，因此任何点都不满足“别人不会经过”，只能全开兜底
    assert mask.any()
    paths2 = paths.clone()
    paths2[1, :, 1] = 3.0
    mask2 = auto_wait_mask(paths2, d_safe=0.35, clearance_ok=clearance_ok, n_candidates=2)
    assert int(mask2.sum()) <= 4, "只应选出少量安全等待点"
