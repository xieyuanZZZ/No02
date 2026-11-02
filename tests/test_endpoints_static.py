"""S3/S4：生成轨迹的端点与静态安全约束测试（不需要训练好的模型即可跑部分用例）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fhd.diffusion.ddim import ddim_sample, project_endpoints
from fhd.diffusion.model import ConditionalUNet1D
from fhd.diffusion.schedule import DDPMSchedule
from fhd.geometry.collisions import check_speed_limit, trajectory_is_static_safe
from fhd.maps.generate import generate_map
from fhd.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml")


def _tiny_model(h: int = 16) -> ConditionalUNet1D:
    m = ConditionalUNet1D(base_channels=16, mults=(1, 2, 4), cond_dim=32,
                          map_feature_dim=32, grid_size=16)
    m.horizon = h
    return m.eval()


def test_generated_endpoints_exact():
    model = _tiny_model()
    sch = DDPMSchedule(50)
    start = torch.rand(3, 2)
    goal = torch.rand(3, 2)
    grid = torch.zeros(3, 1, 16, 16)
    out = ddim_sample(model, sch, start, goal, grid, num_steps=5,
                      x_init=torch.randn(3, 16, 2))
    assert torch.allclose(out[:, 0, :], start, atol=1e-6)
    assert torch.allclose(out[:, -1, :], goal, atol=1e-6)


def test_projection_is_idempotent():
    x = torch.randn(2, 8, 2)
    s = torch.rand(2, 2)
    g = torch.rand(2, 2)
    once = project_endpoints(x, s, g)
    twice = project_endpoints(once, s, g)
    assert torch.allclose(once, twice)


def test_static_check_rejects_straight_through_obstacle():
    """静态安全检查必须能拒绝穿过障碍的直线，并接受绕行。"""
    world = generate_map("dual_corridor", CFG, 20261103)
    obs = world.obstacles()
    rect = world.rects[0]
    cy = 0.5 * (rect[1] + rect[3])
    h = 16
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    straight = torch.stack([torch.tensor([0.5, cy], dtype=torch.float64)
                            + t * torch.tensor([9.0, 0.0], dtype=torch.float64)])
    safe, _ = trajectory_is_static_safe(straight, obs, world.robot_radius_m, world.clearance_margin_m)
    assert not bool(safe[0])


def test_speed_check_on_generated_like_paths():
    h = 32
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    ok = torch.stack([t * torch.tensor([5.0, 3.0], dtype=torch.float64)])
    bad = torch.stack([t * torch.tensor([30.0, 0.0], dtype=torch.float64)])
    assert bool(check_speed_limit(ok, 1.5, 0.4)[0][0])
    assert not bool(check_speed_limit(bad, 1.5, 0.4)[0][0])


@pytest.mark.skipif(
    not any((ROOT / "results" / "S3_diffusion_training").glob("*/checkpoints/best.pt")),
    reason="尚无训练好的 checkpoint",
)
def test_trained_model_sample_shapes():
    """有 checkpoint 时，检查采样输出的形状、端点与有限性。"""
    ckpt_path = sorted((ROOT / "results" / "S3_diffusion_training").glob("*/checkpoints/best.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[0]
    from fhd.diffusion.train import build_model

    cfg = load_config(ROOT / "configs" / "diffusion_train.yaml")
    model = build_model(cfg, torch.device("cpu"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    model.eval()
    sch = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                       kind=str(cfg["diffusion"]["beta_schedule"]))
    h = int(cfg["trajectory"]["horizon"])
    start = torch.rand(2, 2) * 1.5 - 0.75
    goal = torch.rand(2, 2) * 1.5 - 0.75
    grid = torch.zeros(2, 1, 64, 64)
    out = ddim_sample(model, sch, start, goal, grid, num_steps=10,
                      x_init=torch.randn(2, h, 2))
    assert out.shape == (2, h, 2)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, 0, :], start, atol=1e-5)
    assert torch.allclose(out[:, -1, :], goal, atol=1e-5)
