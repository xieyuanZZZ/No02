"""S0 连续碰撞检测测试：相对运动最小距离、对向换位、终点占用、静态障碍、速度约束。"""

from __future__ import annotations

import pytest
import torch

from fhd.execution.simulator import _relative_min_distance, dense_relative_min_distance
from fhd.geometry.collisions import (
    pair_action_min_distances,
    path_segment_lengths,
    check_speed_limit,
    trajectory_obstacle_clearance,
    trajectory_is_static_safe,
)
from fhd.geometry.primitives import (
    Obstacles,
    relative_motion_min_distance,
    segment_circle_distance,
    segment_rect_distance,
)
from fhd.maps.generate import generate_map
from fhd.utils.config import load_config

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml")
D_SAFE = 2.0 * float(CFG["world"]["robot_radius_m"]) + float(CFG["world"]["safety_margin_m"])


def test_d_safe_value():
    """文档 §2.1：d_safe = 2*0.15 + 0.05 = 0.35 m。"""
    assert D_SAFE == pytest.approx(0.35)


def test_relative_motion_matches_dense_sampling():
    """解析相对运动最小距离与加密采样一致（数百个随机用例）。"""
    torch.manual_seed(7)
    n = 500
    r0 = torch.randn(n, 2, dtype=torch.float64)
    v = torch.randn(n, 2, dtype=torch.float64) * 0.8
    d, u = relative_motion_min_distance(r0, v)
    d_ref = dense_relative_min_distance(r0, v, n_samples=4001)
    err = (d - d_ref).abs().max().item()
    assert err < 1e-4, f"相对运动最小距离与加密采样偏差过大: {err}"
    assert torch.all(u >= 0) and torch.all(u <= 1)


def test_simulator_kernel_matches_dp_kernel():
    """批复 §31：MC 与 DP 使用两套独立实现，必须互相一致。"""
    torch.manual_seed(11)
    n = 300
    r0 = torch.randn(n, 2, dtype=torch.float64)
    v = torch.randn(n, 2, dtype=torch.float64) * 0.5
    d_dp, _ = relative_motion_min_distance(r0, v)
    d_mc = _relative_min_distance(r0, v)
    d_brute = dense_relative_min_distance(r0, v, n_samples=4001)
    assert torch.allclose(d_dp, d_mc, atol=1e-12)
    assert (d_dp - d_brute).abs().max().item() < 1e-4


def test_stationary_degenerate():
    """双方都静止时退化为端点距离，不产生 NaN。"""
    r0 = torch.tensor([[0.1, 0.2], [3.0, 4.0]], dtype=torch.float64)
    v = torch.zeros_like(r0)
    d, u = relative_motion_min_distance(r0, v)
    assert torch.allclose(d, r0.norm(dim=-1))
    assert torch.all(u == 0)
    assert torch.isfinite(d).all()


def test_pair_action_swap_collision_r4():
    """R4：两机器人同一时隙对向交换位置，起终点安全但中途相撞。"""
    # 两条横向交叉的线段在 (1,0) 相交：起终点都相距 1.118 m，
    # 只有“双方同时前进”会在交点相撞。
    qi = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float64)
    qj = torch.tensor([[[1.0, -0.5], [1.0, 0.5]]], dtype=torch.float64)
    d = pair_action_min_distances(qi, qj).d_min  # [1,4,2,2]
    # 动作 11（双方前进）：相对位置在中点为零
    assert d[0, 3, 0, 0].item() < 0.02
    # 动作 00（双方静止）：端点相距 sqrt(1^2+0.5^2)
    assert d[0, 0, 0, 0].item() == pytest.approx((1.25) ** 0.5, abs=1e-9)
    # 只有一方前进时仍然安全（0.5 m 与 1.0 m）
    assert d[0, 1, 0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert d[0, 2, 0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_endpoint_only_check_would_miss():
    """对照实验：若只比较端点会漏掉交换碰撞。"""
    qi = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float64)
    qj = torch.tensor([[[1.0, -0.5], [1.0, 0.5]]], dtype=torch.float64)
    start_dist = torch.cdist(qi[0, 0].view(1, 2), qj[0, 0].view(1, 2)).item()
    end_dist = torch.cdist(qi[0, 1].view(1, 2), qj[0, 1].view(1, 2)).item()
    d = pair_action_min_distances(qi, qj).d_min[0, 3, 0, 0].item()
    assert start_dist > D_SAFE and end_dist > D_SAFE and d < D_SAFE


def test_segment_rect_distance_matches_dense():
    torch.manual_seed(3)
    n = 60
    a = torch.rand(n, 2, dtype=torch.float64) * 10
    b = torch.rand(n, 2, dtype=torch.float64) * 10
    rects = torch.tensor([[3.0, 3.0, 6.0, 6.0], [7.5, 1.0, 8.5, 4.0]], dtype=torch.float64)
    d = segment_rect_distance(a, b, rects, iterations=50)
    u = torch.linspace(0, 1, 4001, dtype=torch.float64)
    pts = a.unsqueeze(1) + u.view(1, -1, 1) * (b - a).unsqueeze(1)  # [n,4001,2]
    lo, hi = rects[:, :2], rects[:, 2:]
    excess = torch.clamp(
        torch.maximum(lo.view(1, 1, -1, 2) - pts.unsqueeze(2), pts.unsqueeze(2) - hi.view(1, 1, -1, 2)),
        min=0.0,
    )
    per_rect = (excess**2).sum(-1).sqrt()  # [n,4001,K]
    ref = per_rect.min(dim=2).values.min(dim=1).values
    assert (d - ref).abs().max().item() < 1e-3


def test_segment_circle_distance():
    a = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([[4.0, 0.0]], dtype=torch.float64)
    circles = torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.float64)
    d = segment_circle_distance(a, b, circles)
    assert d.item() == pytest.approx(0.5, abs=1e-9)


def test_terminal_occupancy_geometry_r5():
    """R5 几何层面：i 占据终点后，j 的路径穿越该点。"""
    qi = torch.tensor([[[2.6, 5.0], [5.0, 5.0]]], dtype=torch.float64)
    qj = torch.tensor([[[5.0, 4.2], [5.0, 5.8]]], dtype=torch.float64)
    d = pair_action_min_distances(qi, qj).d_min[0]  # [4,H,H]
    # 都停在起点：相距 2.4 m
    assert d[0, 0, 0].item() > D_SAFE
    # i 停在终点 (5,5)，j 从 (5,4.2) 移动到 (5,5.8)：必然经过 (5,5)
    assert d[3, 1, 0].item() < 1e-6
    assert d[2, 1, 0].item() < 1e-6
    # i 停在终点 (5,5)，j 也停在 (5,4.2)：相距 0.8 m，仍然安全
    assert d[0, 1, 0].item() == pytest.approx(0.8, abs=1e-9)


def test_static_obstacle_detection():
    world = generate_map("dual_corridor", CFG, 20261103)
    obstacles = world.obstacles()
    rect = world.rects[0]
    cx = 0.5 * (rect[0] + rect[2])
    cy = 0.5 * (rect[1] + rect[3])
    h = 16
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    straight = torch.stack([torch.tensor([0.5, cy], dtype=torch.float64) + t * torch.tensor([9.0, 0.0], dtype=torch.float64)])
    detour_y = rect[3] + 0.8
    way = torch.tensor([[0.5, cy], [cx - 1.0, detour_y], [cx + 1.0, detour_y], [9.5, cy]], dtype=torch.float64)
    from fhd.geometry.paths import resample_polyline

    detour = resample_polyline(way, h).unsqueeze(0)
    safe_straight, clr_straight = trajectory_is_static_safe(
        straight, obstacles, world.robot_radius_m, world.clearance_margin_m
    )
    safe_detour, clr_detour = trajectory_is_static_safe(
        detour, obstacles, world.robot_radius_m, world.clearance_margin_m
    )
    assert not bool(safe_straight[0]), "穿越障碍的直线必须被判为碰撞"
    assert clr_straight[0].item() < 0
    assert bool(safe_detour[0]), f"绕行路径应安全，最小净空={clr_detour[0].item():.4f}"
    assert clr_detour[0].item() > 0


def test_speed_limit_check():
    h = 10
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    ok_path = torch.stack([t * 3.0])  # 每段 0.333 m < 0.6
    bad_path = torch.stack([t * 20.0])  # 每段 2.22 m > 0.6
    ok_ok, max_len_ok = check_speed_limit(ok_path, 1.5, 0.4)
    ok_bad, max_len_bad = check_speed_limit(bad_path, 1.5, 0.4)
    assert bool(ok_ok[0]) and max_len_ok[0].item() == pytest.approx(3.0 / 9)
    assert not bool(ok_bad[0]) and max_len_bad[0].item() > 0.6


def test_obstacles_empty():
    obs = Obstacles()
    pts = torch.zeros(4, 2, dtype=torch.float64)
    from fhd.geometry.primitives import point_obstacle_distance

    d = point_obstacle_distance(pts, obs)
    assert torch.isinf(d).all()
