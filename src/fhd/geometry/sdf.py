"""地图 signed distance field（SDF）。

两种用途，**不可混用**：
1. `analytic_sdf`：由矩形/圆的解析距离直接计算，用于可微的静态碰撞惩罚 L_static；
2. `raster_sdf`：在栅格中心采样解析距离得到，用于可视化与 Diffusion 的条件输入。

硬碰撞判定一律走 geometry.collisions 的连续几何检查，绝不以栅格近似替代。
"""

from __future__ import annotations

import torch

from fhd.geometry.primitives import (
    Obstacles,
    point_obstacle_distance,
    safe_sqrt,
    segment_obstacle_distance,
)


def analytic_sdf(
    points: torch.Tensor,
    obstacles: Obstacles,
    robot_radius: float,
    clearance_margin: float,
) -> torch.Tensor:
    """自由空间 SDF：>=0 表示该点放置机器人中心是安全的（已扣除膨胀量）。

    points [..., 2] -> [...]
    """
    inflate = robot_radius + clearance_margin
    dist = point_obstacle_distance(points, obstacles)
    return dist - inflate


def segment_sdf_min(
    a: torch.Tensor,
    b: torch.Tensor,
    obstacles: Obstacles,
    robot_radius: float,
    clearance_margin: float,
) -> torch.Tensor:
    """线段上的最小 SDF（连续精确，不可微），用于复核。"""
    inflate = robot_radius + clearance_margin
    return segment_obstacle_distance(a, b, obstacles) - inflate


def raster_sdf(
    obstacles: Obstacles,
    width_m: float,
    height_m: float,
    grid_size: int,
    robot_radius: float,
    clearance_margin: float,
) -> torch.Tensor:
    """在 grid_size×grid_size 栅格中心采样解析 SDF，返回 [G,G]（单位：米）。

    约定：行索引对应 y（自下而上），列索引对应 x；像素中心坐标
    x = (j+0.5)*w/G, y = (i+0.5)*h/G。
    """
    ys = (torch.arange(grid_size, dtype=torch.float64) + 0.5) * (height_m / grid_size)
    xs = (torch.arange(grid_size, dtype=torch.float64) + 0.5) * (width_m / grid_size)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    sdf = analytic_sdf(pts, obstacles, robot_radius, clearance_margin)
    return sdf.reshape(grid_size, grid_size)


def occupancy_grid(
    obstacles: Obstacles,
    width_m: float,
    height_m: float,
    grid_size: int,
    robot_radius: float,
    clearance_margin: float,
) -> torch.Tensor:
    """膨胀后的占据栅格 [G,G]，1 表示障碍（含膨胀），0 表示自由。"""
    sdf = raster_sdf(obstacles, width_m, height_m, grid_size, robot_radius, clearance_margin)
    return (sdf < 0).to(torch.float32)


def sampled_static_penalty(
    paths: torch.Tensor,
    obstacles: Obstacles,
    robot_radius: float,
    clearance_margin: float,
    samples_per_segment: int = 8,
    margin_m: float = 0.0,
) -> torch.Tensor:
    """可微的静态障碍惩罚 L_static（文档 §7.1）。

    在每段上等距采样 samples_per_segment 个点，对 sdf < margin 的部分取平方惩罚。
    这是一个**优化代理**；最终是否静态安全由连续几何检查独立复核。
    paths [N,H,2] -> 标量
    """
    if obstacles.count == 0:
        return paths.new_zeros(())
    n, h, _ = paths.shape
    a = paths[:, :-1, :]  # [N,H-1,2]
    b = paths[:, 1:, :]
    frac = torch.linspace(
        0.0, 1.0, samples_per_segment + 1, dtype=paths.dtype, device=paths.device
    )[1:]  # 不含起点，避免重复
    pts = a.unsqueeze(2) + frac.view(1, 1, -1, 1) * (b - a).unsqueeze(2)  # [N,H-1,S,2]
    sdf = analytic_sdf(pts.reshape(-1, 2), obstacles, robot_radius, clearance_margin)
    violation = torch.clamp(margin_m - sdf, min=0.0)
    return (violation * violation).mean()


def static_penalty_report(
    paths: torch.Tensor,
    obstacles: Obstacles,
    robot_radius: float,
    clearance_margin: float,
) -> dict[str, float]:
    """静态惩罚的独立复核（连续精确），用于报告与单元测试。"""
    inflate = robot_radius + clearance_margin
    a = paths[:, :-1, :].reshape(-1, 2)
    b = paths[:, 1:, :].reshape(-1, 2)
    if obstacles.count == 0:
        return {"min_clearance_m": float("inf"), "colliding_segments": 0.0}
    d = segment_obstacle_distance(a, b, obstacles) - inflate
    return {
        "min_clearance_m": float(d.min()),
        "colliding_segments": float((d < 0).sum()),
    }


def raster_to_units(grid: torch.Tensor) -> torch.Tensor:
    """辅助函数：把栅格值归一化到 [-1,1] 供网络输入。"""
    return grid * 2.0 - 1.0


def safe_sdf_sqrt(x: torch.Tensor) -> torch.Tensor:
    """兼容导出，便于外部一致的数值保护。"""
    return safe_sqrt(x)
