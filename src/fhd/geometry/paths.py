"""路径构造与重采样工具（几何层，不涉及数据集划分）。"""

from __future__ import annotations

import torch

from fhd.geometry.primitives import safe_sqrt


def path_length(paths: torch.Tensor) -> torch.Tensor:
    """路径总长度。paths [..., H, 2] -> [...]"""
    return safe_sqrt(((paths[..., 1:, :] - paths[..., :-1, :]) ** 2).sum(-1)).sum(-1)


def straight_path(start: torch.Tensor, goal: torch.Tensor, h: int) -> torch.Tensor:
    """两点直线等分 h 个路点（含起终点）。start/goal [2] -> [h,2]"""
    t = torch.linspace(0.0, 1.0, h, dtype=start.dtype, device=start.device).unsqueeze(-1)
    return start + t * (goal - start)


def resample_polyline(points: torch.Tensor, h: int) -> torch.Tensor:
    """按累计弧长把折线等距重采样为 h 个路点（文档 §6.2 步骤 5）。points [M,2] -> [h,2]"""
    points = points.reshape(-1, 2)
    if points.shape[0] == 1:
        return points.expand(h, 2)
    seg = safe_sqrt(((points[1:] - points[:-1]) ** 2).sum(-1))
    cum = torch.cat([points.new_zeros(1), torch.cumsum(seg, 0)])
    total = cum[-1]
    if float(total) <= 0:
        return points[:1].expand(h, 2)
    target = torch.linspace(0.0, float(total), h, dtype=points.dtype, device=points.device)
    # 找到每个目标弧长所在的段
    idx = torch.searchsorted(cum, target, right=True).clamp(1, points.shape[0] - 1) - 1
    t = (target - cum[idx]) / (seg[idx] + 1e-12)
    out = points[idx] + t.unsqueeze(-1) * (points[idx + 1] - points[idx])
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def smooth_path(paths: torch.Tensor, iterations: int = 2, alpha: float = 0.5) -> torch.Tensor:
    """固定端点的平滑（移动平均）。paths [...,H,2] -> 同形状新张量。"""
    out = paths
    for _ in range(iterations):
        mid = 0.5 * (out[..., :-2, :] + out[..., 2:, :])
        smoothed = out.clone()
        smoothed[..., 1:-1, :] = (1.0 - alpha) * out[..., 1:-1, :] + alpha * mid
        out = smoothed
    return out


def polyline_path(waypoints: list[list[float]] | torch.Tensor, h: int) -> torch.Tensor:
    """折线 -> h 点等距路径。"""
    pts = torch.as_tensor(waypoints, dtype=torch.float64).reshape(-1, 2)
    return resample_polyline(pts, h).to(torch.float64)


def dense_sample(paths: torch.Tensor, spacing: float = 0.02) -> torch.Tensor:
    """把路径按固定弧长采样成密集点（用于可视化与独立复核）。"""
    pts = paths.reshape(-1, 2)
    seg = safe_sqrt(((pts[1:] - pts[:-1]) ** 2).sum(-1))
    cum = torch.cat([pts.new_zeros(1), torch.cumsum(seg, 0)])
    total = float(cum[-1])
    n = max(2, int(total / spacing) + 1)
    target = torch.linspace(0.0, total, n, dtype=pts.dtype, device=pts.device)
    idx = torch.searchsorted(cum, target, right=True).clamp(1, pts.shape[0] - 1) - 1
    t = (target - cum[idx]) / (seg[idx] + 1e-12)
    return pts[idx] + t.unsqueeze(-1) * (pts[idx + 1] - pts[idx])


def speed_profile(paths: torch.Tensor, dt_s: float) -> torch.Tensor:
    """逐段平均速度。paths [...,H,2] -> [...,H-1]"""
    return safe_sqrt(((paths[..., 1:, :] - paths[..., :-1, :]) ** 2).sum(-1)) / dt_s
