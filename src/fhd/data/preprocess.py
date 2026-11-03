"""路径预处理：简化、重采样校验、可行性与多样性检查（文档 §6.2 步骤 4–6）。"""

from __future__ import annotations

import numpy as np
import torch

from fhd.geometry.collisions import path_segment_lengths, trajectory_is_static_safe
from fhd.geometry.paths import path_length


def validate_path(
    path: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    world,
    max_speed_mps: float,
    dt_s: float,
    endpoint_tol: float = 1e-6,
) -> dict:
    """单条路径的完整可行性检查，返回诊断字典。"""
    path = path.reshape(1, -1, 2)
    lens = path_segment_lengths(path)[0]
    safe_static, clearance = trajectory_is_static_safe(
        path, world.obstacles(), world.robot_radius_m, world.clearance_margin_m
    )
    return {
        "ok": bool(
            (lens <= max_speed_mps * dt_s + 1e-9).all()
            and safe_static.all()
            and torch.linalg.vector_norm(path[0, 0] - start) <= endpoint_tol
            and torch.linalg.vector_norm(path[0, -1] - goal) <= endpoint_tol
        ),
        "max_segment_m": float(lens.max()),
        "min_clearance_m": float(clearance.min()),
        "length_m": float(path_length(path)[0]),
        "static_safe": bool(safe_static.all()),
        "speed_ok": bool((lens <= max_speed_mps * dt_s + 1e-9).all()),
    }


def path_distance_matrix(paths: torch.Tensor) -> np.ndarray:
    """路径两两之间的平均点距（用于候选多样性统计，只比较同一任务）。"""
    n = paths.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.linalg.vector_norm(paths[i] - paths[j], dim=-1).mean().item()
            out[i, j] = out[j, i] = d
    return out


def diversity_score(paths: torch.Tensor) -> float:
    """同一任务的多个候选之间的平均两两距离（越大越多样）。"""
    if paths.shape[0] < 2:
        return 0.0
    dm = path_distance_matrix(paths)
    iu = np.triu_indices(paths.shape[0], k=1)
    return float(dm[iu].mean())


def normalise_paths(paths: torch.Tensor, width_m: float, height_m: float) -> torch.Tensor:
    """把米制坐标映射到 [-1,1]（文档 §6.4）。"""
    scale = torch.tensor([width_m / 2.0, height_m / 2.0], dtype=paths.dtype, device=paths.device)
    return paths / scale - 1.0


def denormalise_paths(paths: torch.Tensor, width_m: float, height_m: float) -> torch.Tensor:
    """把 [-1,1] 映射回米制坐标。"""
    scale = torch.tensor([width_m / 2.0, height_m / 2.0], dtype=paths.dtype, device=paths.device)
    return (paths + 1.0) * scale
