"""基础几何图元：圆、轴对齐矩形、点/线段距离。

全部使用 torch 张量实现，默认对坐标可微（文档 §4.2 要求不得转 NumPy）。
约定：
    点集   points : [..., 2]
    圆     circles: [M, 3] = (cx, cy, r)
    矩形   rects  : [K, 4] = (x0, y0, x1, y1)，轴对齐
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

EPS_NUM = 1.0e-12
_SQRT_FLOOR = 1.0e-16


def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """带数值保护的 sqrt，避免 0 处梯度为 NaN（文档 §4.2）。"""
    return torch.sqrt(torch.clamp(x, min=_SQRT_FLOOR))


def as_tensor(data, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device) if device is not None else data.to(dtype=dtype)
    return torch.as_tensor(data, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
# 障碍物类型
# --------------------------------------------------------------------------- #
@dataclass
class Obstacles:
    """一组障碍物：圆形 + 轴对齐矩形。

    用张量存储以便向量化；空集用形状 [0,3] / [0,4] 表示。
    """

    circles: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 3))
    rects: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 4))

    def __post_init__(self) -> None:
        self.circles = as_tensor(self.circles).reshape(-1, 3)
        self.rects = as_tensor(self.rects).reshape(-1, 4)

    def to(self, dtype: torch.dtype | None = None, device=None) -> "Obstacles":
        return Obstacles(
            circles=self.circles.to(dtype=dtype) if dtype else self.circles,
            rects=self.rects.to(dtype=dtype) if dtype else self.rects,
        ) if device is None else Obstacles(
            circles=self.circles.to(dtype=dtype, device=device),
            rects=self.rects.to(dtype=dtype, device=device),
        )

    @property
    def count(self) -> int:
        return int(self.circles.shape[0] + self.rects.shape[0])

    def __len__(self) -> int:
        return self.count


# --------------------------------------------------------------------------- #
# 点到障碍物的距离（连续几何，非栅格近似）
# --------------------------------------------------------------------------- #
def point_circle_distance(points: torch.Tensor, circles: torch.Tensor) -> torch.Tensor:
    """点到圆周的距离（可正可负，负表示点落在圆内部）。返回 [..., M]。"""
    points = as_tensor(points)
    circles = as_tensor(circles, dtype=points.dtype)
    if circles.numel() == 0:
        return points.new_zeros(points.shape[:-1] + (0,))
    centers = circles[..., :2]
    radii = circles[..., 2]
    diff = points.unsqueeze(-2) - centers
    return safe_sqrt((diff * diff).sum(-1)) - radii


def point_rect_distance(points: torch.Tensor, rects: torch.Tensor) -> torch.Tensor:
    """点到轴对齐矩形区域的距离（矩形内部为 0）。返回 [..., K]。"""
    points = as_tensor(points)
    rects = as_tensor(rects, dtype=points.dtype)
    if rects.numel() == 0:
        return points.new_zeros(points.shape[:-1] + (0,))
    lo = rects[..., :2]
    hi = rects[..., 2:]
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    excess = (points.unsqueeze(-2) - center).abs() - half
    excess = torch.clamp(excess, min=0.0)
    return safe_sqrt((excess * excess).sum(-1))


def point_obstacle_distance(points: torch.Tensor, obstacles: Obstacles) -> torch.Tensor:
    """点到所有障碍物表面的最小距离。无障碍时返回 +inf。"""
    parts = []
    if obstacles.circles.numel() > 0:
        parts.append(point_circle_distance(points, obstacles.circles))
    if obstacles.rects.numel() > 0:
        parts.append(point_rect_distance(points, obstacles.rects))
    if not parts:
        return torch.full(
            points.shape[:-1], float("inf"), dtype=points.dtype, device=points.device
        )
    return torch.min(torch.cat(parts, dim=-1), dim=-1).values


# --------------------------------------------------------------------------- #
# 线段距离
# --------------------------------------------------------------------------- #
def segment_point_distance(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """线段 a-b 到点 p 的最近距离。a, b, p 形状均为 [..., 2]。"""
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(-1, keepdim=True) + EPS_NUM
    t = torch.clamp((ap * ab).sum(-1, keepdim=True) / denom, 0.0, 1.0)
    closest = a + t * ab
    diff = closest - p
    return safe_sqrt((diff * diff).sum(-1))


def segment_circle_distance(a: torch.Tensor, b: torch.Tensor, circles: torch.Tensor) -> torch.Tensor:
    """线段到圆周的最小距离。a,b: [...,2]；circles: [M,3] -> [..., M]。"""
    a = as_tensor(a)
    centers = circles[..., :2]
    radii = circles[..., 2]
    diff = segment_point_distance(a.unsqueeze(-2), b.unsqueeze(-2), centers)
    return diff - radii


def _segment_single_rect_distance(
    a: torch.Tensor, b: torch.Tensor, rect: torch.Tensor, iterations: int
) -> torch.Tensor:
    """线段到单个轴对齐矩形的最小距离（黄金分割，凸函数，精确到 1e-10）。"""
    lo = rect[:2]
    hi = rect[2:]
    ab = b - a
    inv_phi = (5**0.5 - 1) / 2

    def dist_at(u: torch.Tensor) -> torch.Tensor:
        p = a + u.unsqueeze(-1) * ab
        excess = torch.clamp(torch.maximum(lo - p, p - hi), min=0.0)
        return safe_sqrt((excess * excess).sum(-1))

    left = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
    right = torch.ones(a.shape[0], dtype=a.dtype, device=a.device)
    c = right - inv_phi * (right - left)
    d = left + inv_phi * (right - left)
    fc = dist_at(c)
    fd = dist_at(d)
    for _ in range(iterations):
        take_left = fc <= fd
        left = torch.where(take_left, left, c)
        right = torch.where(take_left, d, right)
        c = right - inv_phi * (right - left)
        d = left + inv_phi * (right - left)
        fc = dist_at(c)
        fd = dist_at(d)
    return torch.minimum(torch.minimum(fc, fd), dist_at(right))


def segment_rect_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    rects: torch.Tensor,
    iterations: int = 45,
) -> torch.Tensor:
    """线段到一组轴对齐矩形的最小距离。

    对**每个矩形单独**求 u∈[0,1] 上的凸函数极小（单矩形距离对 u 是凸的），
    再在所有矩形上取最小值。注意：min 会破坏凸性，因此不能对“到最近矩形的
    距离”直接做黄金分割，必须逐矩形求解。
    45 次迭代把区间缩到 0.618**45 ≈ 2.6e-10。
    该函数**不可微**，只用于硬碰撞判定与验证；可微惩罚项使用采样 SDF。
    返回 [S]。
    """
    a = as_tensor(a)
    b = as_tensor(b)
    rects = as_tensor(rects, dtype=a.dtype)
    if rects.numel() == 0:
        return torch.full((a.shape[0],), float("inf"), dtype=a.dtype, device=a.device)
    best = torch.full((a.shape[0],), float("inf"), dtype=a.dtype, device=a.device)
    for k in range(rects.shape[0]):
        best = torch.minimum(best, _segment_single_rect_distance(a, b, rects[k], iterations))
    return best


def segment_obstacle_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    obstacles: Obstacles,
    rect_iterations: int = 64,
) -> torch.Tensor:
    """线段到所有障碍物的最小距离（圆的解析距离取 min）。返回 [S]。"""
    parts = []
    if obstacles.circles.numel() > 0:
        parts.append(segment_circle_distance(a, b, obstacles.circles).min(-1).values)
    if obstacles.rects.numel() > 0:
        parts.append(segment_rect_distance(a, b, obstacles.rects, iterations=rect_iterations))
    if not parts:
        return torch.full((a.shape[0],), float("inf"), dtype=a.dtype, device=a.device)
    return torch.min(torch.stack(parts, dim=0), dim=0).values


# --------------------------------------------------------------------------- #
# 机器人间连续相对运动
# --------------------------------------------------------------------------- #
def relative_motion_min_distance(
    r0: torch.Tensor, v: torch.Tensor, eps: float = EPS_NUM
) -> tuple[torch.Tensor, torch.Tensor]:
    """两机器人在一个时隙内的最小中心距离（文档 §4.1）。

    r(u) = r0 + u v, u∈[0,1]；u* = clip(-r0·v / (||v||²+eps), 0, 1)。
    返回 (d_min, u_star)。当 ||v||≈0 时自动退化为静止端点距离。
    """
    vv = (v * v).sum(-1)
    dot = (r0 * v).sum(-1)
    u = torch.clamp(-dot / (vv + eps), 0.0, 1.0)
    diff = r0 + u.unsqueeze(-1) * v
    return safe_sqrt((diff * diff).sum(-1)), u
