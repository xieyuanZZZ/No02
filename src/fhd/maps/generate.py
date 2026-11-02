"""地图生成：4 类场景（空旷交叉 / 双通路绕障 / 窄通道门口 / 多交叉瓶颈）。

地图同时给出
    1) 连续障碍物几何（圆 + 轴对齐矩形）—— 一切碰撞最终按连续几何判定；
    2) 64×64 占据栅格与 SDF —— 仅用于网络条件输入与可视化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from fhd.geometry.primitives import Obstacles, point_obstacle_distance
from fhd.geometry.sdf import raster_sdf

MAP_KINDS = ("open", "dual_corridor", "narrow_gate", "bottleneck")


@dataclass
class WorldMap:
    kind: str
    width_m: float
    height_m: float
    robot_radius_m: float
    clearance_margin_m: float
    grid_size: int
    seed: int
    circles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rects: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    meta: dict[str, Any] = field(default_factory=dict)

    # --- 基本属性 -------------------------------------------------------
    @property
    def inflate(self) -> float:
        return self.robot_radius_m + self.clearance_margin_m

    @property
    def n_obstacles(self) -> int:
        return int(len(self.circles) + len(self.rects))

    def obstacles(self, dtype: torch.dtype = torch.float64) -> Obstacles:
        return Obstacles(
            circles=torch.as_tensor(self.circles, dtype=dtype).reshape(-1, 3),
            rects=torch.as_tensor(self.rects, dtype=dtype).reshape(-1, 4),
        )

    # --- 栅格与 SDF -----------------------------------------------------
    def sdf_grid(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        sdf = raster_sdf(
            self.obstacles(torch.float64),
            self.width_m,
            self.height_m,
            self.grid_size,
            self.robot_radius_m,
            self.clearance_margin_m,
        )
        return sdf.to(dtype)

    def occupancy(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (self.sdf_grid(dtype) < 0).to(dtype)

    # --- 自由空间查询 ---------------------------------------------------
    def sdf(self, points: np.ndarray | torch.Tensor) -> np.ndarray:
        """自由空间 SDF（已扣除膨胀），>=0 表示安全。输入 [...,2]，返回 [...]。"""
        pts = torch.as_tensor(np.asarray(points) if isinstance(points, np.ndarray) else points, dtype=torch.float64)
        return (point_obstacle_distance(pts, self.obstacles()) - self.inflate).numpy()

    def is_free(self, points: np.ndarray) -> np.ndarray:
        return self.sdf(points) >= 0.0

    def sample_free_points(
        self,
        rng: np.random.Generator,
        n: int,
        min_sep: float = 0.0,
        border_m: float = 0.4,
        max_tries: int = 20000,
    ) -> np.ndarray:
        """在膨胀后的自由空间内采样 n 个点（两两间距 >= min_sep）。"""
        out: list[np.ndarray] = []
        tries = 0
        while len(out) < n and tries < max_tries:
            tries += 1
            p = np.array(
                [
                    rng.uniform(border_m, self.width_m - border_m),
                    rng.uniform(border_m, self.height_m - border_m),
                ]
            )
            if self.sdf(p[None, :])[0] < 0.0:
                continue
            if out and min_sep > 0:
                d = np.linalg.norm(np.stack(out) - p[None, :], axis=1)
                if d.min() < min_sep:
                    continue
            out.append(p)
        if len(out) < n:
            raise RuntimeError(f"自由空间采样失败（{self.kind}），只采到 {len(out)}/{n} 个点")
        return np.stack(out)

    # --- 序列化 ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "robot_radius_m": self.robot_radius_m,
            "clearance_margin_m": self.clearance_margin_m,
            "grid_size": self.grid_size,
            "seed": self.seed,
            "circles": self.circles.tolist(),
            "rects": self.rects.tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldMap":
        return cls(
            kind=data["kind"],
            width_m=data["width_m"],
            height_m=data["height_m"],
            robot_radius_m=data["robot_radius_m"],
            clearance_margin_m=data["clearance_margin_m"],
            grid_size=data["grid_size"],
            seed=data["seed"],
            circles=np.asarray(data.get("circles", []), dtype=np.float64).reshape(-1, 3),
            rects=np.asarray(data.get("rects", []), dtype=np.float64).reshape(-1, 4),
            meta=data.get("meta", {}),
        )


# --------------------------------------------------------------------------- #
# 具体场景生成
# --------------------------------------------------------------------------- #
def _base(kind: str, cfg: Any, seed: int) -> WorldMap:
    world = cfg["world"] if "world" in cfg else cfg
    return WorldMap(
        kind=kind,
        width_m=float(world["width_m"]),
        height_m=float(world["height_m"]),
        robot_radius_m=float(world["robot_radius_m"]),
        clearance_margin_m=float(world["clearance_margin_m"]),
        grid_size=int(world["grid_size"]),
        seed=int(seed),
    )


def _open_map(cfg: Any, seed: int) -> WorldMap:
    """空旷交叉：无障碍或极少小障碍。"""
    m = _base("open", cfg, seed)
    rng = np.random.default_rng(seed)
    n_small = int(rng.integers(0, 3))
    circles = []
    for _ in range(n_small):
        c = rng.uniform([1.0, 1.0], [m.width_m - 1.0, m.height_m - 1.0])
        r = rng.uniform(0.15, 0.28)
        circles.append([c[0], c[1], r])
    m.circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    m.meta = {"difficulty": 1, "n_small_obstacles": n_small}
    return m


def _dual_corridor_map(cfg: Any, seed: int) -> WorldMap:
    """双通路绕障：中央一个大型矩形，左右留出上下两条绕行通路。"""
    m = _base("dual_corridor", cfg, seed)
    rng = np.random.default_rng(seed)
    cy = m.height_m / 2 + rng.uniform(-0.6, 0.6)
    height = rng.uniform(2.6, 3.6)
    width = rng.uniform(2.2, 3.2)
    cx = m.width_m / 2 + rng.uniform(-0.5, 0.5)
    rect = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
    m.rects = np.asarray([rect], dtype=np.float64).reshape(-1, 4)
    m.meta = {"difficulty": 2, "corridor_width_m": width}
    return m


def _narrow_gate_map(cfg: Any, seed: int) -> WorldMap:
    """窄通道/门口：两道竖墙，中间一个只能单车通过的缺口。"""
    m = _base("narrow_gate", cfg, seed)
    rng = np.random.default_rng(seed)
    gap = rng.uniform(0.55, 0.8)  # 缺口宽度；机器人直径 0.30，需 > 0.36
    wall_x = m.width_m / 2 + rng.uniform(-0.4, 0.4)
    gap_y = m.height_m / 2 + rng.uniform(-1.2, 1.2)
    thickness = 0.25
    y_lo = gap_y - gap / 2
    y_hi = gap_y + gap / 2
    rects = [
        [wall_x - thickness / 2, 0.0, wall_x + thickness / 2, y_lo],
        [wall_x - thickness / 2, y_hi, wall_x + thickness / 2, m.height_m],
    ]
    # 追加 1~2 个随机障碍增加难度（但不堵死）
    extra = []
    for _ in range(int(rng.integers(0, 2))):
        c = rng.uniform([1.0, 1.0], [m.width_m - 1.0, m.height_m - 1.0])
        if abs(c[0] - wall_x) < 0.8:
            continue
        extra.append([c[0] - 0.25, c[1] - 0.25, c[0] + 0.25, c[1] + 0.25])
    m.rects = np.asarray(rects + extra, dtype=np.float64).reshape(-1, 4)
    m.meta = {"difficulty": 3, "gate_width_m": gap, "gate_center": [float(wall_x), float(gap_y)]}
    return m


def _bottleneck_map(cfg: Any, seed: int) -> WorldMap:
    """多交叉/瓶颈混合：一个中央瓶颈 + 若干分散障碍，用于 4/6 台。

    随机障碍必须避开中央横槽与入口通道，否则会把瓶颈堵死（无解地图一律丢弃）。
    """
    m = _base("bottleneck", cfg, seed)
    rng = np.random.default_rng(seed)
    rects = []
    circles = []
    corridor_h = rng.uniform(1.0, 1.5)
    cy = m.height_m / 2 + rng.uniform(-0.8, 0.8)
    half_w = rng.uniform(1.2, 1.8)
    rects.append([m.width_m / 2 - half_w, 0.0, m.width_m / 2 + half_w, cy - corridor_h / 2])
    rects.append([m.width_m / 2 - half_w, cy + corridor_h / 2, m.width_m / 2 + half_w, m.height_m])

    def near_corridor(p) -> bool:
        in_x = (m.width_m / 2 - half_w - 1.0) <= p[0] <= (m.width_m / 2 + half_w + 1.0)
        in_y = abs(p[1] - cy) <= corridor_h / 2 + 1.0
        return in_x and in_y

    placed = 0
    attempts = 0
    target = int(rng.integers(2, 5))
    while placed < target and attempts < 200:
        attempts += 1
        if rng.random() < 0.5:
            c = rng.uniform([1.2, 1.2], [m.width_m - 1.2, m.height_m - 1.2])
            r = rng.uniform(0.2, 0.45)
            if near_corridor(c) or np.hypot(c[0] - m.width_m / 2, c[1] - cy) < r + 1.0:
                continue
            circles.append([c[0], c[1], r])
        else:
            c = rng.uniform([1.5, 1.5], [m.width_m - 1.5, m.height_m - 1.5])
            w = rng.uniform(0.4, 0.9)
            h = rng.uniform(0.4, 0.9)
            if near_corridor(c):
                continue
            rects.append([c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2])
        placed += 1
    m.rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    m.circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    m.meta = {"difficulty": 4, "corridor_height_m": corridor_h, "corridor_y": float(cy),
              "half_width_m": float(half_w)}
    return m


_GENERATORS = {
    "open": _open_map,
    "dual_corridor": _dual_corridor_map,
    "narrow_gate": _narrow_gate_map,
    "bottleneck": _bottleneck_map,
}


def generate_map(kind: str, cfg: Any, seed: int) -> WorldMap:
    """按类型与种子生成地图（同种子可复现）。"""
    if kind not in _GENERATORS:
        raise ValueError(f"未知地图类型: {kind}，可选 {MAP_KINDS}")
    m = _GENERATORS[kind](cfg, seed)
    # 自检：起点/终点区（四角与中心）不能全部被堵死
    probes = np.array(
        [
            [0.6, 0.6],
            [m.width_m - 0.6, m.height_m - 0.6],
            [0.6, m.height_m - 0.6],
            [m.width_m - 0.6, 0.6],
            [m.width_m / 2, m.height_m / 2],
        ]
    )
    if not m.is_free(probes).any():
        raise RuntimeError(f"地图 {kind}/{seed} 没有可用自由空间")
    return m
