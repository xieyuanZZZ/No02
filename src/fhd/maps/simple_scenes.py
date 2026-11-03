"""第二版方案 · S1：三类简单仿真场景。

设计原则（对应方案 §3.1）：
  * 障碍物少（0–3 个）、形状简单（矩形/圆）、单机容易规划；
  * 多机之间仍有明显的路径交叉 / 共享通道 / 时序冲突；
  * 通过起终点与机器人数量控制交互难度。

三类场景：
  A `open_crossing`        —— 开放交叉：0 或 1 个角落小障碍，路径在中心相交
  B `single_obstacle`      —— 单障碍绕行：中央 1 个矩形/圆，两侧可绕
  C `bottleneck_channel`   —— 单瓶颈通行：两道墙形成只容单车通过的通道

与旧场景的区别：不再生成密集障碍、双通路大块、多障碍瓶颈。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fhd.maps.generate import WorldMap

SIMPLE_KINDS = ("open_crossing", "single_obstacle", "bottleneck_channel")

KIND_LABELS = {
    "open_crossing": "开放交叉",
    "single_obstacle": "单障碍绕行",
    "bottleneck_channel": "单瓶颈通行",
}


def _base(kind: str, cfg, seed: int) -> WorldMap:
    world = cfg["world"]
    return WorldMap(
        kind=kind,
        width_m=float(world["width_m"]),
        height_m=float(world["height_m"]),
        robot_radius_m=float(world["robot_radius_m"]),
        clearance_margin_m=float(world["clearance_margin_m"]),
        grid_size=int(world["grid_size"]),
        seed=int(seed),
    )


# --------------------------------------------------------------------------- #
def open_crossing(cfg, seed: int) -> WorldMap:
    """A：开放交叉。0 个障碍，或在远离中心的角落放 1 个小圆。"""
    m = _base("open_crossing", cfg, seed)
    rng = np.random.default_rng(seed)
    circles = []
    if rng.random() < 0.5:
        # 角落区域：离中心至少 2.5 m，避免干扰交叉
        for _ in range(50):
            c = rng.uniform([1.0, 1.0], [m.width_m - 1.0, m.height_m - 1.0])
            if np.hypot(c[0] - m.width_m / 2, c[1] - m.height_m / 2) >= 2.5:
                circles.append([c[0], c[1], float(rng.uniform(0.2, 0.35))])
                break
    m.circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    m.meta = {"difficulty": 1, "kind_label": KIND_LABELS["open_crossing"],
              "n_circles": len(circles), "n_rects": 0}
    return m


def single_obstacle(cfg, seed: int) -> WorldMap:
    """B：单障碍绕行。中央 1 个矩形或圆，两侧留出足够绕行宽度。"""
    m = _base("single_obstacle", cfg, seed)
    rng = np.random.default_rng(seed)
    cx = m.width_m / 2 + float(rng.uniform(-0.6, 0.6))
    cy = m.height_m / 2 + float(rng.uniform(-0.6, 0.6))
    if rng.random() < 0.5:
        r = float(rng.uniform(0.6, 0.95))
        m.circles = np.asarray([[cx, cy, r]], dtype=np.float64).reshape(-1, 3)
        m.rects = np.zeros((0, 4))
        shape = {"type": "circle", "r": r, "center": [cx, cy]}
    else:
        w = float(rng.uniform(1.2, 2.0))
        h = float(rng.uniform(1.2, 2.0))
        m.rects = np.asarray([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]],
                             dtype=np.float64).reshape(-1, 4)
        m.circles = np.zeros((0, 3))
        shape = {"type": "rect", "w": w, "h": h, "center": [cx, cy]}
    m.meta = {"difficulty": 2, "kind_label": KIND_LABELS["single_obstacle"],
              "obstacle": shape, "n_circles": int(len(m.circles)), "n_rects": int(len(m.rects))}
    return m


def bottleneck_channel(cfg, seed: int) -> WorldMap:
    """C：单瓶颈通行。两道墙形成共享通道。

    通道净宽 g 取 0.45–0.62 m：
      * 单机通过所需净空 = 2*(r+clearance) = 0.36 m  -> 能过；
      * 两机并排所需 = 2*d_safe + 2*clearance = 0.76 m -> 不能并排。
    通道两侧留出大量等待空间。
    """
    m = _base("bottleneck_channel", cfg, seed)
    rng = np.random.default_rng(seed)
    g = float(rng.uniform(0.45, 0.62))
    wall_t = 0.25
    horizontal = bool(rng.random() < 0.5)
    cx = m.width_m / 2 + float(rng.uniform(-0.5, 0.5))
    cy = m.height_m / 2 + float(rng.uniform(-0.5, 0.5))
    if horizontal:
        # 通道沿 x 方向通行，两道墙分别位于 y 的上下
        rects = [
            [cx - 2.5, cy + g / 2, cx + 2.5, cy + g / 2 + wall_t],
            [cx - 2.5, cy - g / 2 - wall_t, cx + 2.5, cy - g / 2],
        ]
    else:
        rects = [
            [cx + g / 2, cy - 2.5, cx + g / 2 + wall_t, cy + 2.5],
            [cx - g / 2 - wall_t, cy - 2.5, cx - g / 2, cy + 2.5],
        ]
    m.rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    m.circles = np.zeros((0, 3))
    m.meta = {"difficulty": 3, "kind_label": KIND_LABELS["bottleneck_channel"],
              "gap_m": g, "wall_thickness_m": wall_t, "horizontal": horizontal,
              "center": [cx, cy], "n_circles": 0, "n_rects": 2}
    return m


_GEN = {
    "open_crossing": open_crossing,
    "single_obstacle": single_obstacle,
    "bottleneck_channel": bottleneck_channel,
}


def generate_simple_map(kind: str, cfg, seed: int) -> WorldMap:
    if kind not in _GEN:
        raise ValueError(f"未知场景 {kind}，可选 {SIMPLE_KINDS}")
    m = _GEN[kind](cfg, seed)
    probes = np.array([[0.8, 0.8], [m.width_m - 0.8, m.height_m - 0.8],
                       [0.8, m.height_m - 0.8], [m.width_m - 0.8, 0.8],
                       [m.width_m / 2, m.height_m / 2]])
    if not m.is_free(probes).any():
        raise RuntimeError(f"{kind}/{seed} 没有可用自由空间")
    return m


def map_id_of(kind: str, seed: int) -> str:
    return f"{kind}_{seed}"


def channel_axis(m: WorldMap) -> tuple[float, float]:
    """返回通道中心点（用于任务设计）。"""
    if m.kind != "bottleneck_channel":
        return (m.width_m / 2, m.height_m / 2)
    c = m.meta.get("center", [m.width_m / 2, m.height_m / 2])
    return (float(c[0]), float(c[1]))
