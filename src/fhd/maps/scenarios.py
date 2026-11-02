"""手工场景：S0/S1/S2 用的交叉、追尾、换位、终点占用、双通路、窄通道、多瓶颈。

场景包含地图、起终点、手工路径（或由折线重采样得到）以及允许主动等待的路点掩码。
S1/S2 使用**手动** wait_mask（批复 §22）；S4 才加入自动等待点检测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from fhd.geometry.paths import path_length, resample_polyline
from fhd.maps.generate import WorldMap, generate_map


@dataclass
class Scenario:
    name: str
    description: str
    world: WorldMap
    starts: torch.Tensor  # [N,2]
    goals: torch.Tensor  # [N,2]
    paths: torch.Tensor  # [N,H,2]（float64）
    wait_mask: torch.Tensor  # [N,H-1] bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_agents(self) -> int:
        return int(self.starts.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.paths.shape[1])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "world": self.world.to_dict(),
            "starts": self.starts.tolist(),
            "goals": self.goals.tolist(),
            "paths": self.paths.tolist(),
            "wait_mask": self.wait_mask.tolist(),
            "meta": self.meta,
        }


def _blank_mask(n: int, h: int, indices: dict[int, list[int]] | None = None) -> torch.Tensor:
    """构造 [N,H-1] 的等待掩码；indices 给出每台机器人允许等待的路点序号。"""
    mask = torch.zeros(n, h - 1, dtype=torch.bool)
    if indices:
        for agent, idxs in indices.items():
            for m in idxs:
                if 0 <= m < h - 1:
                    mask[agent, m] = True
    return mask


def _paths_from_waypoints(waypoints: list[list[list[float]]], h: int) -> torch.Tensor:
    return torch.stack([resample_polyline(torch.tensor(w, dtype=torch.float64), h) for w in waypoints], dim=0)


def _world_kind(kind: str, cfg: Any, seed: int, **meta) -> WorldMap:
    world = generate_map(kind, cfg, seed)
    world.meta.update(meta)
    return world


# --------------------------------------------------------------------------- #
# 1) 空旷交叉（核心机制验证）
# --------------------------------------------------------------------------- #
def crossing(cfg: Any, h: int | None = None, offset: float = 0.5, seed: int | None = None) -> Scenario:
    """两台机器人垂直交叉，交叉时刻相差约 2*offset/(边长) 个时隙。

    offset 越大，同步执行时错峰越明显；offset=0 时同步执行即在中点相撞。
    """
    h = h or int(cfg["trajectory"]["horizon"])
    seed = int(cfg["seed"]) if seed is None else seed
    world = _world_kind("open", cfg, seed)
    d = offset
    cy = 5.0 - d
    cx = 5.0 + d
    starts = torch.tensor([[0.6, cy], [cx, 0.6]], dtype=torch.float64)
    goals = torch.tensor([[9.4, cy], [cx, 9.4]], dtype=torch.float64)
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    p0 = starts[0] + t * (goals[0] - starts[0])
    p1 = starts[1] + t * (goals[1] - starts[1])
    paths = torch.stack([p0, p1], dim=0)
    # 允许在接近交叉点前 2~3 个路点主动等待
    mask = _blank_mask(2, h, {0: list(range(9, 19)), 1: list(range(9, 19))})
    return Scenario(
        name="crossing",
        description="空旷区域两台机器人垂直交叉；同步执行安全，随机暂停可能撞车",
        world=world,
        starts=starts,
        goals=goals,
        paths=paths,
        wait_mask=mask,
        meta={"offset": d, "cross_x": cx, "cross_y": cy},
    )


# --------------------------------------------------------------------------- #
# 2) 对向换位（R4 中途相撞）
# --------------------------------------------------------------------------- #
def head_on(cfg: Any, h: int | None = None, lateral: float = 0.0, seed: int | None = None) -> Scenario:
    """两台机器人沿同一条走廊相向而行并交换位置。"""
    h = h or int(cfg["trajectory"]["horizon"])
    seed = int(cfg["seed"]) if seed is None else seed
    world = _world_kind("open", cfg, seed)
    y = 5.0 + lateral
    starts = torch.tensor([[0.8, y], [9.2, y]], dtype=torch.float64)
    goals = torch.tensor([[9.2, y], [0.8, y]], dtype=torch.float64)
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    paths = torch.stack([starts[i] + t * (goals[i] - starts[i]) for i in range(2)], dim=0)
    mask = _blank_mask(2, h, {0: [0, 1, 2], 1: [0, 1, 2]})
    return Scenario(
        name="head_on",
        description="对向换位：起终点各自安全，中途穿越必然相撞；等待可错峰",
        world=world,
        starts=starts,
        goals=goals,
        paths=paths,
        wait_mask=mask,
        meta={"lateral": lateral},
    )


# --------------------------------------------------------------------------- #
# 3) 追尾
# --------------------------------------------------------------------------- #
def follow(cfg: Any, h: int | None = None, gap_steps: int = 2, seed: int | None = None) -> Scenario:
    """同向追尾：后车紧跟前车，前车被暂停后后车追上。

    gap_steps 用路径弧长换算：两车起点相距 gap_steps 个路点间距。
    """
    h = h or int(cfg["trajectory"]["horizon"])
    seed = int(cfg["seed"]) if seed is None else seed
    world = _world_kind("open", cfg, seed)
    y = 5.0
    seg = 8.0 / (h - 1)
    lead_start = np.array([1.0 + gap_steps * seg, y])
    starts = torch.tensor([[lead_start[0], y], [1.0, y]], dtype=torch.float64)
    goals = torch.tensor([[9.0, y], [9.0 - gap_steps * seg, y]], dtype=torch.float64)
    a = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    paths = torch.stack([starts[i] + a * (goals[i] - starts[i]) for i in range(2)], dim=0)
    mask = _blank_mask(2, h, {0: list(range(0, h - 2)), 1: list(range(0, h - 2))})
    return Scenario(
        name="follow",
        description="同向追尾：前车随机暂停会让后车逼近甚至相撞",
        world=world,
        starts=starts,
        goals=goals,
        paths=paths,
        wait_mask=mask,
        meta={"gap_steps": gap_steps, "spacing_m": seg},
    )


# --------------------------------------------------------------------------- #
# 4) 终点占用（R5）
# --------------------------------------------------------------------------- #
def terminal_occupancy(cfg: Any, h: int | None = None, seed: int | None = None) -> Scenario:
    """i 先完成并停在终点，j 之后穿越该终点位置。"""
    h = h or int(cfg["trajectory"]["horizon"])
    seed = int(cfg["seed"]) if seed is None else seed
    world = _world_kind("open", cfg, seed)
    starts = torch.tensor([[2.6, 5.0], [5.0, 0.8]], dtype=torch.float64)
    goals = torch.tensor([[5.0, 5.0], [5.0, 9.2]], dtype=torch.float64)
    a = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    paths = torch.stack([starts[i] + a * (goals[i] - starts[i]) for i in range(2)], dim=0)
    # i 的终点 (5,5) 在 j 的路径中点；不允许 i 在其终点等待，j 允许在穿过前等待
    mask = _blank_mask(2, h, {1: list(range(8, 20))})
    return Scenario(
        name="terminal_occupancy",
        description="i 提前占住终点，j 随后穿越该点；完成后仍占据终点位置",
        world=world,
        starts=starts,
        goals=goals,
        paths=paths,
        wait_mask=mask,
        meta={"occupied_point": [5.0, 5.0]},
    )


# --------------------------------------------------------------------------- #
# 5) 双通路绕障
# --------------------------------------------------------------------------- #
def dual_corridor(cfg: Any, h: int | None = None, same_side: bool = False, seed: int | None = None) -> Scenario:
    """中央大障碍，上下两条绕行通路。same_side=True 时两车都走同一侧（更危险）。"""
    h = h or int(cfg["trajectory"]["horizon"])
    seed = int(cfg["seed"]) if seed is None else seed
    world = _world_kind("dual_corridor", cfg, seed)
    rect = world.rects[0]
    cx = 0.5 * (rect[0] + rect[2])
    top_y = rect[3] + 1.0
    bottom_y = rect[1] - 1.0
    y_first = top_y
    y_second = top_y if same_side else bottom_y
    waypoints = [
        [[0.6, 5.0], [1.0, 5.0], [cx - 1.2, y_first], [cx + 1.2, y_first], [9.4, 5.0]],
        [[0.6, 5.0], [1.0, 5.0], [cx - 1.2, y_second], [cx + 1.2, y_second], [9.4, 5.0]],
    ]
    paths = _paths_from_waypoints(waypoints, h)
    starts = paths[:, 0, :].contiguous()
    goals = paths[:, -1, :].contiguous()
    mask = _blank_mask(2, h, {0: list(range(2, 14)), 1: list(range(2, 14))})
    return Scenario(
        name="dual_corridor",
        description="中央大障碍，上下两条绕行通路；两条路径可能汇合于同一侧",
        world=world,
        starts=starts,
        goals=goals,
        paths=paths,
        wait_mask=mask,
        meta={"same_side": same_side, "corridor_rect": rect.tolist()},
    )


# --------------------------------------------------------------------------- #
# 6) 窄通道门口（互斥通行）
# --------------------------------------------------------------------------- #
def narrow_gate(cfg: Any, h: int | None = None, seed: int | None = None) -> Scenario:
    """两车从两侧接近同一个只有单车宽的门口，必须错峰通过。"""
    h = h or int(cfg["trajectory"]["horizon"])
    seed = int(cfg["seed"]) if seed is None else seed
    world = _world_kind("narrow_gate", cfg, seed)
    gate = world.meta["gate_center"]
    gx, gy = gate
    waypoints = [
        [[0.6, gy], [gx, gy], [9.4, gy]],
        [[9.4, gy], [gx, gy], [0.6, gy]],
    ]
    paths = _paths_from_waypoints(waypoints, h)
    starts = paths[:, 0, :].contiguous()
    goals = paths[:, -1, :].contiguous()
    mask = _blank_mask(2, h, {0: list(range(4, 20)), 1: list(range(4, 20))})
    return Scenario(
        name="narrow_gate",
        description="窄门口互斥通行：两车对向进入只有单车宽度的缺口",
        world=world,
        starts=starts,
        goals=goals,
        paths=paths,
        wait_mask=mask,
        meta={"gate_center": gate, "gate_width_m": world.meta.get("gate_width_m")},
    )


# --------------------------------------------------------------------------- #
# 7) 多交叉/瓶颈（4/6 台）
# --------------------------------------------------------------------------- #
def multi_bottleneck(cfg: Any, h: int | None = None, n_agents: int = 4, seed: int | None = None) -> Scenario:
    """4 台或 6 台机器人穿越同一瓶颈区域（中央横槽），多条路径交叉。

    路径由 A* 在膨胀后的自由空间上生成，并用连续几何复核，保证静态可行。
    """
    from fhd.data.astar import build_grid, plan_path

    h = h or int(cfg["trajectory"]["horizon"])
    seed = int(cfg["seed"]) if seed is None else seed
    resolution = int(cfg["dataset"]["astar_resolution"])
    v_limit = float(cfg["trajectory"]["max_speed_mps"]) * float(cfg["trajectory"]["dt_s"])

    last_error = None
    for attempt in range(12):
        world = _world_kind("bottleneck", cfg, seed + attempt)
        y_mid = float(world.meta["corridor_y"])
        dy = 1.2
        base_patterns = [
            ([0.6, y_mid], [9.4, y_mid]),
            ([9.4, y_mid], [0.6, y_mid]),
            ([0.6, y_mid - dy], [9.4, y_mid + dy]),
            ([9.4, y_mid - dy], [0.6, y_mid + dy]),
            ([0.6, y_mid + dy], [9.4, y_mid - dy]),
            ([9.4, y_mid + dy], [0.6, y_mid - dy]),
        ]
        patterns = base_patterns[:n_agents]
        grid = build_grid(world, resolution)
        rng = np.random.default_rng(seed + attempt)
        paths = []
        for start, goal in patterns:
            p = plan_path(
                world, np.asarray(start, dtype=float), np.asarray(goal, dtype=float),
                resolution=resolution, h=h, rng=rng, noise_scale=0.0, grid=grid,
            )
            if p is None:
                last_error = f"第 {attempt} 次尝试：{start} -> {goal} 无可行路径"
                paths = None
                break
            paths.append(p)
        if paths is not None:
            path_t = torch.stack(paths, dim=0)
            lens = torch.linalg.vector_norm(path_t[:, 1:] - path_t[:, :-1], dim=-1)
            if float(lens.max()) <= v_limit + 1e-9:
                break
            last_error = f"第 {attempt} 次尝试：段长 {float(lens.max()):.3f} m 超过上限 {v_limit:.3f} m"
            paths = None
    if paths is None:
        raise RuntimeError(f"无法为多瓶颈场景生成可行路径: {last_error}")

    starts = path_t[:, 0, :].contiguous()
    goals = path_t[:, -1, :].contiguous()
    mask = _blank_mask(n_agents, h, {i: list(range(0, h - 4)) for i in range(n_agents)})
    return Scenario(
        name=f"multi_bottleneck_{n_agents}",
        description=f"{n_agents} 台机器人穿越同一瓶颈区域，多条路径交叉",
        world=world,
        starts=starts,
        goals=goals,
        paths=path_t,
        wait_mask=mask,
        meta={
            "n_agents": n_agents,
            "map_attempt": attempt,
            "map_seed_used": world.seed,
            "planner": "astar",
            "astar_resolution": resolution,
        },
    )


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
SCENARIO_BUILDERS = {
    "crossing": crossing,
    "head_on": head_on,
    "follow": follow,
    "terminal_occupancy": terminal_occupancy,
    "dual_corridor": dual_corridor,
    "narrow_gate": narrow_gate,
    "multi_bottleneck": multi_bottleneck,
}


def build_scenario(name: str, cfg: Any, **kwargs) -> Scenario:
    if name not in SCENARIO_BUILDERS:
        raise ValueError(f"未知场景: {name}，可选 {list(SCENARIO_BUILDERS)}")
    return SCENARIO_BUILDERS[name](cfg, **kwargs)


def scenario_summary(sc: Scenario) -> dict:
    """场景的几何摘要（写进报告）。"""
    lengths = path_length(sc.paths)
    return {
        "name": sc.name,
        "n_agents": sc.n_agents,
        "horizon": sc.horizon,
        "map_kind": sc.world.kind,
        "map_seed": sc.world.seed,
        "n_obstacles": sc.world.n_obstacles,
        "path_lengths_m": [round(float(x), 4) for x in lengths],
        "n_wait_points": int(sc.wait_mask.sum()),
        "min_start_distance_m": float(
            torch.cdist(sc.starts, sc.starts).masked_fill(torch.eye(sc.n_agents, dtype=torch.bool), 1e9).min()
        ),
    }
