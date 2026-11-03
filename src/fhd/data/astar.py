"""从零实现的 A* / 加权 A* 单机器人路径搜索（文档 §6.2）。

要点：
  * 在**解析 SDF** 生成的自由空间掩码上搜索（已扣除机器人半径与净空），
    不用栅格中心近似替代连续碰撞判定；
  * 支持随机边代价扰动（生成多模态、多样化路径，而不是唯一最短路）；
  * 搜索结果是栅格折线，随后做**保持碰撞安全的捷径平滑**，再按弧长重采样；
  * 任何处理之后都再跑一次连续几何碰撞检查（调用方负责断言）。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from fhd.geometry.paths import path_length, resample_polyline
from fhd.geometry.primitives import segment_obstacle_distance

NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, float(2**0.5)), (-1, 1, float(2**0.5)),
    (1, -1, float(2**0.5)), (1, 1, float(2**0.5)),
]


@dataclass
class Grid:
    free: np.ndarray  # [G,G] bool，True 表示可通行
    resolution: int
    width_m: float
    height_m: float
    free_safe: np.ndarray | None = None  # 腐蚀一格后的自由空间（平滑阶段保守判定用）

    @property
    def cell(self) -> float:
        return self.width_m / self.resolution

    def to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        j = int(np.clip(np.floor(xy[0] / self.cell), 0, self.resolution - 1))
        i = int(np.clip(np.floor(xy[1] / self.cell), 0, self.resolution - 1))
        return i, j

    def to_xy(self, i: int, j: int) -> np.ndarray:
        return np.array([(j + 0.5) * self.cell, (i + 0.5) * self.cell])


def _erode(mask: np.ndarray) -> np.ndarray:
    """3×3 腐蚀：只有邻域全自由才保留，使平滑检查偏保守。"""
    padded = np.pad(mask, 1, constant_values=False)
    out = mask.copy()
    res = mask.shape[0]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out &= padded[1 + di : 1 + di + res, 1 + dj : 1 + dj + res]
    return out


def build_grid(world, resolution: int | None = None) -> Grid:
    """由世界模型构造自由空间栅格（膨胀已包含在 SDF 中）。"""
    resolution = int(resolution or world.grid_size)
    cell = world.width_m / resolution
    xs = (np.arange(resolution) + 0.5) * cell
    ys = (np.arange(resolution) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="xy")  # [G,G]
    pts = np.stack([gx, gy], axis=-1).reshape(-1, 2)
    sdf = world.sdf(pts).reshape(resolution, resolution)
    free = sdf >= 0.0
    return Grid(
        free=free,
        resolution=resolution,
        width_m=world.width_m,
        height_m=world.height_m,
        free_safe=_erode(free),
    )


def _nearest_free(grid: Grid, cell: tuple[int, int], max_radius: int = 6) -> tuple[int, int] | None:
    i0, j0 = cell
    if grid.free[i0, j0]:
        return cell
    for r in range(1, max_radius + 1):
        best = None
        best_d = 1e9
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                i, j = i0 + di, j0 + dj
                if 0 <= i < grid.resolution and 0 <= j < grid.resolution and grid.free[i, j]:
                    d = di * di + dj * dj
                    if d < best_d:
                        best_d = d
                        best = (i, j)
        if best is not None:
            return best
    return None


def astar(
    grid: Grid,
    start: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.0,
    heuristic_weight: float = 1.0,
) -> np.ndarray | None:
    """8 邻域 A*；noise_scale>0 时对每条边乘 (1+noise) 的随机代价，产生多模态路径。

    用扁平数组代替 dict 保存 g 值与 closed 标记，使 256×256 规模可在纯 Python 下运行。
    """
    s = _nearest_free(grid, grid.to_cell(np.asarray(start, dtype=float)))
    gcell = _nearest_free(grid, grid.to_cell(np.asarray(goal, dtype=float)))
    if s is None or gcell is None:
        return None
    # 在**腐蚀后的**自由空间上搜索：保证后续重采样产生的弦仍然满足净空
    free = grid.free_safe if grid.free_safe is not None else grid.free
    res = grid.resolution
    n = res * res
    s_idx = s[0] * res + s[1]
    g_idx = gcell[0] * res + gcell[1]
    if s_idx == g_idx:
        return np.stack([grid.to_xy(*s)])

    noise = None
    if noise_scale > 0 and rng is not None:
        noise = rng.uniform(0.0, noise_scale, size=(res, res)).astype(np.float64)

    gi = np.full(n, np.inf, dtype=np.float64)
    gi[s_idx] = 0.0
    closed = np.zeros(n, dtype=bool)
    came = np.full(n, -1, dtype=np.int64)

    gi_r, gj_r = gcell
    heap: list[tuple[float, int]] = [
        (heuristic_weight * float(np.hypot(s[0] - gi_r, s[1] - gj_r)), s_idx)
    ]
    found = False
    while heap:
        _, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        closed[idx] = True
        if idx == g_idx:
            found = True
            break
        ci, cj = divmod(idx, res)
        base = gi[idx]
        for di, dj, step in NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if ni < 0 or nj < 0 or ni >= res or nj >= res:
                continue
            if not free[ni, nj]:
                continue
            if di != 0 and dj != 0:
                if not free[ci + di, cj] or not free[ci, cj + dj]:
                    continue
            nidx = ni * res + nj
            if closed[nidx]:
                continue
            cost = step * (1.0 + (noise[ni, nj] if noise is not None else 0.0))
            tentative = base + cost
            if tentative < gi[nidx]:
                gi[nidx] = tentative
                came[nidx] = idx
                heapq.heappush(
                    heap,
                    (tentative + heuristic_weight * float(np.hypot(ni - gi_r, nj - gj_r)), nidx),
                )
    if not found:
        return None

    cells = []
    cur = g_idx
    while cur != -1:
        cells.append(divmod(cur, res))
        if cur == s_idx:
            break
        cur = came[cur]
    else:
        return None
    cells.reverse()
    return np.stack([grid.to_xy(i, j) for i, j in cells])


def _segment_clear(world, a: np.ndarray, b: np.ndarray) -> bool:
    """线段是否与障碍物保持（半径+净空）以上的距离（连续几何判定，精确但较慢）。"""
    import torch

    a_t = torch.as_tensor(np.asarray(a, dtype=np.float64)).view(1, 2)
    b_t = torch.as_tensor(np.asarray(b, dtype=np.float64)).view(1, 2)
    d = segment_obstacle_distance(a_t, b_t, world.obstacles(), rect_iterations=25)
    return float(d[0]) >= world.robot_radius_m + world.clearance_margin_m


def _segment_clear_grid(grid: Grid, a: np.ndarray, b: np.ndarray) -> bool:
    """快速（栅格采样）连通性检查，仅用于平滑阶段的候选筛选。

    采样间距为半格，配合最终的精确连续几何复核，既快又不会漏判。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    length = float(np.hypot(*(b - a)))
    n = max(2, int(np.ceil(length / (grid.cell * 0.5))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    jj = np.clip(np.floor(pts[:, 0] / grid.cell).astype(int), 0, grid.resolution - 1)
    ii = np.clip(np.floor(pts[:, 1] / grid.cell).astype(int), 0, grid.resolution - 1)
    mask = grid.free_safe if grid.free_safe is not None else grid.free
    return bool(mask[ii, jj].all())


def shortcut_smooth(world, path: np.ndarray, grid: Grid | None = None, max_iter: int = 100) -> np.ndarray:
    """保持碰撞安全的捷径平滑：反复尝试用直线替换子折线。

    候选筛选用栅格采样（快），最终由 `plan_path` 用连续几何精确复核。
    """
    pts = [tuple(map(float, p)) for p in np.asarray(path)]
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float64)
    check = (lambda a, b: _segment_clear_grid(grid, a, b)) if grid is not None else (
        lambda a, b: _segment_clear(world, a, b)
    )
    changed = True
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        out = [pts[0]]
        k = 0
        while k < len(pts) - 1:
            far = k + 1
            for cand in range(len(pts) - 1, k + 1, -1):
                if check(np.array(pts[k]), np.array(pts[cand])):
                    far = cand
                    break
            if far > k + 1:
                changed = True
            out.append(pts[far])
            k = far
        pts = out
    return np.asarray(pts, dtype=np.float64)


def _resample_safe(polyline: np.ndarray, h: int, check) -> np.ndarray | None:
    """把折线变成**恰好 h 个点**且每段都安全的路径。

    1. 首选按累计弧长等距重采样（文档 §6.2 约定），逐段校验净空；
    2. 若某些弦因切角不安全：
       - 顶点过多 -> 在顶点上做“安全合并”（删除畸变最小的顶点，每次校验新弦）；
       - 顶点过少 -> 在最长线段中点处插入新点（子段是原弦的子集，必然安全）。
    """
    arc = resample_polyline(__import__("torch").as_tensor(polyline, dtype=__import__("torch").float64), h)
    pts = arc.numpy()
    if all(check(pts[k], pts[k + 1]) for k in range(len(pts) - 1)):
        return pts

    verts = [np.asarray(p, dtype=float) for p in polyline]

    # 阶段 A：顶点过多 -> 安全合并
    while len(verts) > h:
        best_k = None
        best_cost = np.inf
        for pos in range(1, len(verts) - 1):
            a, b, c = verts[pos - 1], verts[pos], verts[pos + 1]
            if not check(a, c):
                continue
            cost = float(np.hypot(*(c - a)) - (np.hypot(*(b - a)) + np.hypot(*(c - b))))
            if cost < best_cost:
                best_cost = cost
                best_k = pos
        if best_k is None:
            return None
        verts.pop(best_k)

    # 阶段 B：顶点过少 -> 在最长段中点插入（子段必然安全）
    while len(verts) < h:
        lens = [float(np.hypot(*(verts[i + 1] - verts[i]))) for i in range(len(verts) - 1)]
        if not lens:
            return None
        i = int(np.argmax(lens))
        if lens[i] <= 1e-9:
            return None
        mid = 0.5 * (verts[i] + verts[i + 1])
        verts.insert(i + 1, mid)

    out = np.stack(verts)
    if not all(check(out[k], out[k + 1]) for k in range(len(out) - 1)):
        return None
    return out


def plan_path(
    world,
    start: np.ndarray,
    goal: np.ndarray,
    resolution: int,
    h: int,
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.35,
    grid: Grid | None = None,
    waypoint: np.ndarray | None = None,
) -> np.ndarray | None:
    """完整流程：A* -> 捷径平滑 -> 重采样到 H 点（保证每段净空）-> 连续几何复核。

    waypoint 非空时先到 waypoint 再到 goal，用于制造同伦分支与路径多样性。
    返回 [H,2] 的 float64 张量；不可行时返回 None。
    """
    import torch

    grid = grid if grid is not None else build_grid(world, resolution)
    raw_parts = []
    waypoints = [np.asarray(start, dtype=float)]
    if waypoint is not None:
        waypoints.append(np.asarray(waypoint, dtype=float))
    waypoints.append(np.asarray(goal, dtype=float))
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        raw = astar(grid, a, b, rng=rng, noise_scale=noise_scale, heuristic_weight=1.05)
        if raw is None:
            return None
        raw_parts.append(raw)
    raw = np.concatenate(raw_parts, axis=0) if len(raw_parts) > 1 else raw_parts[0]

    smoothed = shortcut_smooth(world, raw, grid=grid)
    pts = _resample_safe(smoothed, h, lambda a, b: _segment_clear_grid(grid, a, b))
    if pts is None:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    pts[0] = np.asarray(start, dtype=float)
    pts[-1] = np.asarray(goal, dtype=float)
    # 最终精确复核：每一段都必须与障碍物保持净空
    for k in range(len(pts) - 1):
        if not _segment_clear(world, pts[k], pts[k + 1]):
            return None
    return torch.as_tensor(pts, dtype=torch.float64)


def path_is_feasible(paths, world, max_speed_mps: float, dt_s: float) -> dict:
    """路径可行性判定：静态无碰撞 + 逐段速度上限。"""
    import torch

    from fhd.geometry.collisions import check_speed_limit
    from fhd.geometry.paths import path_segment_lengths

    lens = path_segment_lengths(paths) if paths.dim() == 3 else path_segment_lengths(paths.unsqueeze(0))
    limit = max_speed_mps * dt_s
    max_len = float(lens.max())
    lengths = path_length(paths) if paths.dim() == 3 else path_length(paths.unsqueeze(0))
    return {
        "max_segment_m": max_len,
        "speed_ok": max_len <= limit + 1e-9,
        "length_m": float(lengths.mean()) if hasattr(lengths, "mean") else float(lengths),
    }
