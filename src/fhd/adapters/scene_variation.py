"""Per-run scene jitter and A* reference-path regeneration.

The base task is frozen; for each (task, p, seed) we may move start/goal,
swap start/goal, and shift/resize obstacles. Paths are rebuilt with A* so the
reference path remains obstacle-free after the jitter.
"""
from __future__ import annotations
import heapq
import numpy as np

CELL = 0.1
W = H = 10.0
ROBOT_R, CLEAR = 0.15, 0.03
MARGIN = ROBOT_R + CLEAR + 0.10
H_PTS = 64


def _rect_dist(q, r):
    x1, y1, x2, y2 = r
    return float(np.hypot(max(x1 - q[0], 0, q[0] - x2),
                          max(y1 - q[1], 0, q[1] - y2)))


def _blocked(x, y, rects):
    return any(_rect_dist((x, y), r) < MARGIN for r in rects)


def _astar(start, goal, rects):
    if _blocked(*start, rects) or _blocked(*goal, rects):
        raise ValueError("start/goal in inflated obstacle")
    nx = int(W / CELL) + 1
    s = (int(round(start[0] / CELL)), int(round(start[1] / CELL)))
    g = (int(round(goal[0] / CELL)), int(round(goal[1] / CELL)))
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    heap = [(0.0, s)]; best = {s: 0.0}; parent = {}
    while heap:
        _, u = heapq.heappop(heap)
        if u == g:
            break
        for dx, dy in dirs:
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < nx and 0 <= v[1] < nx):
                continue
            q = (v[0] * CELL, v[1] * CELL)
            if _blocked(q[0], q[1], rects):
                continue
            ng = best[u] + (2 ** 0.5 if dx and dy else 1.0) * CELL
            if ng < best.get(v, 1e18) - 1e-12:
                best[v] = ng; parent[v] = u
                heapq.heappush(heap, (ng + np.hypot((v[0] - g[0]) * CELL,
                                                    (v[1] - g[1]) * CELL), v))
    if g not in parent and g != s:
        raise ValueError("A* no path")
    nodes = [g]
    while nodes[-1] != s:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    pts = np.array([(x * CELL, y * CELL) for x, y in nodes])
    out = [pts[0]]; i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            ts = np.linspace(0, 1, 20)
            seg = pts[i][None] * (1 - ts[:, None]) + pts[j][None] * ts[:, None]
            if all(not _blocked(q[0], q[1], rects) for q in seg):
                break
            j -= 1
        out.append(pts[j]); i = j
    return np.asarray(out)


def _resample(p, h=H_PTS):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], h)
    out = np.zeros((h, 2))
    for k, s in enumerate(t):
        j = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
        f = (s - cum[j]) / max(seg[j], 1e-12)
        out[k] = p[j] + f * (p[j + 1] - p[j])
    return out


def sample_variant(task, seed: int, jitter: float = 0.20, swap_p: float = 0.2):
    """Return a jittered task with newly planned A* paths, or raise."""
    rng = np.random.default_rng(int(seed))
    rects = []
    for x1, y1, x2, y2 in task["rects"]:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
        cx += rng.uniform(-0.12, 0.12); cy += rng.uniform(-0.12, 0.12)
        hw = max(0.25, hw + rng.uniform(-0.08, 0.08))
        hh = max(0.25, hh + rng.uniform(-0.08, 0.08))
        rects.append((cx - hw, cy - hh, cx + hw, cy + hh))
    starts = np.asarray(task["starts"], float).copy()
    goals = np.asarray(task["goals"], float).copy()
    for i in range(len(starts)):
        if rng.random() < swap_p:
            starts[i], goals[i] = goals[i], starts[i]
        starts[i] += rng.uniform(-jitter, jitter, 2)
        goals[i] += rng.uniform(-jitter, jitter, 2)
        starts[i] = np.clip(starts[i], 0.4, 9.6)
        goals[i] = np.clip(goals[i], 0.4, 9.6)
    paths = [_resample(_astar(s, g, rects)) for s, g in zip(starts, goals)]
    if np.min(np.linalg.norm(starts[:, None] - starts[None], axis=-1)
              + np.eye(4) * 99) < 0.35:
        raise ValueError("jittered start collision")
    if min((_rect_dist(q, r) for p in paths for q in p for r in rects),
           default=float("inf")) < ROBOT_R + CLEAR + 0.02:
        raise ValueError("jittered clearance")
    out = dict(task)
    out["starts"] = starts.tolist(); out["goals"] = goals.tolist()
    out["paths"] = [p.tolist() for p in paths]
    out["rects"] = [list(r) for r in rects]
    out["variant_seed"] = int(seed)
    return out


def sample_variant_retry(task, seed: int, tries: int = 20):
    for k in range(tries):
        try:
            return sample_variant(task, seed + k * 7919)
        except Exception:
            continue
    raise RuntimeError(f"could not generate variant for {task['task_id']}")
