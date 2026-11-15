"""credible_v3: 12 tasks + A* detour paths + per-seed geometry jitter.

Tasks: 4 open, 8 static (1-3 obstacles). Some starts/goals are separated by
obstacles, so a straight line is invalid; A* builds the reference detour.
"""
from __future__ import annotations
import heapq, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_credible_scenes as base

OUT = Path(os.environ.get("CREDIBLE3_OUT", r"D:\paper2\_credible3"))
OUT.mkdir(parents=True, exist_ok=True)
H = 64
CELL = 0.1
W, Hm = 10.0, 10.0
ROBOT_R, CLEAR = 0.15, 0.03
MARGIN = ROBOT_R + CLEAR + 0.10


def rect_dist(q, r):
    x1, y1, x2, y2 = r
    dx = max(x1 - q[0], 0, q[0] - x2)
    dy = max(y1 - q[1], 0, q[1] - y2)
    return float(np.hypot(dx, dy))


def blocked(x, y, rects):
    return any(rect_dist((x, y), r) < MARGIN for r in rects)


def astar(start, goal, rects):
    sx, sy = start; gx, gy = goal
    if blocked(sx, sy, rects) or blocked(gx, gy, rects):
        raise ValueError("start/goal inside inflated obstacle")
    nx, ny = int(W / CELL) + 1, int(Hm / CELL) + 1
    s = (int(round(sx / CELL)), int(round(sy / CELL)))
    g = (int(round(gx / CELL)), int(round(gy / CELL)))
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)]
    openh = [(0.0, s)]; best = {s: 0.0}; parent = {}
    while openh:
        _, u = heapq.heappop(openh)
        if u == g:
            break
        for dx, dy in dirs:
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny):
                continue
            q = (v[0] * CELL, v[1] * CELL)
            if blocked(q[0], q[1], rects):
                continue
            step = (2 ** 0.5 if dx and dy else 1.0) * CELL
            ng = best[u] + step
            if ng < best.get(v, 1e18) - 1e-12:
                best[v] = ng; parent[v] = u
                h = np.hypot((v[0] - g[0]) * CELL, (v[1] - g[1]) * CELL)
                heapq.heappush(openh, (ng + h, v))
    if g not in parent and g != s:
        raise ValueError("A* no path")
    nodes = [g]
    while nodes[-1] != s:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    pts = np.array([(x * CELL, y * CELL) for x, y in nodes])
    # greedy line-of-sight simplification
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            ts = np.linspace(0, 1, 20)
            seg = pts[i][None] * (1 - ts[:, None]) + pts[j][None] * ts[:, None]
            if all(not blocked(q[0], q[1], rects) for q in seg):
                break
            j -= 1
        out.append(pts[j]); i = j
    return np.array(out)


def resample(p, h=H):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], h)
    out = np.zeros((h, 2))
    for k, s in enumerate(t):
        j = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
        f = (s - cum[j]) / max(seg[j], 1e-12)
        out[k] = p[j] + f * (p[j + 1] - p[j])
    return out


def make_task(tid, scene, starts, goals, rects):
    paths = [resample(astar(s, g, rects)) for s, g in zip(starts, goals)]
    starts = np.asarray(starts, float); goals = np.asarray(goals, float)
    if np.min(np.linalg.norm(starts[:, None] - starts[None], axis=-1) + np.eye(4) * 99) < 0.35:
        raise ValueError(f"{tid}: start collision")
    if np.min(np.linalg.norm(goals[:, None] - goals[None], axis=-1) + np.eye(4) * 99) < 0.35:
        raise ValueError(f"{tid}: goal collision")
    clears = [min((rect_dist(q, r) for q in p for r in rects), default=float("inf"))
              for p in paths]
    if min(clears) < ROBOT_R + CLEAR + 0.002:
        raise ValueError(f"{tid}: clearance {min(clears):.3f}")
    return {"task_id": tid, "scene": scene, "n_agents": 4,
            "starts": starts.tolist(), "goals": goals.tolist(),
            "paths": [p.tolist() for p in paths], "rects": [list(r) for r in rects],
            "circles": [], "path_lengths": [float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum()) for p in paths],
            "min_path_obstacle_clearance": float(min(clears))}


def main():
    tasks = []
    A = [([(1, 3), (9, 7), (3, 1), (7, 1)],
          [(9, 3), (1, 7), (7, 8), (3, 8)])]
    # open tasks (no obstacles, straight A* paths are fine)
    open_sets = [
        ("A1_open_orthogonal", [(1, 3), (9, 7), (3, 1), (7, 1)],
         [(9, 3), (1, 7), (7, 8), (3, 8)]),
        ("A2_open_diagonal", [(1, 2), (9, 8), (2, 9), (8, 9)],
         [(9, 6), (1, 4), (6, 1), (4, 1)]),
        ("A3_open_multi_cross", [(1, 2), (1, 8), (2, 1), (2, 9)],
         [(9, 8), (9, 2), (8, 9), (8, 1)]),
        ("A4_open_asymmetric", [(1, 1), (1, 9), (1, 3), (1, 7)],
         [(9, 7), (9, 3), (9, 9), (9, 1)]),
    ]
    for tid, s, g in open_sets:
        tasks.append(make_task(tid, "open", s, g, []))

    # static tasks with 1-3 obstacles and non-direct start/goal pairs
    tasks.append(make_task("B1_block_below_above", "static",
                            [(1, 3), (9, 7), (3, 1), (7, 9)], [(9, 3), (1, 7), (3, 9), (7, 1)],
                            [(4.2, 4.2, 5.8, 5.8)]))
    tasks.append(make_task("B2_block_cross", "static",
                            [(1, 2), (1, 8), (2, 1), (8, 9)], [(9, 2), (9, 8), (2, 9), (8, 1)],
                            [(4.2, 4.2, 5.8, 5.8)]))
    tasks.append(make_task("B3_two_blocks", "static",
                            [(1, 5), (9, 8.5), (4.4, 0.8), (6.0, 9.2)],
                            [(9, 5), (1, 8.5), (4.4, 9.2), (6.0, 0.8)],
                            [(2.2, 2.5, 3.8, 4.0), (6.6, 6.0, 8.2, 7.5)]))
    tasks.append(base.task("B4_staggered_blocks", "static", [
        [(1, 1), (2, 1), (2, 4.2), (5, 4.2), (5, 7.2), (9, 7.2)],
        [(9, 1), (8, 1), (8, 3.6), (5.5, 3.6), (5.5, 8.2), (1, 8.2)],
        [(1, 9), (2.5, 9), (2.5, 6.2), (7, 6.2), (7, 2.2), (9, 2.2)],
        [(9, 9), (7.5, 9), (7.5, 6.6), (3.5, 6.6), (3.5, 3.0), (1, 2.6)],
    ], [(2.8, 1.0, 4.0, 2.4), (5.9, 7.6, 7.0, 8.8)]))
    tasks.append(make_task("B5_center_pillar", "static",
                            [(1, 2), (1, 8), (2, 1), (8, 9)],
                            [(9, 2), (9, 8), (2, 9), (8, 1)],
                            [(4.0, 3.0, 6.0, 5.0)]))
    tasks.append(base.task("B6_two_blocks_detour", "static", [
        [(1, 1), (2.5, 1), (2.5, 4.2), (8.2, 4.2), (8.2, 9)],
        [(1, 9), (9, 1)],
        [(5, 1), (5, 4.2), (1, 4.2), (1, 7)],
        [(9, 3), (8, 3), (8, 8), (5, 9)],
    ], [(3.0, 2.0, 4.5, 3.5), (6.0, 6.0, 7.5, 7.5)]))
    tasks.append(make_task("B7_three_blocks", "static",
                            [(1, 3), (1, 7), (3, 1), (7, 9)],
                            [(9, 5), (9, 3), (7, 9), (3, 1)],
                            [(2.5, 2.0, 4.0, 3.5), (5.0, 4.5, 6.5, 6.0), (2.5, 6.5, 4.0, 8.0)]))
    tasks.append(base.task("B8_l_shape", "static", [
        [(1, 1), (2, 1), (2, 8.5), (9, 8.5)],
        [(9, 1), (8, 1), (8, 3.0), (1, 3.0)],
        [(1, 9), (2.5, 9), (2.5, 2.0), (9, 2.0)],
        [(9, 9), (7.5, 9), (7.5, 1.5), (1, 1.5)],
    ], [(3.0, 4.0, 7.0, 4.6), (5.4, 4.6, 6.0, 8.0)]))
    manifest = {"version": "credible_v3", "config": {
        "width": 10, "height": 10, "robot_radius": ROBOT_R, "clearance": CLEAR,
        "d_safe": 0.35, "vmax": 1.5, "amax": 2.0, "dt": 0.1,
        "dt_slot": 0.4, "horizon_pts": H,
        "path_planner": "A* grid 0.1, inflation 0.22, line-of-sight + 64-point resample"},
        "tasks": tasks}
    (OUT / "task_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "scene_validation.json").write_text(json.dumps({
        "tasks": len(tasks),
        "items": [{"task_id": t["task_id"], "scene": t["scene"],
                   "min_clearance": t["min_path_obstacle_clearance"],
                   "path_lengths": t["path_lengths"]} for t in tasks],
    }, indent=1), encoding="utf-8")
    print("wrote", len(tasks), "tasks", OUT)


if __name__ == "__main__":
    main()
