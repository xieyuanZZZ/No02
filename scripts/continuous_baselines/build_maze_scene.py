"""Build a 10x10 irregular maze scene with curved reference paths.

Five rectangular obstacles create staggered gaps; four agents traverse curved
paths with several crossings. Paths are checked against robot radius+clearance.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("MAZE_OUT", r"D:\paper2\_maze"))
OUT.mkdir(parents=True, exist_ok=True)
ROBOT_R = 0.15
CLEAR = 0.03
H = 64

RECTS = [
    (0.4, 3.3, 3.0, 3.5),
    (5.0, 3.3, 9.6, 3.5),
    (0.4, 7.2, 4.6, 7.4),
    (7.0, 7.2, 9.6, 7.4),
    (4.6, 4.3, 4.8, 6.3),
]


def catmull(pts, samples=12):
    p = np.asarray(pts, float)
    ext = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(q)
    out.append(p[-1])
    return np.asarray(out)


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


def rect_dist(q, r):
    x1, y1, x2, y2 = r
    dx = max(x1 - q[0], 0, q[0] - x2)
    dy = max(y1 - q[1], 0, q[1] - y2)
    return float(np.hypot(dx, dy))


def min_clearance(p):
    vals = [rect_dist(q, r) for q in p for r in RECTS]
    return float(min(vals))


def make_path(wp):
    p = resample(catmull(wp))
    c = min_clearance(p)
    if c < ROBOT_R + CLEAR + 0.01:
        raise ValueError(f"clearance {c:.3f}")
    return p, c


def main():
    r0 = [(1.0, 1.0), (1.4, 2.2), (3.8, 3.1), (3.8, 4.5), (3.8, 6.5),
          (5.8, 6.75), (5.8, 8.3), (8.8, 8.8)]
    r1 = list(reversed(r0))
    r2 = [(8.8, 1.0), (8.5, 2.5), (6.5, 3.0), (4.0, 3.35), (4.0, 5.0),
          (4.0, 6.5), (5.8, 6.75), (5.8, 8.0), (1.2, 8.8)]
    r3 = [(5.0, 0.6), (4.6, 1.5), (4.0, 2.7), (4.0, 3.35), (4.0, 5.0),
          (4.0, 6.5), (5.8, 6.75), (5.8, 8.2), (5.0, 9.4)]
    paths = []
    clears = []
    for wp in (r0, r1, r2, r3):
        p, c = make_path(wp)
        paths.append(p); clears.append(c)
    starts = np.stack([p[0] for p in paths])
    goals = np.stack([p[-1] for p in paths])
    dstart = np.linalg.norm(starts[:, None] - starts[None], axis=-1) + np.eye(4) * 99
    dgoal = np.linalg.norm(goals[:, None] - goals[None], axis=-1) + np.eye(4) * 99
    if dstart.min() < 0.35 or dgoal.min() < 0.35:
        raise ValueError("start/goal collision")
    task = {"task_id": "M1_maze_curves", "scene": "maze", "n_agents": 4,
            "starts": starts.tolist(), "goals": goals.tolist(),
            "paths": [p.tolist() for p in paths],
            "rects": [list(r) for r in RECTS], "circles": [],
            "path_lengths": [float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum()) for p in paths],
            "min_path_obstacle_clearance": min(clears)}
    manifest = {"version": "maze_v1",
                "config": {"width": 10, "height": 10, "robot_radius": ROBOT_R,
                           "clearance": CLEAR, "d_safe": 0.35, "vmax": 1.5,
                           "amax": 2.0, "dt": 0.1, "dt_slot": 0.4,
                           "horizon_pts": H},
                "tasks": [task]}
    (OUT / "task_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "scene_validation.json").write_text(json.dumps({
        "task_id": task["task_id"], "path_lengths": task["path_lengths"],
        "min_clearance": min(clears), "start_min": float(dstart.min()),
        "goal_min": float(dgoal.min()),
        "rects": RECTS,
    }, indent=1), encoding="utf-8")
    print("wrote", OUT, "clearance", min(clears), "lengths", task["path_lengths"])


if __name__ == "__main__":
    main()
