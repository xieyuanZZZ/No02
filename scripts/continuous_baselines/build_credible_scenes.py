"""Build the 3x2 credible scene set (open / static obstacles / narrow corridor).

Paths are common fixed reference paths; this is not a Diffusion planning claim.
The manifest is frozen before any method is run.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import os

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("CREDIBLE_OUT", str(ROOT / "results/Continuous_Baseline_Comparison/credible_v1")))
W, H = 10.0, 10.0
ROBOT_R = 0.15
CLEAR = 0.03
N = 4
H_PTS = 32


def polyline(pts, h=H_PTS):
    p = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], h)
    out = np.zeros((h, 2))
    for k, s in enumerate(t):
        j = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
        f = (s - cum[j]) / max(seg[j], 1e-12)
        out[k] = p[j] + f * (p[j + 1] - p[j])
    return out


def dist_rect(pt, r):
    x1, y1, x2, y2 = r
    dx = max(x1 - pt[0], 0, pt[0] - x2)
    dy = max(y1 - pt[1], 0, pt[1] - y2)
    return float(np.hypot(dx, dy))


def clearance(paths, rects, circles):
    vals = []
    for p in paths:
        for q in p:
            for r in rects:
                vals.append(dist_rect(q, r))
            for c in circles:
                vals.append(np.hypot(q[0] - c[0], q[1] - c[1]) - c[2])
    return float(min(vals)) if vals else float("inf")


def task(tid, scene, paths, rects=(), circles=()):
    paths = [polyline(p) for p in paths]
    starts = np.stack([p[0] for p in paths])
    goals = np.stack([p[-1] for p in paths])
    if np.min(np.linalg.norm(starts[:, None] - starts[None], axis=-1) + np.eye(N) * 99) <= 0.35:
        raise ValueError(f"{tid}: start collision")
    if np.min(np.linalg.norm(goals[:, None] - goals[None], axis=-1) + np.eye(N) * 99) <= 0.35:
        raise ValueError(f"{tid}: goal collision")
    c = clearance(paths, rects, circles)
    if c < ROBOT_R + CLEAR + 0.02:
        raise ValueError(f"{tid}: path obstacle clearance {c:.3f}")
    return {"task_id": tid, "scene": scene, "n_agents": N,
            "starts": starts.tolist(), "goals": goals.tolist(),
            "paths": [p.tolist() for p in paths], "rects": [list(r) for r in rects],
            "circles": [list(c) for c in circles],
            "path_lengths": [float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum()) for p in paths],
            "min_path_obstacle_clearance": c}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    # A: open crossings (two distinct geometries)
    tasks.append(task("A1_open_orthogonal", "open", [
        [(1, 5), (9, 5)], [(9, 5), (1, 5)],
        [(5, 1), (5, 9)], [(5, 9), (5, 1)]]))
    tasks.append(task("A2_open_diagonal", "open", [
        [(1, 1), (9, 9)], [(9, 9), (1, 1)],
        [(1, 9), (9, 1)], [(9, 1), (1, 9)]]))
    # B: static obstacles; paths run around a central block
    rects_b = [(4.2, 4.2, 5.8, 5.8)]
    tasks.append(task("B1_block_below_above", "static", [
        [(1, 3), (9, 3)], [(9, 7), (1, 7)],
        [(3, 1), (3, 9)], [(7, 9), (7, 1)]], rects_b))
    tasks.append(task("B2_block_cross", "static", [
        [(1, 2), (9, 2)], [(1, 8), (9, 8)],
        [(2, 1), (2, 9)], [(8, 9), (8, 1)]], rects_b))
    # C: narrow corridor (single-file); walls leave a 0.6 m channel
    rects_c = [(4.7, 2.5, 4.72, 7.5), (5.28, 2.5, 5.30, 7.5)]
    tasks.append(task("C1_corridor_4way", "narrow", [
        [(3.2, 0.3), (4.2, 1.4), (4.2, 2.0), (5.0, 2.0), (5.0, 8.0), (5.8, 8.6), (6.8, 9.7)],
        [(6.8, 9.7), (5.8, 8.6), (5.0, 8.0), (5.0, 2.0), (4.2, 2.0), (4.2, 1.4), (3.2, 0.3)],
        [(6.8, 0.3), (5.8, 1.4), (5.8, 2.0), (5.0, 2.0), (5.0, 8.0), (4.2, 8.6), (3.2, 9.7)],
        [(3.2, 9.7), (4.2, 8.6), (5.0, 8.0), (5.0, 2.0), (5.8, 2.0), (5.8, 1.4), (6.8, 0.3)],
    ], rects_c))
    tasks.append(task("C2_corridor_pair", "narrow", [
        [(2.5, 0.5), (4.0, 1.5), (4.0, 2.0), (5.0, 2.0), (5.0, 8.0), (6.0, 8.5), (7.5, 9.5)],
        [(7.5, 9.5), (6.0, 8.5), (5.0, 8.0), (5.0, 2.0), (4.0, 2.0), (4.0, 1.5), (2.5, 0.5)],
        [(7.5, 0.5), (6.0, 1.5), (6.0, 2.0), (5.0, 2.0), (5.0, 8.0), (4.0, 8.5), (2.5, 9.5)],
        [(2.5, 9.5), (4.0, 8.5), (5.0, 8.0), (5.0, 2.0), (6.0, 2.0), (6.0, 1.5), (7.5, 0.5)],
    ], rects_c))
    manifest = {"version": "credible_v1", "config": {"width": W, "height": H,
                "robot_radius": ROBOT_R, "clearance": CLEAR,
                "d_safe": 0.35, "vmax": 1.5, "amax": 2.0, "dt": 0.1,
                "dt_slot": 0.4, "horizon_pts": H_PTS},
                "tasks": tasks}
    (OUT / "task_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "scene_validation.json").write_text(json.dumps({
        "tasks": len(tasks),
        "all_clearance_ok": all(t["min_path_obstacle_clearance"] >= 0.20 for t in tasks),
        "items": [{"task_id": t["task_id"], "min_clearance": t["min_path_obstacle_clearance"],
                   "path_lengths": t["path_lengths"]} for t in tasks],
    }, indent=1), encoding="utf-8")
    print("wrote", len(tasks), "tasks", OUT)


if __name__ == "__main__":
    main()
