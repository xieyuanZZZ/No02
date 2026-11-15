"""credible_v2: 8 frozen tasks (4 open + 4 static), N=4."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_credible_scenes as base

OUT = Path(os.environ.get("CREDIBLE2_OUT", r"D:\paper2\_credible2_scenes"))
OUT.mkdir(parents=True, exist_ok=True)
T = base.task

tasks = []
# A: open intersections with different angles / asymmetry
tasks.append(T("A1_open_orthogonal", "open", [
    [(1, 3), (9, 3)], [(9, 7), (1, 7)],
    [(3, 1), (7, 8)], [(7, 1), (3, 8)]]))
tasks.append(T("A2_open_diagonal", "open", [
    [(1, 2), (9, 6)], [(9, 8), (1, 4)],
    [(2, 9), (6, 1)], [(8, 9), (4, 1)]]))
tasks.append(T("A3_open_multi_cross", "open", [
    [(1, 2), (9, 8)], [(1, 8), (9, 2)],
    [(2, 1), (8, 9)], [(2, 9), (8, 1)]]))
tasks.append(T("A4_open_asymmetric", "open", [
    [(1, 1), (9, 7)], [(1, 9), (9, 3)],
    [(1, 3), (9, 9)], [(1, 7), (9, 1)]]))
# B: static obstacles, two different obstacle layouts
rects_b1 = [(4.2, 4.2, 5.8, 5.8)]
tasks.append(T("B1_block_below_above", "static", [
    [(1, 3), (9, 3)], [(9, 7), (1, 7)],
    [(3, 1), (3, 9)], [(7, 9), (7, 1)]], rects_b1))
tasks.append(T("B2_block_cross", "static", [
    [(1, 2), (9, 2)], [(1, 8), (9, 8)],
    [(2, 1), (2, 9)], [(8, 9), (8, 1)]], rects_b1))
rects_b3 = [(2.2, 2.5, 3.8, 4.0), (6.2, 6.0, 7.8, 7.5)]
tasks.append(T("B3_two_blocks", "static", [
    [(1, 5), (9, 5)], [(9, 8.5), (1, 8.5)],
    [(4.4, 0.8), (4.4, 9.2)], [(6.0, 9.2), (6.0, 0.8)]], rects_b3))
rects_b4 = [(3.0, 2.0, 5.0, 3.2), (5.0, 6.2, 7.0, 7.4)]
tasks.append(T("B4_staggered_blocks", "static", [
    [(1, 4), (9, 4)], [(1, 8.8), (9, 8.8)],
    [(2, 1), (2, 9)], [(8, 9), (8, 1)]], rects_b4))

manifest = {"version": "credible_v2", "config": {
    "width": 10, "height": 10, "robot_radius": 0.15, "clearance": 0.03,
    "d_safe": 0.35, "vmax": 1.5, "amax": 2.0, "dt": 0.1,
    "dt_slot": 0.4, "horizon_pts": 32}, "tasks": tasks}
(OUT / "task_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
(OUT / "scene_validation.json").write_text(json.dumps({
    "tasks": len(tasks),
    "items": [{"task_id": t["task_id"], "scene": t["scene"],
               "min_clearance": t["min_path_obstacle_clearance"],
               "path_lengths": t["path_lengths"]} for t in tasks],
}, indent=1), encoding="utf-8")
print("wrote", len(tasks), "tasks", OUT)
