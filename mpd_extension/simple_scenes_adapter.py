"""把 No02 的 S1 三类场景接到官方 MPD 的 2D 环境上。

官方 2D 世界：`limits = [[-1, -1], [1, 1]]`（点质量机器人，2 维）。
我们的 S1 世界：10 m × 10 m，矩形/圆障碍，机器人半径 0.15 m，安全距离 d_safe=0.35 m。

做法：
  * 保持**相对几何不变**，把 10×10 m 线性映射到 [-1,1]²（比例 s = 2/10 = 0.2）；
  * 机器人半径、安全距离同比例缩放（0.15→0.03，0.35→0.07），
    这样"是否碰撞/是否够安全"在原尺度与新尺度下等价；
  * 障碍物用官方 `MultiBoxField`（矩形）与 `MultiSphereField`（圆）表达，
    与 `EnvSimple2D` 的构造方式一致，因此可以直接沿用官方的 SDF 预计算与碰撞代价。

注意：坐标缩放不改变拓扑与相对位置，但会改变数值尺度；比较实验时
必须在同一缩放下进行，并在报告里注明。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import MultiBoxField, MultiSphereField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS

WORLD_M = 10.0          # 我们的世界边长
TARGET_HALF = 1.0       # 官方 2D 世界半边长（limits ±1）
SCALE = TARGET_HALF / (WORLD_M / 2.0)   # 0.2


def _to_official(xy: np.ndarray) -> np.ndarray:
    """我们的米制坐标（0..10）-> 官方坐标（-1..1）。"""
    return (np.asarray(xy, dtype=np.float64) / WORLD_M) * 2.0 * TARGET_HALF - TARGET_HALF


@dataclass
class SceneTask:
    """一个任务：地图 + 起终点（都是官方坐标）。"""

    map_id: str
    kind: str
    start: np.ndarray        # [2] 官方坐标
    goal: np.ndarray         # [2] 官方坐标
    start_m: np.ndarray      # [2] 米制
    goal_m: np.ndarray


def load_scene_maps(data_dir: str | Path, split: str = "test") -> list[dict]:
    p = Path(data_dir) / f"maps_{split}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def build_env_from_scene(scene: dict, tensor_args=DEFAULT_TENSOR_ARGS):
    """用我们的一张地图构造官方风格的环境对象。"""
    rects = np.asarray(scene["rects"], dtype=np.float64).reshape(-1, 4)
    circles = np.asarray(scene["circles"], dtype=np.float64).reshape(-1, 3)
    obj_list = []
    if len(rects):
        lows = _to_official(rects[:, :2])
        highs = _to_official(rects[:, 2:])
        obj_list.append(MultiBoxField(lows, highs, tensor_args=tensor_args))
    if len(circles):
        centers = _to_official(circles[:, :2])
        radii = circles[:, 2] * SCALE
        obj_list.append(MultiSphereField(centers, radii, tensor_args=tensor_args))
    if not obj_list:
        # 与官方 EnvEmpty2D 一致：空场景放一个极小半径的哑元球，
        # 否则 GridMapSDF 的 sdf 为 None，vmap 直接报错。
        obj_list.append(MultiSphereField(np.array([[0.0, 0.0]]), np.array([1e-4]),
                                         tensor_args=tensor_args))

    class _SceneEnv(EnvBase):
        def __init__(self):
            super().__init__(
                limits=torch.tensor([[-TARGET_HALF, -TARGET_HALF],
                                     [TARGET_HALF, TARGET_HALF]], **tensor_args),
                obj_fixed_list=[ObjectField(obj_list, "dense2d")] if obj_list else [],
                precompute_sdf_obj_fixed=True,
                sdf_cell_size=0.005,
                tensor_args=tensor_args,
            )

    return _SceneEnv()


def load_scene_tasks(data_dir: str | Path, split: str = "test",
                     kinds: tuple[str, ...] | None = None) -> list[SceneTask]:
    """读 S1 的 tasks_<split>.jsonl，转成官方坐标。"""
    data_dir = Path(data_dir)
    maps = {m["map_id"] if "map_id" in m else f"{m['kind']}_{m['seed']}": m
            for m in load_scene_maps(data_dir, split)}
    tasks = []
    with (data_dir / f"tasks_{split}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            if kinds and t["kind"] not in kinds:
                continue
            start_m = np.asarray(t["start"], dtype=np.float64)
            goal_m = np.asarray(t["goal"], dtype=np.float64)
            tasks.append(SceneTask(
                map_id=t["map_id"], kind=t["kind"],
                start=_to_official(start_m), goal=_to_official(goal_m),
                start_m=start_m, goal_m=goal_m,
            ))
    return tasks, maps


def scene_summary(data_dir: str | Path, split: str = "test") -> dict:
    """快速统计：每类场景的地图数 / 任务数 / 障碍物数量分布。"""
    maps = load_scene_maps(data_dir, split)
    out: dict[str, dict] = {}
    for m in maps:
        k = m["kind"]
        d = out.setdefault(k, {"maps": 0, "rects": 0, "circles": 0})
        d["maps"] += 1
        d["rects"] += len(np.asarray(m["rects"]).reshape(-1, 4))
        d["circles"] += len(np.asarray(m["circles"]).reshape(-1, 3))
    return out


if __name__ == "__main__":
    import sys

    data = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/paper2/data/revised_s1"
    print(json.dumps(scene_summary(data), ensure_ascii=False, indent=2))
    tasks, maps = load_scene_tasks(data)
    print("tasks:", len(tasks), "maps:", len(maps))
    env = build_env_from_scene(next(iter(maps.values())))
    print("env limits:", env.limits.tolist())
