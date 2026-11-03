"""用官方 MPD 库跑 2D 点质量机器人规划（A: diffusion_prior / B: mpd）。

为什么不用官方 `scripts/inference/inference.py` 直接跑：
  该脚本**无条件**构造 `EvaluationSamplesGenerator`，而它内部会实例化
  `GenerateDataOMPL` → `PbOMPLRobot`，需要 OMPL 的 Python 绑定（CMake 编译）。
  本机没有、且与我们能拿到的权重无关。

本驱动做的事与官方脚本**完全一致**的部分：
  * 同一份 `cfg_inference_path`（模型选择、DDIM 参数、代价权重、引导步数）；
  * 同一个 `get_planning_task_and_dataset`（官方环境 + 官方数据集 + 归一化统计）；
  * 同一个 `GenerativeOptimizationPlanner`、同一个 `CostGuideManagerParametricTrajectory`
    （即官方 SDF 几何代价引导）；
  * 同一个 `plan_trajectory` 采样与最优轨迹选择。

不同的部分（必须记录在报告里）：
  * 起终点取自官方数据集（`selection_start_goal` 的等价做法），
    而不是由 pybullet+OMPL 随机采样；
  * 跳过了 pybullet/Isaac Gym 可视化与 rrtconnect 基线（`sampling_based_planner_fn=None`）。

碰撞检查用**连续几何**（点到圆/矩形的精确距离），不用栅格 SDF。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotmap import DotMap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wsl_setup" / "shims"))
sys.path.insert(0, "/mnt/d/paper2/src")

from mpd.inference.inference import GenerativeOptimizationPlanner  # noqa: E402
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml  # noqa: E402
from torch_robotics.torch_utils.seed import fix_random_seed  # noqa: E402
from torch_robotics.torch_utils.torch_utils import to_numpy  # noqa: E402


# --------------------------------------------------------------------------- #
# 连续几何碰撞检查
# --------------------------------------------------------------------------- #
def extract_primitives(env):
    """从官方环境里抽出障碍物：(circles[N,3], boxes[M,4])。"""
    circles, boxes = [], []
    fields = []
    for attr in ("obj_fixed_list", "obj_list"):
        if hasattr(env, attr):
            fields.extend(getattr(env, attr))
    for f in fields:
        inner = getattr(f, "fields", [f])
        for g in inner:
            name = type(g).__name__
            if "Sphere" in name or "Circle" in name:
                c = to_numpy(g.centers).reshape(-1, g.centers.shape[-1])
                r = to_numpy(g.radii).reshape(-1)
                if c.shape[-1] == 2:
                    for i in range(len(r)):
                        circles.append([c[i, 0], c[i, 1], float(r[i])])
            elif "Box" in name:
                c = to_numpy(g.centers).reshape(-1, g.centers.shape[-1])
                s = to_numpy(g.sizes).reshape(-1, g.sizes.shape[-1])
                if c.shape[-1] == 2:
                    for i in range(len(s)):
                        boxes.append([c[i, 0] - s[i, 0] / 2, c[i, 1] - s[i, 1] / 2,
                                      c[i, 0] + s[i, 0] / 2, c[i, 1] + s[i, 1] / 2])
    return np.asarray(circles, dtype=np.float64).reshape(-1, 3), \
        np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def point_clearance(pts: np.ndarray, circles: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """每个点到最近障碍物边界的**精确**距离（连续几何，正值表示在障碍外）。"""
    d = np.full(len(pts), np.inf)
    if len(circles):
        c = circles[:, :2][None, :, :]
        r = circles[:, 2][None, :]
        dc = np.linalg.norm(pts[:, None, :] - c, axis=-1) - r
        d = np.minimum(d, dc.min(axis=1))
    if len(boxes):
        lo = boxes[:, :2][None, :, :]
        hi = boxes[:, 2:][None, :, :]
        p = pts[:, None, :]
        inside = np.all((p >= lo) & (p <= hi), axis=-1)
        dx = np.maximum(np.maximum(lo[..., 0] - p[..., 0], p[..., 0] - hi[..., 0]), 0.0)
        dy = np.maximum(np.maximum(lo[..., 1] - p[..., 1], p[..., 1] - hi[..., 1]), 0.0)
        dbox = np.sqrt(dx ** 2 + dy ** 2)
        dbox = np.where(inside, -np.minimum.reduce([
            p[..., 0] - lo[..., 0], hi[..., 0] - p[..., 0],
            p[..., 1] - lo[..., 1], hi[..., 1] - p[..., 1]]), dbox)
        d = np.minimum(d, dbox.min(axis=1))
    return d


def path_metrics(traj: np.ndarray, circles, boxes, margin: float) -> dict:
    clr = point_clearance(traj, circles, boxes)
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    return {
        "static_safe": bool(clr.min() > margin),
        "min_clearance": float(clr.min()),
        "path_length": float(seg.sum()),
        "smoothness": float(np.sum(np.diff(traj, n=2, axis=0) ** 2)),
    }


# --------------------------------------------------------------------------- #
class TimedGuide:
    """包装官方 CostGuideManager，只统计引导耗时与调用次数（不改其行为）。"""

    def __init__(self, inner):
        self.inner = inner
        self.total_s = 0.0
        self.calls = 0

    def __call__(self, *a, **k):
        t0 = time.perf_counter()
        out = self.inner(*a, **k)
        self.total_s += time.perf_counter() - t0
        self.calls += 1
        return out

    def warmup(self, *a, **k):
        return self.inner.warmup(*a, **k)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config_EnvSimple2D-RobotPointMass2D_00.yaml")
    ap.add_argument("--planner", default="diffusion_prior", choices=["diffusion_prior", "mpd"])
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results_mpd"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    fix_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tensor_args = {"device": device, "dtype": torch.float32}

    cfg_path = ROOT / "scripts" / "inference" / "cfgs" / args.cfg
    args_inference = DotMap(load_params_from_yaml(str(cfg_path)))
    args_inference.model_dir = os.path.expandvars(args_inference.model_dir_ddpm_bspline)
    args_inference.planner_alg = args.planner
    args_inference.n_trajectory_samples = args.n_samples

    out_dir = Path(args.out) / f"{args.planner}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(**args_inference, gripper=True, reload_data=False,
                      results_dir=str(out_dir), load_indices=True, tensor_args=tensor_args)

    print(f"[1] 环境与数据集：{args_inference.get('env_id_replace') or 'EnvSimple2D'}")
    planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)
    circles, boxes = extract_primitives(planning_task.env)
    print(f"    障碍物：{len(circles)} 个圆，{len(boxes)} 个矩形")

    print(f"[2] 构造官方 planner（planner_alg={args.planner}）")
    planner = GenerativeOptimizationPlanner(
        planning_task, train_subset.dataset, args_train, args_inference, tensor_args,
        sampling_based_planner_fn=None, debug=False,
    )
    timed = None
    if planner.cost_guide is not None:
        timed = TimedGuide(planner.cost_guide)
        planner.cost_guide = timed
        print("    引导已启用（官方 SDF 代价）")
    else:
        print("    引导未启用（纯扩散先验）")

    margin = float(getattr(planning_task, "min_distance_robot_env", 0.0))
    robot = planning_task.robot
    r_attr = None
    for name in ("radius", "robot_radius", "radius_m"):
        if hasattr(robot, name):
            r_attr = float(getattr(robot, name))
            break
    print(f"    连续碰撞裕度：min_distance_robot_env={margin}, robot_radius={r_attr}")
    margin_total = margin + (r_attr or 0.0)

    print(f"[3] 评估 {args.n_tasks} 个任务，每个 {args.n_samples} 条候选")
    rows = []
    dss = val_subset if len(val_subset) > 0 else train_subset
    for i in range(args.n_tasks):
        idx = int(np.random.randint(len(dss)))
        sample = dss[idx]
        q_start = sample["q_start"]
        q_goal = sample["q_goal"]
        ee_goal = sample.get("ee_goal_pose", None)

        if timed is not None:
            timed.total_s, timed.calls = 0.0, 0
        t0 = time.perf_counter()
        res = planner.plan_trajectory(q_start, q_goal, ee_goal,
                                      n_trajectory_samples=args.n_samples)
        total_s = time.perf_counter() - t0

        trajs = res.q_trajs_pos_valid
        if trajs is None or len(trajs) == 0:
            print(f"    任务 {i}: 无有效候选（被官方 cost 判为无效）")
            continue
        trajs = to_numpy(trajs)              # [K, T, 2]
        if trajs.ndim == 4:
            trajs = trajs[:, 0]

        per = [path_metrics(trajs[k], circles, boxes, margin_total) for k in range(len(trajs))]
        ok = np.array([p["static_safe"] for p in per])
        guide_s = timed.total_s if timed is not None else 0.0
        rows.append({
            "task": i, "subset_idx": idx,
            "n_candidates": len(per),
            "success_rate": float(ok.mean()),
            "task_success_at_k": bool(ok.any()),
            "min_clearance": float(min(p["min_clearance"] for p in per)),
            "path_length_mean": float(np.mean([p["path_length"] for p in per])),
            "smoothness_mean": float(np.mean([p["smoothness"] for p in per])),
            "total_s": total_s,
            "guide_s": guide_s,
            "diffusion_s": max(total_s - guide_s, 0.0),
            "guide_calls": timed.calls if timed is not None else 0,
        })
        if (i + 1) % 5 == 0:
            print(f"    任务 {i+1}/{args.n_tasks}：成功率 {rows[-1]['success_rate']:.2f}，"
                  f"总耗时 {total_s:.2f}s，引导 {guide_s:.2f}s")

    if not rows:
        print("没有成功规划的任务")
        return 1

    first = rows[0]["n_candidates"]
    summary = {
        "planner": args.planner,
        "n_tasks": len(rows),
        "n_samples": args.n_samples,
        "single_traj_safe_rate": float(np.mean([r["success_rate"] for r in rows])),
        "task_success_at_k": float(np.mean([r["task_success_at_k"] for r in rows])),
        "min_clearance_mean": float(np.mean([r["min_clearance"] for r in rows])),
        "path_length_mean": float(np.mean([r["path_length_mean"] for r in rows])),
        "total_s_mean": float(np.mean([r["total_s"] for r in rows])),
        "total_s_p95": float(np.percentile([r["total_s"] for r in rows], 95)),
        "guide_s_mean": float(np.mean([r["guide_s"] for r in rows])),
        "diffusion_s_mean": float(np.mean([r["diffusion_s"] for r in rows])),
        "margin_used": margin_total,
        "n_circles": int(len(circles)),
        "n_boxes": int(len(boxes)),
        "first_task_candidates": first,
    }
    (out_dir / "metrics.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
