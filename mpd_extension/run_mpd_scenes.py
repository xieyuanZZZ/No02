"""把官方 MPD 接到 No02 的三类简单场景上（A: prior / B: mpd）。

做法：
  * 环境：用 `simple_scenes_adapter` 把我们 10×10 m 的地图线性映射到官方 [-1,1]²；
  * 机器人 / B-spline / 数据集：全部沿用官方对象（`RobotPointMass2D`、
    `ParametricTrajectoryBspline`、官方 EnvSimple2D 数据集做归一化）；
  * 规划器：官方 `GenerativeOptimizationPlanner`，`sampling_based_planner_fn=None`；
  * 起终点：直接用 S1 测试集的同一批 start/goal（映射到官方坐标）；
  * 碰撞检查：**连续几何**（点到矩形/圆的精确距离）。

输出：results_mpd/scenes_<planner>/metrics.json + 每任务明细。
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
sys.path.insert(0, str(ROOT / "mpd_extension"))

from mpd.inference.inference import GenerativeOptimizationPlanner  # noqa: E402
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml  # noqa: E402
from torch_robotics.tasks.tasks import PlanningTask  # noqa: E402
from torch_robotics.torch_utils.seed import fix_random_seed  # noqa: E402
from torch_robotics.torch_utils.torch_utils import to_numpy  # noqa: E402

from run_mpd_2d import TimedGuide, path_metrics  # noqa: E402
from simple_scenes_adapter import (  # noqa: E402
    SCALE,
    TARGET_HALF,
    build_env_from_scene,
    load_scene_tasks,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config_EnvSimple2D-RobotPointMass2D_00.yaml")
    ap.add_argument("--planner", default="diffusion_prior", choices=["diffusion_prior", "mpd"])
    ap.add_argument("--data", default="/mnt/d/paper2/data/revised_s1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-kind", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results_mpd"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()

    fix_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tensor_args = {"device": device, "dtype": torch.float32}

    cfg_path = ROOT / "scripts" / "inference" / "cfgs" / args.cfg
    args_inference = DotMap(load_params_from_yaml(str(cfg_path)))
    args_inference.model_dir = os.path.expandvars(args_inference.model_dir_ddpm_bspline)
    args_inference.planner_alg = args.planner
    args_inference.n_trajectory_samples = args.n_samples

    out_dir = Path(args.out) / f"scenes_{args.planner}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(**args_inference, gripper=True, reload_data=False,
                      results_dir=str(out_dir), load_indices=True, tensor_args=tensor_args)

    print("[1] 载入官方环境/机器人/数据集（只为拿 robot、bspline 与归一化）")
    base_task, train_subset, _, _, _ = get_planning_task_and_dataset(**args_train)
    robot = base_task.robot
    bspline = base_task.parametric_trajectory
    dataset = train_subset.dataset
    print(f"    robot={type(robot).__name__}  bspline n_cp={bspline.n_control_points if hasattr(bspline,'n_control_points') else '?'}")

    print("[2] 读 S1 测试集任务")
    tasks, maps = load_scene_tasks(args.data, args.split)
    rng = np.random.default_rng(args.seed)
    by_kind: dict[str, list] = {}
    for t in tasks:
        by_kind.setdefault(t.kind, []).append(t)
    chosen = []
    for kind, lst in by_kind.items():
        idxs = rng.choice(len(lst), size=min(args.per_kind, len(lst)), replace=False)
        for i in sorted(int(x) for x in idxs):
            chosen.append(lst[i])
    print(f"    选中 {len(chosen)} 个任务（每类 {args.per_kind}）")

    rows = []
    cur_map = None
    planner = None
    timed = None
    env = None
    for n, t in enumerate(chosen):
        if t.map_id != cur_map:
            cur_map = t.map_id
            env = build_env_from_scene(maps[t.map_id], tensor_args=tensor_args)
            pt = PlanningTask(
                parametric_trajectory=bspline, env=env, robot=robot,
                tensor_args=tensor_args, min_distance_robot_env=args.margin,
                obstacle_cutoff_margin=args.margin,
                margin_for_dense_collision_checking=0.0,
            )
            planner = GenerativeOptimizationPlanner(
                pt, dataset, args_train, args_inference, tensor_args,
                sampling_based_planner_fn=None, debug=False)
            timed = None
            if planner.cost_guide is not None:
                timed = TimedGuide(planner.cost_guide)
                planner.cost_guide = timed

        q_start = torch.as_tensor(t.start, **tensor_args)
        q_goal = torch.as_tensor(t.goal, **tensor_args)
        ee_goal = torch.eye(4, **tensor_args)[:3, :]
        if timed is not None:
            timed.total_s, timed.calls = 0.0, 0
        t0 = time.perf_counter()
        try:
            res = planner.plan_trajectory(q_start, q_goal, ee_goal,
                                          n_trajectory_samples=args.n_samples)
        except Exception as exc:  # noqa: BLE001
            print(f"    任务 {n} ({t.kind}) 规划异常：{type(exc).__name__}: {exc}")
            continue
        total_s = time.perf_counter() - t0

        trajs = res.q_trajs_pos_valid
        if trajs is None or len(trajs) == 0:
            rows.append({"task": n, "kind": t.kind, "map_id": t.map_id,
                         "n_candidates": 0, "success_rate": 0.0,
                         "task_success_at_k": False, "min_clearance": float("nan"),
                         "path_length_mean": float("nan"), "smoothness_mean": float("nan"),
                         "total_s": total_s, "guide_s": 0.0,
                         "diffusion_s": total_s, "guide_calls": 0})
            continue
        trajs = to_numpy(trajs)
        if trajs.ndim == 4:
            trajs = trajs[:, 0]

        circles, boxes = None, None
        sc = maps[t.map_id]
        rects = np.asarray(sc["rects"], dtype=np.float64).reshape(-1, 4)
        circ = np.asarray(sc["circles"], dtype=np.float64).reshape(-1, 3)
        # 转换到官方坐标
        def conv_xy(xy):
            return (np.asarray(xy, dtype=np.float64) / 10.0) * 2.0 * TARGET_HALF - TARGET_HALF
        boxes = np.concatenate([conv_xy(rects[:, :2]), conv_xy(rects[:, 2:])], axis=1) \
            if len(rects) else np.zeros((0, 4))
        circles = np.concatenate([conv_xy(circ[:, :2]), circ[:, 2:3] * SCALE], axis=1) \
            if len(circ) else np.zeros((0, 3))

        np.savez_compressed(
            out_dir / f"traj_task{n:03d}.npz",
            trajs=trajs.astype(np.float32),
            circles=circles.astype(np.float32),
            boxes=boxes.astype(np.float32),
            start=np.asarray(t.start, dtype=np.float32),
            goal=np.asarray(t.goal, dtype=np.float32),
            kind=np.array([t.kind]),
        )
        per = [path_metrics(trajs[k], circles, boxes, args.margin) for k in range(len(trajs))]
        ok = np.array([p["static_safe"] for p in per])
        guide_s = timed.total_s if timed is not None else 0.0
        rows.append({
            "task": n, "kind": t.kind, "map_id": t.map_id,
            "n_candidates": len(per),
            "success_rate": float(ok.mean()),
            "task_success_at_k": bool(ok.any()),
            "min_clearance": float(min(p["min_clearance"] for p in per)),
            "path_length_mean": float(np.mean([p["path_length"] for p in per])),
            "smoothness_mean": float(np.mean([p["smoothness"] for p in per])),
            "total_s": total_s, "guide_s": guide_s,
            "diffusion_s": max(total_s - guide_s, 0.0),
            "guide_calls": timed.calls if timed is not None else 0,
        })
        if (n + 1) % 5 == 0:
            print(f"    {n+1}/{len(chosen)}：{t.kind} 成功率 {rows[-1]['success_rate']:.2f} "
                  f"净空 {rows[-1]['min_clearance']:.3f} 总耗时 {total_s:.2f}s")

    summary = {"planner": args.planner, "n_tasks": len(rows),
               "n_samples": args.n_samples, "margin": args.margin,
               "scale": SCALE}
    for kind in by_kind:
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        summary[kind] = {
            "n": len(sel),
            "single_traj_safe_rate": float(np.mean([r["success_rate"] for r in sel])),
            "task_success_at_k": float(np.mean([r["task_success_at_k"] for r in sel])),
            "min_clearance_mean": float(np.nanmean([r["min_clearance"] for r in sel])),
            "path_length_mean": float(np.nanmean([r["path_length_mean"] for r in sel])),
            "total_s_mean": float(np.mean([r["total_s"] for r in sel])),
            "guide_s_mean": float(np.mean([r["guide_s"] for r in sel])),
        }
    allr = rows
    summary["overall"] = {
        "single_traj_safe_rate": float(np.mean([r["success_rate"] for r in allr])),
        "task_success_at_k": float(np.mean([r["task_success_at_k"] for r in allr])),
        "total_s_mean": float(np.mean([r["total_s"] for r in allr])),
        "total_s_p95": float(np.percentile([r["total_s"] for r in allr], 95)),
        "guide_s_mean": float(np.mean([r["guide_s"] for r in allr])),
        "diffusion_s_mean": float(np.mean([r["diffusion_s"] for r in allr])),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
