"""S2/S3 共用的单机器人评估：按地图类型分别统计。

指标（方案 §5.3）：
  1. 单条轨迹有效率（静态安全 且 每段 ≤ v·Δt）
  2. Task Success@K（一个任务 K 条候选里至少 1 条有效）
  3. 静态碰撞率、速度约束通过率
  4. 最小障碍物净空
  5. 平均路径长度 / 相对参考路径的长度比
  6. 平滑度（二阶差分能量）
  7. 推理耗时
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from fhd.data.dataset import TrajectoryDataset
from fhd.diffusion.ddim import ddim_sample
from fhd.evaluation.statistics import cluster_bootstrap_ci
from fhd.geometry.collisions import check_speed_limit, trajectory_is_static_safe
from fhd.geometry.paths import path_length
from fhd.maps.simple_scenes import KIND_LABELS


def pick_tasks(ds: TrajectoryDataset, per_kind: int, seed: int = 7) -> list[dict]:
    """按地图类型分层抽样任务（同一种子 -> 同一批任务，供多个模型共用）。"""
    by_kind = defaultdict(list)
    for idx in range(len(ds)):
        by_kind[ds.maps[int(ds.data["map_idx"][idx])].kind].append(idx)
    rng = np.random.default_rng(seed)
    picked = []
    for kind in KIND_LABELS:
        pool = by_kind.get(kind, [])
        if not pool:
            continue
        sel = rng.choice(pool, size=min(per_kind, len(pool)), replace=False)
        for idx in sorted(int(i) for i in sel):
            picked.append({"task": idx, "kind": kind})
    return picked


@torch.no_grad()
def evaluate(model, schedule, ds: TrajectoryDataset, tasks: list[dict], n_samples: int,
             steps: int, v_max: float, dt_s: float, device, seed: int = 0) -> list[dict]:
    gen = torch.Generator(device=device.type).manual_seed(seed)
    rows = []
    for t in tasks:
        idx = t["task"]
        item = ds[idx]
        start = item["start"].unsqueeze(0).expand(n_samples, -1).to(device)
        goal = item["goal"].unsqueeze(0).expand(n_samples, -1).to(device)
        grid = item["grid"].unsqueeze(0).expand(n_samples, -1, -1, -1).to(device)
        t0 = time.time()
        samples = ddim_sample(model, schedule, start, goal, grid, num_steps=steps,
                              generator=gen)
        infer_s = (time.time() - t0) / n_samples
        paths = ds.denormalise(samples.float()).detach().cpu()
        world = ds.maps[int(item["map_idx"])]
        safe, clr = trajectory_is_static_safe(paths, world.obstacles(), world.robot_radius_m,
                                              world.clearance_margin_m)
        speed_ok, max_seg = check_speed_limit(paths, v_max, dt_s)
        lengths = path_length(paths)
        gt_len = float(path_length(ds.data["paths"][idx].unsqueeze(0))[0])
        accel = paths[:, 2:] - 2 * paths[:, 1:-1] + paths[:, :-2]
        ok = safe & speed_ok
        rows.append({
            "task": idx, "kind": t["kind"],
            "success_rate": float(ok.float().mean()),
            "task_success_at_k": float(bool(ok.any())),
            "static_safe_rate": float(safe.float().mean()),
            "speed_ok_rate": float(speed_ok.float().mean()),
            "min_clearance_m": float(clr.min()),
            "path_length_mean_m": float(lengths.mean()),
            "length_ratio": float((lengths / max(gt_len, 1e-6)).mean()),
            "smoothness": float(accel.pow(2).sum(-1).mean()),
            "max_segment_m": float(max_seg.max()),
            "infer_s_per_sample": infer_s,
            "n_samples": n_samples,
        })
    return rows


def summarize(rows: list[dict], tag: str) -> dict:
    vals = np.array([r["success_rate"] for r in rows])
    clusters = np.array([r["task"] for r in rows])
    ci = cluster_bootstrap_ci(vals, clusters, n_boot=2000)
    by_kind = {}
    for kind in KIND_LABELS:
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        by_kind[kind] = {
            "label": KIND_LABELS[kind],
            "n_tasks": len(sel),
            "success_rate": float(np.mean([r["success_rate"] for r in sel])),
            "task_success_at_k": float(np.mean([r["task_success_at_k"] for r in sel])),
            "static_safe_rate": float(np.mean([r["static_safe_rate"] for r in sel])),
            "speed_ok_rate": float(np.mean([r["speed_ok_rate"] for r in sel])),
            "min_clearance_m": float(np.mean([r["min_clearance_m"] for r in sel])),
            "length_ratio": float(np.mean([r["length_ratio"] for r in sel])),
            "smoothness": float(np.mean([r["smoothness"] for r in sel])),
            "infer_s_per_sample": float(np.mean([r["infer_s_per_sample"] for r in sel])),
        }
    return {
        "tag": tag, "n_tasks": len(rows),
        "success_rate": float(vals.mean()),
        "task_success_at_k": float(np.mean([r["task_success_at_k"] for r in rows])),
        "ci": ci,
        "static_safe_rate": float(np.mean([r["static_safe_rate"] for r in rows])),
        "speed_ok_rate": float(np.mean([r["speed_ok_rate"] for r in rows])),
        "length_ratio": float(np.mean([r["length_ratio"] for r in rows])),
        "smoothness": float(np.mean([r["smoothness"] for r in rows])),
        "infer_s_per_sample": float(np.mean([r["infer_s_per_sample"] for r in rows])),
        "by_kind": by_kind,
    }
