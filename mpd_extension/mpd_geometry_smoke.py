"""几何引导冒烟测试：1 台 / 2 台 / 4 台的同步 DDIM，含引导前后对比。

只跑少量任务、少量候选，用于排除接口错误，不做正式实验。

用法（WSL/mpd 环境）：
  bash /mnt/d/mpd_official/wsl_setup/run_geometry_smoke.sh [N]
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

from first_hit_geometry_guide import FirstHitGeometryGuide, SumGuide  # noqa: E402
from multi_robot_mpd_sampler import MultiRobotSampler, official_to_meters  # noqa: E402
from simple_scenes_adapter import build_env_from_scene, load_scene_tasks  # noqa: E402

sys.path.insert(0, "/mnt/d/paper2/scripts")


def load_risk():
    from fhd.risk.multi_pair import FirstHitRisk
    from fhd.utils.config import load_config
    cfg = load_config("/mnt/d/paper2/configs/base.yaml")
    return FirstHitRisk.from_config(cfg), cfg


def static_safe(paths_m: np.ndarray, circles, boxes, margin=0.0) -> tuple[bool, float]:
    """连续几何静态安全检查（米制）。"""
    d = np.full(paths_m.shape[:-1], np.inf)
    pts = paths_m.reshape(-1, 2)
    if len(circles):
        c = circles[:, :2][None, :, :]
        r = circles[:, 2][None, :]
        dc = np.linalg.norm(pts[:, None, :] - c, axis=-1) - r
        d = np.minimum(d.reshape(-1), dc.min(axis=1)).reshape(paths_m.shape[:-1])
    if len(boxes):
        lo = boxes[:, :2][None, :, :]
        hi = boxes[:, 2:][None, :, :]
        p = pts[:, None, :]
        inside = np.all((p >= lo) & (p <= hi), axis=-1)
        dx = np.maximum(np.maximum(lo[..., 0] - p[..., 0], p[..., 0] - hi[..., 0]), 0.0)
        dy = np.maximum(np.maximum(lo[..., 1] - p[..., 1], p[..., 1] - hi[..., 1]), 0.0)
        dbox = np.sqrt(dx ** 2 + dy ** 2)
        dbox = np.where(inside, -1.0, dbox)
        d = np.minimum(d.reshape(-1), dbox.min(axis=1)).reshape(paths_m.shape[:-1])
    return bool((d > margin).all()), float(d.min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--stagger", type=int, default=3)
    ap.add_argument("--lam-R", type=float, default=1.0)
    ap.add_argument("--lam-D", type=float, default=1.0)
    ap.add_argument("--pause-p", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--n-guide-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "results_mpd" / "geometry_smoke"))
    args = ap.parse_args()

    fix_random_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_args = {"device": device, "dtype": torch.float32}
    n = int(args.agents)

    cfg_path = ROOT / "scripts" / "inference" / "cfgs" / "config_EnvSimple2D-RobotPointMass2D_00.yaml"
    args_inference = DotMap(load_params_from_yaml(str(cfg_path)))
    args_inference.model_dir = os.path.expandvars(args_inference.model_dir_ddpm_bspline)
    args_inference.planner_alg = "mpd"
    args_inference.ddim_sampling_timesteps = args.steps
    args_inference.n_trajectory_samples = args.samples
    args_inference.n_guide_steps = args.n_guide_steps
    out_dir = Path(args.out) / f"n{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(**args_inference, gripper=True, reload_data=False,
                      results_dir=str(out_dir), load_indices=True, tensor_args=tensor_args)

    print(f"[1] 载入官方 MPD（N={n}，B={args.samples}）")
    base_task, train_subset, _, _, _ = get_planning_task_and_dataset(**args_train)
    tasks, maps = load_scene_tasks("/mnt/d/paper2/data/revised_s1", "test")
    risk, risk_cfg = load_risk()
    d_safe = risk.d_safe
    T = int(risk_cfg["execution"]["horizon_steps"])
    H = int(risk_cfg["trajectory"]["horizon"])

    # 取同一张地图上的前 n 个**互不相同**的任务
    map_id = tasks[0].map_id
    seen, group = set(), []
    for t in tasks:
        if t.map_id != map_id:
            continue
        key = (tuple(np.round(t.start, 4)), tuple(np.round(t.goal, 4)))
        if key in seen:
            continue
        seen.add(key)
        group.append(t)
        if len(group) >= n:
            break
    if len(group) < n:
        raise SystemExit(f"地图 {map_id} 任务不足 {n} 个")
    sc = maps[map_id]
    env = build_env_from_scene(sc, tensor_args=tensor_args)
    pt = PlanningTask(parametric_trajectory=base_task.parametric_trajectory, env=env,
                      robot=base_task.robot, tensor_args=tensor_args,
                      min_distance_robot_env=0.01, obstacle_cutoff_margin=0.01,
                      margin_for_dense_collision_checking=0.0)
    planner = GenerativeOptimizationPlanner(pt, train_subset.dataset, args_train,
                                            args_inference, tensor_args,
                                            sampling_based_planner_fn=None, debug=False)
    sampler = MultiRobotSampler(planner, n, tensor_args)
    print(f"    map={map_id}  horizon_cp={sampler.horizon}  官方 SDF guide={'有' if planner.cost_guide else '无'}")

    starts = torch.tensor(np.stack([t.start for t in group]), dtype=torch.float32, device=device)
    goals = torch.tensor(np.stack([t.goal for t in group]), dtype=torch.float32, device=device)
    rects = np.asarray(sc["rects"], dtype=np.float64).reshape(-1, 4)
    circ = np.asarray(sc["circles"], dtype=np.float64).reshape(-1, 3)

    def evaluate(paths_m, tag):
        """paths_m: [B,N,H,2]（米制）"""
        paths_np = to_numpy(paths_m)
        b = paths_np.shape[0]
        rows = []
        for bi in range(b):
            p = torch.tensor(paths_np[bi], dtype=torch.float64)
            ok_static, min_clr = static_safe(paths_np[bi], circ, rects)
            # 与引导内部一致：基准发车间隔 stagger*i
            w0 = torch.zeros(n, H - 1, dtype=torch.float64)
            w0[:, 0] = torch.arange(n, dtype=torch.float64) * float(args.stagger)
            with torch.no_grad():
                hr = risk(p, w0,
                          torch.full((n,), args.pause_p, dtype=torch.float64), T,
                          collision_mode="hard", gate_mode="hard", return_diagnostics=False)
                h0 = risk(p, w0,
                          torch.zeros(n, dtype=torch.float64), T,
                          collision_mode="hard", gate_mode="hard", return_diagnostics=False)
            seg = torch.linalg.vector_norm(p[:, 1:] - p[:, :-1], dim=-1)
            rows.append({"tag": tag, "batch": bi,
                         "hard_risk_p": float(hr["pair_sum_risk"]),
                         "hard_risk_p0": float(h0["pair_sum_risk"]),
                         "static_safe": ok_static, "min_clearance": min_clr,
                         "max_seg": float(seg.max()),
                         "length_mean": float(seg.sum(dim=-1).mean())})
        return rows

    print("[2] 采样：无 First-Hit 引导（仅官方 SDF）")
    t0 = time.perf_counter()
    cp0, chain0, _ = sampler.sample(starts, goals, args.samples)
    t_plain = time.perf_counter() - t0
    geo0 = sampler.control_points_to_paths(cp0, starts, goals, args.samples, h=H)
    rows = evaluate(geo0["paths_m"].reshape(args.samples, n, H, 2), "A_sdf_only")
    pm = to_numpy(geo0["paths_m"].reshape(args.samples, n, H, 2))
    print(f"    路径范围: x[{pm[...,0].min():.2f},{pm[...,0].max():.2f}] "
          f"y[{pm[...,1].min():.2f},{pm[...,1].max():.2f}]")
    for bi in range(args.samples):
        dmat = np.linalg.norm(pm[bi][0][:, None, :] - pm[bi][1][None, :, :], axis=-1)
        print(f"    b{bi} 两机同刻最小间距(未加等待)={np.diag(dmat).min():.3f} m  "
              f"任意时刻最小={dmat.min():.3f} m")
    for r in rows:
        print(f"    b{r['batch']}: hard(p={args.pause_p})={r['hard_risk_p']:.3f} "
              f"hard(p=0)={r['hard_risk_p0']:.3f} static={r['static_safe']} "
              f"clr={r['min_clearance']:.3f} maxseg={r['max_seg']:.3f}")

    print(f"[3] 采样：官方 SDF + First-Hit 几何引导（λ_R={args.lam_R}）")
    guide = FirstHitGeometryGuide(risk, sampler, risk_cfg, n_agents=n, lam_R=args.lam_R,
                                  pause_p=args.pause_p, time_steps=T, h_waypoints=H,
                                  stagger_slots=args.stagger, lam_D=args.lam_D)
    combo = SumGuide([planner.cost_guide, guide], weights=[1.0, 1.0], names=["sdf", "firsthit"])
    t0 = time.perf_counter()
    cp1, chain1, _ = sampler.sample(starts, goals, args.samples, guide=combo)
    t_guided = time.perf_counter() - t0
    geo1 = sampler.control_points_to_paths(cp1, starts, goals, args.samples, h=H)
    rows1 = evaluate(geo1["paths_m"].reshape(args.samples, n, H, 2), "B_sdf_firsthit")
    for r in rows1:
        print(f"    b{r['batch']}: hard(p={args.pause_p})={r['hard_risk_p']:.3f} "
              f"hard(p=0)={r['hard_risk_p0']:.3f} static={r['static_safe']} "
              f"clr={r['min_clearance']:.3f} maxseg={r['max_seg']:.3f}")

    # 几何位移
    disp = to_numpy(torch.linalg.vector_norm(geo1["paths_m"] - geo0["paths_m"], dim=-1))
    print(f"[4] 引导引起的路点位移：max={disp.max():.4f} m  mean={disp.mean():.4f} m")
    print(f"    Guide 调用次数={guide.n_calls}  末次诊断={guide.last}")

    (out_dir / "smoke.json").write_text(json.dumps({
        "n_agents": n, "n_samples": args.samples, "lam_R": args.lam_R,
        "pause_p": args.pause_p, "map_id": map_id,
        "rows_sdf_only": rows, "rows_sdf_firsthit": rows1,
        "path_disp_max": float(disp.max()), "path_disp_mean": float(disp.mean()),
        "guide_trace": guide.trace, "t_plain_s": t_plain, "t_guided_s": t_guided,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[5] 输出：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
