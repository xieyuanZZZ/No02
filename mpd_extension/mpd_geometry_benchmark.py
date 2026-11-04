"""几何引导正式基准：G0/G2/G3/G5 在 N=4 冻结任务上的统一对照。

关键设计（保证公平）：
  * **G5 与 G3 使用完全相同的候选选择程序**（beam search + p=0 硬约束），
    唯一区别是候选池是否在 DDIM 采样时接了 First-Hit 几何引导。
    这样 G3−G5 才真正回答"去噪过程内引导是否优于纯候选筛选"。
  * 所有方法共用同一套最终安全审计（静态连续、速度、p=0、MC）。
  * 规划覆盖率与延迟鲁棒性分开统计。

方法：
  G0  冻结 MPD 轨迹 + 原始等待（参照）
  G2  SDF-only 候选池 B=1，不筛选（展示原始采样覆盖率）
  G3  First-Hit 引导候选池 B=8 + 筛选
  G5  SDF-only 候选池 B=8 + 同一筛选
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotmap import DotMap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/d/paper2/src")
sys.path.insert(0, "/mnt/d/paper2/scripts")
sys.path.insert(0, str(ROOT / "mpd_extension"))

from mpd.inference.inference import GenerativeOptimizationPlanner  # noqa: E402
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml  # noqa: E402
from torch_robotics.tasks.tasks import PlanningTask  # noqa: E402
from torch_robotics.torch_utils.seed import fix_random_seed  # noqa: E402

from first_hit_geometry_guide import FirstHitGeometryGuide, SumGuide  # noqa: E402
from multi_robot_mpd_sampler import MultiRobotSampler  # noqa: E402
from simple_scenes_adapter import build_env_from_scene  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", default="/mnt/d/paper2/data/mpd_delay_plan_bank")
    ap.add_argument("--out", default=str(ROOT / "results_mpd" / "geometry_benchmark"))
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--lam-D", type=float, default=200.0)
    ap.add_argument("--lam-R", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--chunk", type=int, default=4,
                    help="每批候选数；官方 conv1d 在大批大小下会触发 cuDNN 错误")
    ap.add_argument("--n-guide-steps", type=int, default=10)
    ap.add_argument("--stagger", type=int, default=3)
    ap.add_argument("--mc-seeds", default="20261201,20261202,20261203")
    ap.add_argument("--mc-samples", type=int, default=200)
    ap.add_argument("--ps", default="0,0.05,0.1,0.2")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    fix_random_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_args = {"device": device, "dtype": torch.float32}
    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    from fhd.execution.simulator import simulate_execution
    from fhd.risk.multi_pair import FirstHitRisk
    from fhd.utils.config import load_config
    risk_cfg = load_config("/mnt/d/paper2/configs/base.yaml")
    risk = FirstHitRisk.from_config(risk_cfg)
    d_safe = risk.d_safe
    T = int(risk_cfg["execution"]["horizon_steps"])
    H = int(risk_cfg["trajectory"]["horizon"])
    dt_s = float(risk_cfg["trajectory"]["dt_s"])

    bank = json.loads((Path(args.banks) / "plan_bank_n4.json").read_text(encoding="utf-8"))
    trajs = np.load(Path(args.banks) / "plan_bank_n4_trajs.npz")
    tasks_bank = bank["tasks"][: args.n_tasks]
    maps_all = {m["map_id"] if "map_id" in m else f"{m['kind']}_{m['seed']}": m
                for m in json.loads(Path("/mnt/d/paper2/data/revised_s1/maps_test.json")
                                    .read_text(encoding="utf-8"))}

    cfg_path = ROOT / "scripts" / "inference" / "cfgs" / "config_EnvSimple2D-RobotPointMass2D_00.yaml"
    args_inference = DotMap(load_params_from_yaml(str(cfg_path)))
    args_inference.model_dir = os.path.expandvars(args_inference.model_dir_ddpm_bspline)
    args_inference.planner_alg = "mpd"
    args_inference.ddim_sampling_timesteps = args.steps
    args_inference.n_trajectory_samples = args.candidates
    args_inference.n_guide_steps = args.n_guide_steps
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(**args_inference, gripper=True, reload_data=False,
                      results_dir=str(out_dir), load_indices=True, tensor_args=tensor_args)
    base_task, train_subset, _, _, _ = get_planning_task_and_dataset(**args_train)

    def make_planner(map_id):
        sc = maps_all[map_id]
        env = build_env_from_scene(sc, tensor_args=tensor_args)
        pt = PlanningTask(parametric_trajectory=base_task.parametric_trajectory, env=env,
                          robot=base_task.robot, tensor_args=tensor_args,
                          min_distance_robot_env=0.01, obstacle_cutoff_margin=0.01,
                          margin_for_dense_collision_checking=0.0)
        return GenerativeOptimizationPlanner(pt, train_subset.dataset, args_train,
                                             args_inference, tensor_args,
                                             sampling_based_planner_fn=None, debug=False), sc

    def static_clearance(paths_m, sc):
        pts = paths_m.reshape(-1, 2)
        d = np.full(len(pts), np.inf)
        circ = np.asarray(sc.get("circles", []), dtype=np.float64).reshape(-1, 3)
        rects = np.asarray(sc.get("rects", []), dtype=np.float64).reshape(-1, 4)
        if len(circ):
            dc = np.linalg.norm(pts[:, None, :] - circ[None, :, :2], axis=-1) - circ[None, :, 2]
            d = np.minimum(d, dc.min(axis=1))
        if len(rects):
            lo, hi = rects[None, :, :2], rects[None, :, 2:]
            p = pts[:, None, :]
            inside = np.all((p >= lo) & (p <= hi), axis=-1)
            dx = np.maximum(np.maximum(lo[..., 0] - p[..., 0], p[..., 0] - hi[..., 0]), 0.0)
            dy = np.maximum(np.maximum(lo[..., 1] - p[..., 1], p[..., 1] - hi[..., 1]), 0.0)
            db = np.where(inside, -1.0, np.sqrt(dx ** 2 + dy ** 2))
            d = np.minimum(d, db.min(axis=1))
        return float(d.min())

    def joint_risk(paths_np, n, p=0.1):
        pa = torch.tensor(paths_np, dtype=torch.float64)
        w = torch.zeros(n, paths_np.shape[1] - 1, dtype=torch.float64)
        w[:, 0] = torch.arange(n, dtype=torch.float64) * args.stagger
        with torch.no_grad():
            hr = risk(pa, w, torch.full((n,), p, dtype=torch.float64), T,
                      collision_mode="hard", gate_mode="hard", return_diagnostics=False)
            h0 = risk(pa, w, torch.zeros(n, dtype=torch.float64), T,
                      collision_mode="hard", gate_mode="hard", return_diagnostics=False)
        return float(hr["pair_sum_risk"]), float(h0["pair_sum_risk"])

    def select_combo(paths_all, n):
        beam = [(0.0, 0.0, [])]
        for _ in range(n):
            cand = []
            for p0v, sc_, chosen in beam:
                for c in range(paths_all.shape[0]):
                    trial = chosen + [c]
                    if len(trial) < 2:
                        cand.append((0.0, 0.0, trial))
                        continue
                    pa = np.stack([paths_all[cc, k] for k, cc in enumerate(trial)])
                    hr, h0 = joint_risk(pa, len(trial))
                    cand.append((max(p0v, 1.0 if h0 > 1e-9 else 0.0), sc_ + hr, trial))
            cand.sort(key=lambda x: (x[0], x[1]))
            beam = cand[: args.beam]
        return beam[0]

    def eval_plan(paths_np, w0_np, tag):
        n = paths_np.shape[0]
        p_t = torch.tensor(paths_np, dtype=torch.float64)
        w0 = torch.tensor(w0_np, dtype=torch.float64)
        min_clr = static_clearance(paths_np, sc_map)
        seg = np.linalg.norm(np.diff(paths_np, axis=1), axis=-1)
        row = {"min_static_clearance": min_clr,
               "static_ok": 1.0 if min_clr > 0.0 else 0.0,
               "max_seg": float(seg.max()),
               "speed_ok": 1.0 if seg.max() <= 1.5 * dt_s + 1e-6 else 0.0,
               "length_mean": float(seg.sum(axis=-1).mean())}
        with torch.no_grad():
            h0 = risk(p_t, w0, torch.zeros(n, dtype=torch.float64), T,
                      collision_mode="hard", gate_mode="hard", return_diagnostics=False)
        row["p0_hard_risk"] = float(h0["pair_sum_risk"])
        row["p0_safe"] = 1.0 if row["p0_hard_risk"] <= 1e-9 else 0.0
        row["valid"] = float(row["static_ok"] * row["speed_ok"] * row["p0_safe"])
        for p in [float(x) for x in args.ps.split(",")]:
            cr, sr, mk = [], [], []
            for s in [int(x) for x in args.mc_seeds.split(",")]:
                gen = torch.Generator().manual_seed(s * 7919 + (abs(hash(tag)) % 997))
                pause = torch.full((n,), p, dtype=torch.float64)
                sim = simulate_execution(p_t, w0, pause, d_safe, T, dt_s,
                                         n_samples=args.mc_samples, generator=gen)
                coll = sim.collision.to(torch.float64)
                arr = sim.all_arrived.to(torch.float64)
                cr.append(float(coll.mean()))
                sr.append(float(((1 - coll) * arr).mean()))
                mk.append(float(sim.makespan_steps.to(torch.float64).mean()))
            row[f"collision_p{p}"] = float(np.mean(cr))
            row[f"safe_p{p}"] = float(np.mean(sr))
            row[f"makespan_p{p}"] = float(np.mean(mk))
        return row

    rows, traces = [], []
    t_start = time.perf_counter()
    for ti, jt in enumerate(tasks_bank):
        map_id = jt["map_id"]
        planner, sc_map = make_planner(map_id)
        sampler = MultiRobotSampler(planner, 4, tensor_args)
        starts = torch.tensor(np.asarray(jt["starts_m"], dtype=np.float32) / 5.0 - 1.0, device=device)
        goals = torch.tensor(np.asarray(jt["goals_m"], dtype=np.float32) / 5.0 - 1.0, device=device)
        w0_bank = np.asarray(jt["baseline_wait"], dtype=np.float64)
        paths_bank = np.asarray(trajs[jt["task_id"]], dtype=np.float64)

        rec = {"task_id": jt["task_id"], "map_id": map_id, "kind": jt["kind"]}
        rec.update({f"G0_{k}": v for k, v in
                    eval_plan(paths_bank, w0_bank, f"G0_{jt['task_id']}").items()})

        torch.manual_seed(args.seed * 100 + ti)
        t0 = time.perf_counter()
        cp1, _, _ = sampler.sample(starts, goals, 1)
        geo1 = sampler.control_points_to_paths(cp1, starts, goals, 1, h=H)
        rec["t_g2_s"] = time.perf_counter() - t0
        rec.update({f"G2_{k}": v for k, v in eval_plan(
            geo1["paths_m"].reshape(4, H, 2).detach().cpu().numpy().astype(np.float64),
            w0_bank, f"G2_{jt['task_id']}").items()})

        torch.manual_seed(args.seed * 100 + ti + 500)
        t0 = time.perf_counter()
        cpA, _, _ = sampler.sample(starts, goals, args.candidates, chunk=args.chunk)
        geoA = sampler.control_points_to_paths(cpA, starts, goals, args.candidates, h=H)
        rec["t_pool_A_s"] = time.perf_counter() - t0
        pathsA = geoA["paths_m"].reshape(args.candidates, 4, H, 2).detach().cpu().numpy().astype(np.float64)

        guide = FirstHitGeometryGuide(risk, sampler, risk_cfg, n_agents=4,
                                      lam_R=args.lam_R, lam_D=args.lam_D,
                                      pause_p=0.1, time_steps=T, h_waypoints=H,
                                      stagger_slots=args.stagger)
        combo = SumGuide([planner.cost_guide, guide], weights=[1.0, 1.0],
                         names=["sdf", "firsthit"])
        torch.manual_seed(args.seed * 100 + ti + 500)
        t0 = time.perf_counter()
        cpB, _, _ = sampler.sample(starts, goals, args.candidates, guide=combo,
                                   chunk=args.chunk)
        geoB = sampler.control_points_to_paths(cpB, starts, goals, args.candidates, h=H)
        rec["t_pool_B_s"] = time.perf_counter() - t0
        pathsB = geoB["paths_m"].reshape(args.candidates, 4, H, 2).detach().cpu().numpy().astype(np.float64)

        for tag, pool in (("G5", pathsA), ("G3", pathsB)):
            p0v, score, choice = select_combo(pool, 4)
            rec[f"{tag}_p0_violation"] = p0v
            rec[f"{tag}_best_risk"] = score
            rec[f"{tag}_choice"] = str(choice)
            pm = np.stack([pool[c, k] for k, c in enumerate(choice)])
            rec.update({f"{tag}_{k}": v for k, v in
                        eval_plan(pm, w0_bank, f"{tag}_{jt['task_id']}").items()})

        rec["guide_calls"] = guide.n_calls
        rec["guide_grad_norm_last"] = float(guide.last.get("grad_norm", 0.0))
        rec["guide_hinge_last"] = float(guide.last.get("hinge_gap", 0.0))
        if guide.trace:
            traces.append({"task_id": jt["task_id"], "trace": guide.trace})
        rec["pool_disp_mean"] = float(np.abs(pathsB - pathsA).mean())
        rec["pool_disp_max"] = float(np.abs(pathsB - pathsA).max())

        rows.append(rec)
        print(f"[{ti+1}/{len(tasks_bank)}] {jt['task_id']}: "
              f"G0={rec['G0_collision_p0.1']:.3f} G2={rec['G2_collision_p0.1']:.3f} "
              f"G3={rec['G3_collision_p0.1']:.3f} G5={rec['G5_collision_p0.1']:.3f} "
              f"| valid G3={int(rec['G3_valid'])} G5={int(rec['G5_valid'])} "
              f"| disp={rec['pool_disp_mean']:.3f}m", flush=True)

    total = time.perf_counter() - t_start
    with (out_dir / "benchmark_tasks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "guide_traces.json").write_text(json.dumps(traces, indent=1), encoding="utf-8")

    summary = {"n_tasks": len(rows), "config": vars(args), "total_time_s": total, "means": {}}
    for tag in ("G0", "G2", "G3", "G5"):
        d = {}
        for k in rows[0]:
            if k.startswith(tag + "_") and isinstance(rows[0][k], (int, float, bool)):
                d[k[len(tag) + 1:]] = float(np.mean([float(r[k]) for r in rows]))
        summary["means"][tag] = d
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()
                          if kk.startswith(("collision_p0.1", "collision_p0.2", "safe_p0.1",
                                            "makespan_p0.1", "valid", "static_ok", "p0_safe"))}
                      for k, v in summary["means"].items()}, ensure_ascii=False, indent=2))
    print(f"输出：{out_dir}  总耗时 {total/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
