"""Small closed-loop pilot on credible_v1 scenes (Ours/ORCA/MPC-CBF).

Writes one JSONL per run and NPZ trajectories, evaluated by the independent
common_evaluator. Static obstacles are evaluated from actual trajectories.
"""

from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fhd.adapters.common_evaluator import evaluate_run  # noqa: E402
from fhd.adapters.orca_adapter import run_orca  # noqa: E402
from fhd.adapters.mpc_cbf_adapter import run_mpc_cbf  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, make_pause_events, simulate_mpc  # noqa: E402
from fhd.optim.online_firsthit import OnlineFirstHitCoordinator  # noqa: E402
from fhd.optim.wait_policies import WaitBudget, allowed_positions, make_mask, policy_w2plus  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402


def w2plus(paths, risk, T, H, n, seed):
    w0 = torch.zeros((n, H - 1), dtype=torch.float64)
    return policy_w2plus(w0, make_mask(H, allowed_positions(H, 5), n),
                         WaitBudget(time_steps=T, horizon=H), risk,
                         torch.tensor(paths), torch.full((n,), 0.1), T,
                         lam_T=0.05, max_evals=800, restarts=3, seed=seed,
                         require_p0=False, p0_weight=1.0,
                         initial_stagger=True).wait.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "results/Continuous_Baseline_Comparison/credible_v1/task_manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "results/Continuous_Baseline_Comparison/credible_v1/small_pilot"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    base = load_config(ROOT / "configs/base.yaml")
    T = int(base["execution"]["horizon_steps"]); H = int(base["trajectory"]["horizon"])
    risk = FirstHitRisk.from_config(base)
    jsonl = out / "per_run.jsonl"
    rows = []
    for ti, task in enumerate([t for t in m["tasks"] if t["scene"] != "narrow"]):
        paths = np.asarray(task["paths"], float)
        n = len(paths)
        w2 = w2plus(paths, risk, T, H, n, ti)
        for p in (0.0, 0.1):
            for k in range(1 if p == 0 else 2):
                seed = 20281701 + ti * 100 + k
                for method in ("ours_frozen", "orca_rvo2", "mpc_cbf"):
                    cfg = MPCSimConfig(d_safe=0.35, horizon_steps=T)
                    events = make_pause_events(seed, p, n, T, cfg)
                    t0 = time.perf_counter()
                    try:
                        if method == "ours_frozen":
                            co = OnlineFirstHitCoordinator(paths, w2, 0.35, 0.4, 3.2, p,
                                                           risk_model=risk,
                                                           config={"hold_slots": 1,
                                                                   "lookahead_m": 2.0})
                            r = simulate_mpc(paths, w2, p, cfg, seed=seed, record=True,
                                             events=events, coordinator=co)
                            pos = np.asarray(r.trajectories)
                            times = np.arange(len(pos)) * 0.1
                            vel = np.gradient(pos, 0.1, axis=0)
                            info = {"compute_ms": float(r.coordinator_stats.get("p95_ms", 0.0)),
                                    "solver_fail": 0}
                        elif method == "orca_rvo2":
                            r = run_orca(paths, events, steps=int(T * 0.4 / 0.1))
                            times = np.arange(len(r["traj"])) * 0.1
                            pos, vel = r["traj"], r["vel"]
                            info = {"compute_ms": 0.0, "solver_fail": 0}
                        else:
                            r = run_mpc_cbf(paths, events, steps=int(T * 0.4 / 0.1))
                            times = np.arange(len(r["traj"])) * 0.1
                            pos, vel = r["traj"], r["vel"]
                            info = {"compute_ms": float(r["solve_p95_ms"]),
                                    "solver_fail": int(r["fails"])}
                    except Exception as exc:
                        rec = {"method": method, "task_id": task["task_id"],
                               "scene": task["scene"], "p": p, "seed": seed,
                               "error": f"{type(exc).__name__}: {exc}",
                               "safe_completion": False, "collision": None}
                        rows.append(rec)
                        with jsonl.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue
                    ev = evaluate_run(times, pos, vel, paths,
                                      rects=task["rects"], circles=task["circles"],
                                      d_safe=0.35, robot_radius=0.15, clearance=0.03)
                    rec = {"method": method, "task_id": task["task_id"],
                           "scene": task["scene"], "p": p, "seed": seed,
                           "collision": ev["inter_robot_violation"],
                           "static_collision": ev["static_obstacle_violation"],
                           "static_status": ev["static_status"],
                           "safe_completion": ev["safe_completion"],
                           "all_arrived": ev["all_arrived"],
                           "completion_s": float(times[-1]),
                           "min_pair": ev["min_pair_distance"],
                           "min_static": ev["min_static_clearance"],
                           "path_deviation": 0.0,
                           "compute_ms": info["compute_ms"],
                           "solver_fail": info["solver_fail"],
                           "error": None}
                    rows.append(rec)
                    with jsonl.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    np.savez_compressed(out / "raw" / f"{method}_{task['task_id']}_p{p}_s{seed}.npz",
                                        times=times, positions=pos, velocities=vel,
                                        paths=paths)
                    print(method, task["task_id"], p, seed, ev["safe_completion"], flush=True)
    with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {"stage": "credible_v1_small_pilot", "n_tasks": len(m["tasks"]),
               "by": {}}
    for method in ("ours_frozen", "orca_rvo2", "mpc_cbf"):
        for scene in ("open", "static", "narrow"):
            sel = [r for r in rows if r["method"] == method and r["scene"] == scene
                   and r.get("error") is None]
            summary["by"][f"{method}|{scene}"] = {
                "n": len(sel),
                "safe_rate": float(np.mean([r["safe_completion"] for r in sel])) if sel else None,
                "inter_collision": float(np.mean([r["collision"] for r in sel])) if sel else None,
                "static_collision": float(np.mean([bool(r["static_collision"]) for r in sel if r["static_collision"] is not None])) if any(r["static_collision"] is not None for r in sel) else None,
                "completion_s": float(np.mean([r["completion_s"] for r in sel])) if sel else None,
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
