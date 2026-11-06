"""ORCA RVO2 vs frozen Ours on N4 tasks, same paths and pause events."""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fhd.adapters.orca_adapter import run_orca, path_deviation  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, make_pause_events, simulate_mpc  # noqa: E402
from fhd.optim.online_firsthit import OnlineFirstHitCoordinator  # noqa: E402
from fhd.optim.wait_policies import WaitBudget, allowed_positions, make_mask, policy_w2plus  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
import csv


def w2plus(paths, w0, risk, T, H, n, seed):
    budget = WaitBudget(time_steps=T, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), n)
    return policy_w2plus(torch.tensor(w0), mask, budget, risk,
                         torch.tensor(paths), torch.full((n,), 0.1), T,
                         lam_T=0.05, max_evals=300, restarts=2, seed=seed).wait.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\paper2\_orca_pilot")
    ap.add_argument("--n-tasks", type=int, default=4)
    ap.add_argument("--n-mc", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / "configs/base.yaml")
    T = int(base["execution"]["horizon_steps"]); H = int(base["trajectory"]["horizon"])
    risk = FirstHitRisk.from_config(base)
    bank = json.loads((ROOT / "data/mpd_delay_plan_bank/plan_bank_n4.json").read_text(encoding="utf-8"))
    trajs = np.load(ROOT / "data/mpd_delay_plan_bank/plan_bank_n4_trajs.npz")
    tasks = bank["tasks"][:args.n_tasks]
    rows = []
    for ti, jt in enumerate(tasks):
        tid = jt["task_id"]; paths = np.asarray(trajs[tid], float)
        w0 = np.asarray(jt["baseline_wait"], float); w2 = w2plus(paths, w0, risk, T, H, 4, ti)
        for p in (0.0, 0.1, 0.2):
            for k in range(args.n_mc):
                seed = 20281301 + ti * 1000 + k
                events = make_pause_events(seed, p, 4, T, MPCSimConfig(d_safe=0.35, horizon_steps=T))
                r = run_orca(paths, events, steps=int(T * 0.4 / 0.1))
                rows.append({"method": "orca_rvo2", "task_id": tid, "p": p, "seed": seed,
                             "collision": r["collision"], "safe_completion": bool(r["all_arrived"] and not r["collision"]),
                             "timeout": r["timeout"], "all_arrived": r["all_arrived"],
                             "completion_s": r["completion_s"], "online_requests": r["online_requests"],
                             "online_wait_s": float(r["paused"].sum() * 0.1),
                             "max_speed": r["max_speed"], "path_deviation": path_deviation(r["traj"], paths),
                             "coord_p95_ms": 0.0, "coord_errors": 0, "mpc_failures": 0,
                             "infeasible_stops": 0})
                co = OnlineFirstHitCoordinator(paths, w2, 0.35, 0.4, 3.2, p, risk_model=risk,
                                               config={"hold_slots": 1})
                rr = simulate_mpc(paths, w2, p, MPCSimConfig(d_safe=0.35, horizon_steps=T),
                                  seed=seed, record=False, events=events, coordinator=co)
                rows.append({"method": "ours_frozen", "task_id": tid, "p": p, "seed": seed,
                             "collision": bool(rr.collision), "safe_completion": bool(rr.safe_completion),
                             "timeout": bool(rr.timeout), "all_arrived": bool(rr.all_arrived),
                             "completion_s": float(rr.completion_time),
                             "online_requests": int(rr.online_requests),
                             "online_wait_s": float(rr.online_wait_time),
                             "max_speed": float(rr.max_speed),
                             "path_deviation": 0.0,
                             "coord_p95_ms": float(rr.coordinator_stats.get("p95_ms", 0.0)),
                             "coord_errors": int(rr.coordinator_stats.get("errors", 0)),
                             "mpc_failures": int(rr.mpc_failures),
                             "infeasible_stops": int(rr.infeasible_stops)})
        print("done", tid, flush=True)
    with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {"stage": "orca_closed_loop_pilot", "n_tasks": args.n_tasks, "n_mc": args.n_mc,
               "by": {}}
    for m in ("ours_frozen", "orca_rvo2"):
        for p in (0.0, 0.1, 0.2):
            sel = [r for r in rows if r["method"] == m and r["p"] == p]
            summary["by"][f"{m}|{p}"] = {
                "n": len(sel), "collision_rate": float(np.mean([x["collision"] for x in sel])),
                "safe_rate": float(np.mean([x["safe_completion"] for x in sel])),
                "completion_s": float(np.mean([x["completion_s"] for x in sel])),
                "online_wait_s": float(np.mean([x["online_wait_s"] for x in sel])),
                "path_deviation": float(np.mean([x["path_deviation"] for x in sel])),
                "p95_ms": float(np.mean([x["coord_p95_ms"] for x in sel])),
            }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
