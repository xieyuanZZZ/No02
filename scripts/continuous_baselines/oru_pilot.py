"""Closed-loop ORU vs frozen Ours pilot (N4), same paths/events/MPC.

Run:
  python scripts/continuous_baselines/oru_pilot.py --n-tasks 4 --n-mc 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fhd.adapters.coordination_oru_adapter import OruCoordinator  # noqa: E402
from fhd.execution.mpc_simulator import (MPCSimConfig, make_pause_events,  # noqa: E402
                                         simulate_mpc)
from fhd.optim.online_firsthit import OnlineFirstHitCoordinator  # noqa: E402
from fhd.optim.wait_policies import (WaitBudget, allowed_positions, make_mask,  # noqa: E402
                                     policy_w2plus)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402


def _w2plus(paths, w0, risk, T, H, n, seed=0):
    budget = WaitBudget(time_steps=T, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), n)
    pt = torch.tensor(paths, dtype=torch.float64)
    wt = torch.tensor(w0, dtype=torch.float64)
    p01 = torch.full((n,), 0.1, dtype=torch.float64)
    return policy_w2plus(wt, mask, budget, risk, pt, p01, T, lam_T=0.05,
                         max_evals=300, restarts=2, seed=seed).wait.numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/Continuous_Baseline_Comparison/pilot/oru"))
    ap.add_argument("--n-tasks", type=int, default=4)
    ap.add_argument("--n-mc", type=int, default=5)
    ap.add_argument("--ps", default="0,0.1,0.2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = load_config(ROOT / "configs/base.yaml")
    T = int(base["execution"]["horizon_steps"])
    H = int(base["trajectory"]["horizon"])
    dt_slot = float(base["trajectory"]["dt_s"])
    risk = FirstHitRisk.from_config(base)
    bank = json.loads((ROOT / "data/mpd_delay_plan_bank/plan_bank_n4.json").read_text(encoding="utf-8"))
    trajs = np.load(ROOT / "data/mpd_delay_plan_bank/plan_bank_n4_trajs.npz")
    tasks = bank["tasks"][:args.n_tasks]
    ps = [float(x) for x in args.ps.split(",")]
    rows = []
    t_start = time.perf_counter()
    for ti, jt in enumerate(tasks):
        tid = jt["task_id"]
        paths = np.asarray(trajs[tid], dtype=float)
        w0 = np.asarray(jt["baseline_wait"], dtype=float)
        w2 = _w2plus(paths, w0, risk, T, H, 4, seed=ti)
        for p in ps:
            for k in range(args.n_mc):
                seed = 20281201 + ti * 1000 + k
                cfg = MPCSimConfig(d_safe=0.35, horizon_steps=T)
                events = make_pause_events(seed, p, 4, T, cfg)
                variants = [
                    ("ours_frozen", w2, OnlineFirstHitCoordinator(
                        paths, w2, 0.35, dt_slot, 3.2, p, risk_model=risk,
                        config={"hold_slots": 1, "lookahead_m": 2.0})),
                    ("coordination_oru", w0, OruCoordinator(
                        paths, w0, 0.35, dt_slot, 3.0, p, {"oru_hold_s": 3.0})),
                ]
                for name, wait, co in variants:
                    try:
                        r = simulate_mpc(paths, wait, p, cfg, seed=seed, record=False,
                                         events=events, coordinator=co)
                        err = None
                    except Exception as exc:
                        r = None
                        err = f"{type(exc).__name__}: {exc}"
                    finally:
                        if hasattr(co, "close"):
                            co.close()
                    if r is None:
                        rows.append({"method": name, "task_id": tid, "p": p, "seed": seed,
                                     "error": err})
                        continue
                    rows.append({
                        "method": name, "task_id": tid, "p": p, "seed": seed,
                        "collision": bool(r.collision),
                        "safe_completion": bool(r.safe_completion),
                        "timeout": bool(r.timeout),
                        "all_arrived": bool(r.all_arrived),
                        "completion_s": float(r.completion_time),
                        "total_wait_s": float(r.actual_wait_time + r.random_hold_time
                                               + r.online_wait_time),
                        "online_wait_s": float(r.online_wait_time),
                        "online_requests": int(r.online_requests),
                        "online_accepted": int(r.online_accepted),
                        "online_rejected": int(r.online_rejected),
                        "coord_mean_ms": float(r.coordinator_stats.get("mean_ms", 0.0)),
                        "coord_p95_ms": float(r.coordinator_stats.get("p95_ms", 0.0)),
                        "coord_p99_ms": float(r.coordinator_stats.get("p99_ms", 0.0)),
                        "coord_errors": int(r.coordinator_stats.get("errors", 0)),
                        "mpc_failures": int(r.mpc_failures),
                        "infeasible_stops": int(r.infeasible_stops),
                        "min_pair_distance": float(np.nanmin(r.pair_min_distance)),
                        "error": None,
                    })
        print("done", tid, flush=True)
    fields = list(rows[0].keys())
    with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    summary = {"stage": "oru_closed_loop_pilot", "n_tasks": len(tasks),
               "n_mc": args.n_mc, "ps": ps, "runtime_s": time.perf_counter() - t_start,
               "by_method_p": {}}
    for name in ("ours_frozen", "coordination_oru"):
        for p in ps:
            sel = [r for r in rows if r["method"] == name and abs(r["p"] - p) < 1e-12
                   and r.get("error") is None]
            summary["by_method_p"][f"{name}|{p}"] = {
                "n": len(sel),
                "collision_rate": float(np.mean([r["collision"] for r in sel])) if sel else None,
                "safe_rate": float(np.mean([r["safe_completion"] for r in sel])) if sel else None,
                "completion_s": float(np.mean([r["completion_s"] for r in sel])) if sel else None,
                "online_wait_s": float(np.mean([r["online_wait_s"] for r in sel])) if sel else None,
                "p95_ms": float(np.mean([r["coord_p95_ms"] for r in sel])) if sel else None,
            }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
