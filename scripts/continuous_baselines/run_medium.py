"""Resumable medium comparison: Ours, ORCA, MPC-CBF on 4 N4 tasks.

p=0 uses one deterministic seed; p>0 uses three paired seeds.
Writes one JSONL record per (method, task, p, seed), skipping completed ones.
MMD is reported separately (native scene) because common-scene MMD is blocked.
"""

from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fhd.adapters.orca_adapter import run_orca, path_deviation  # noqa: E402
from fhd.adapters.mpc_cbf_adapter import run_mpc_cbf  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, make_pause_events, simulate_mpc  # noqa: E402
from fhd.optim.online_firsthit import OnlineFirstHitCoordinator  # noqa: E402
from fhd.optim.wait_policies import WaitBudget, allowed_positions, make_mask, policy_w2plus  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402


def w2plus(paths, w0, risk, T, H, n, seed):
    return policy_w2plus(torch.tensor(w0), make_mask(H, allowed_positions(H, 5), n),
                         WaitBudget(time_steps=T, horizon=H), risk,
                         torch.tensor(paths), torch.full((n,), 0.1), T,
                         lam_T=0.05, max_evals=300, restarts=2, seed=seed).wait.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/Continuous_Baseline_Comparison/medium_v1"))
    ap.add_argument("--n-tasks", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"; raw.mkdir(exist_ok=True)
    jsonl = out / "per_run.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["method"], r["task_id"], r["p"], r["seed"]))
            except Exception:
                pass
    base = load_config(ROOT / "configs/base.yaml")
    T = int(base["execution"]["horizon_steps"]); H = int(base["trajectory"]["horizon"])
    risk = FirstHitRisk.from_config(base)
    bank = json.loads((ROOT / "data/mpd_delay_plan_bank/plan_bank_n4.json").read_text(encoding="utf-8"))
    trajs = np.load(ROOT / "data/mpd_delay_plan_bank/plan_bank_n4_trajs.npz")
    tasks = bank["tasks"][:args.n_tasks]
    for ti, jt in enumerate(tasks):
        tid = jt["task_id"]; paths = np.asarray(trajs[tid], float)
        w0 = np.asarray(jt["baseline_wait"], float)
        w2 = w2plus(paths, w0, risk, T, H, 4, ti)
        for p in (0.0, 0.1, 0.2):
            nseed = 1 if p == 0.0 else 3
            for k in range(nseed):
                seed = 20281601 + ti * 1000 + k
                for method in ("ours_frozen", "orca_rvo2", "mpc_cbf"):
                    key = (method, tid, p, seed)
                    if key in done:
                        continue
                    cfg = MPCSimConfig(d_safe=0.35, horizon_steps=T)
                    events = make_pause_events(seed, p, 4, T, cfg)
                    t0 = time.perf_counter()
                    try:
                        if method == "ours_frozen":
                            co = OnlineFirstHitCoordinator(paths, w2, 0.35, 0.4, 3.2, p,
                                                           risk_model=risk,
                                                           config={"hold_slots": 1})
                            r = simulate_mpc(paths, w2, p, cfg, seed=seed, record=False,
                                             events=events, coordinator=co)
                            row = dict(collision=bool(r.collision),
                                       safe_completion=bool(r.safe_completion),
                                       all_arrived=bool(r.all_arrived), timeout=bool(r.timeout),
                                       completion_s=float(r.completion_time),
                                       planning_time_s=0.0, online_ms=float(r.coordinator_stats.get("p95_ms", 0)),
                                       path_deviation=0.0, min_pair=float(np.nanmin(r.pair_min_distance)),
                                       solver_fail=0, static_status="not_evaluated")
                        elif method == "orca_rvo2":
                            r = run_orca(paths, events, steps=int(T * 0.4 / 0.1))
                            row = dict(collision=bool(r["collision"]),
                                       safe_completion=bool(r["all_arrived"] and not r["collision"]),
                                       all_arrived=bool(r["all_arrived"]), timeout=bool(r["timeout"]),
                                       completion_s=float(r["completion_s"]), planning_time_s=0.0,
                                       online_ms=0.0, path_deviation=path_deviation(r["traj"], paths),
                                       min_pair=float(np.min(r["pair_min"])), solver_fail=0,
                                       static_status="not_evaluated")
                        else:
                            r = run_mpc_cbf(paths, events, steps=int(T * 0.4 / 0.1))
                            row = dict(collision=bool(r["collision"]),
                                       safe_completion=bool(r["all_arrived"] and not r["collision"]),
                                       all_arrived=bool(r["all_arrived"]), timeout=bool(r["timeout"]),
                                       completion_s=float(r["completion_s"]), planning_time_s=0.0,
                                       online_ms=float(r["solve_p95_ms"]), path_deviation=0.0,
                                       min_pair=float(np.min(r["pair_min"])), solver_fail=int(r["fails"]),
                                       static_status="not_evaluated")
                    except Exception as exc:
                        row = dict(collision=None, safe_completion=False, all_arrived=False,
                                   timeout=True, completion_s=float(T * 0.4),
                                   planning_time_s=0.0, online_ms=0.0,
                                   path_deviation=None, min_pair=None, solver_fail=1,
                                   static_status="not_evaluated",
                                   error=f"{type(exc).__name__}: {exc}")
                    rec = {"method": method, "task_id": tid, "p": p, "seed": seed,
                           "wall_s": time.perf_counter() - t0, **row}
                    with jsonl.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(method, tid, p, seed, rec.get("collision"), rec.get("all_arrived"),
                          flush=True)
    rows = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    # write CSV for convenience
    fields = sorted({k for r in rows for k in r})
    with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {"stage": "medium_v1", "n_tasks": len(tasks), "by": {}}
    for method in ("ours_frozen", "orca_rvo2", "mpc_cbf"):
        for p in (0.0, 0.1, 0.2):
            sel = [r for r in rows if r["method"] == method and r["p"] == p]
            summary["by"][f"{method}|{p}"] = {
                "n": len(sel),
                "collision_rate": float(np.mean([bool(r["collision"]) for r in sel])) if sel else None,
                "safe_rate": float(np.mean([bool(r["safe_completion"]) for r in sel])) if sel else None,
                "completion_s": float(np.mean([r["completion_s"] for r in sel])) if sel else None,
                "online_ms": float(np.mean([r["online_ms"] for r in sel])) if sel else None,
                "path_deviation": float(np.mean([r["path_deviation"] for r in sel if r["path_deviation"] is not None])) if sel else None,
                "solver_fail": int(sum(r["solver_fail"] for r in sel)),
            }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
