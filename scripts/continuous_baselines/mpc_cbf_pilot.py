"""Minimal MPC-CBF closed-loop pilot for N=2 and N=4 (double integrator)."""

from __future__ import annotations
import json, csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fhd.adapters.mpc_cbf_adapter import run_mpc_cbf  # noqa: E402

T = 96
d2 = json.loads((ROOT / "data/mpd_delay_plan_bank/plan_bank.json").read_text(encoding="utf-8"))
t2 = np.load(ROOT / "data/mpd_delay_plan_bank/plan_bank_trajs.npz")
d4 = json.loads((ROOT / "data/mpd_delay_plan_bank/plan_bank_n4.json").read_text(encoding="utf-8"))
t4 = np.load(ROOT / "data/mpd_delay_plan_bank/plan_bank_n4_trajs.npz")
rows = []
for kind, bank, trajs, tid in [
    ("N2", d2, t2, "open_crossing_300000_pair02_11"),
    ("N4", d4, t4, "open_crossing_300000_n4_0"),
]:
    jt = [x for x in bank["tasks"] if x["task_id"] == tid][0]
    paths = np.asarray(trajs[tid], float)
    rng = np.random.default_rng(20281401)
    events = [(slot, int(rng.integers(len(paths))), 0.4) for slot in range(T)
              if rng.random() < 0.1]
    r = run_mpc_cbf(paths, events, steps=int(T * 0.4 / 0.1))
    rows.append({"method": "mpc_cbf", "group": kind, "task_id": tid, "p": 0.1,
                 "seed": 20281401, "collision": r["collision"],
                 "safe_completion": bool(r["all_arrived"] and not r["collision"]),
                 "all_arrived": r["all_arrived"], "timeout": r["timeout"],
                 "completion_s": r["completion_s"], "fails": r["fails"],
                 "solve_mean_ms": r["solve_mean_ms"], "solve_p95_ms": r["solve_p95_ms"],
                 "min_pair_distance": float(np.min(r["pair_min"])) if len(r["pair_min"]) else None})
    print(kind, rows[-1], flush=True)
out = Path(r"D:\paper2\_mpc_cbf_pilot"); out.mkdir(parents=True, exist_ok=True)
with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
(out / "summary.json").write_text(json.dumps({"stage": "mpc_cbf_pilot", "runs": rows},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
