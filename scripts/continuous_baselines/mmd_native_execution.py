"""Execute the official MMD plan in a double-integrator tracker.

This is a native-plan execution demonstration, not the No02-map Experiment B:
the official MMD example has its own map/start-goals.  No First-Hit/offline
waiting is added to MMD.
"""

from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fhd.adapters.mmd_adapter import load_mmd_paths  # noqa: E402

plan = load_mmd_paths(ROOT / "results/Continuous_Baseline_Comparison/native_reproduction/mmd/mmd_paths.json")
paths = plan.paths
n = len(paths); dt = 0.1; duration = 3.0; steps = int(duration / dt)
vmax = 1.5; amax = 2.0; d_safe = 0.35
rng = np.random.default_rng(20281501)
events = [(slot, int(rng.integers(n)), 0.4) for slot in range(30) if rng.random() < 0.1]
pos = paths[:, 0, :].copy(); vel = np.zeros((n, 2)); pause_until = np.full(n, -1.0)
traj = np.zeros((steps + 1, n, 2))
for s in range(steps + 1):
    t = s * dt
    for slot, rid, dur in events:
        if abs(slot * 0.4 - t) < 1e-9:
            pause_until[rid] = t + dur
    traj[s] = pos
    if s == steps:
        break
    frac = min(t / duration, 1.0)
    k = min(int(frac * (paths.shape[1] - 1)), paths.shape[1] - 1)
    for i in range(n):
        if t < pause_until[i] - 1e-9:
            vdes = np.zeros(2)
        else:
            k2 = min(k + 1, paths.shape[1] - 1)
            vdes = (paths[i, k2] - paths[i, k]) / dt
            sp = np.linalg.norm(vdes)
            if sp > vmax:
                vdes *= vmax / sp
        acc = np.clip((vdes - vel[i]) / dt, -amax, amax)
        vel[i] += acc * dt
        sp = np.linalg.norm(vel[i])
        if sp > vmax:
            vel[i] *= vmax / sp
        pos[i] += vel[i] * dt
d = np.linalg.norm(traj[:, :, None, :] - traj[:, None, :, :], axis=-1)
iu = np.triu_indices(n, 1)
pair_min = d[:, iu[0], iu[1]].min(axis=0)
goal = paths[:, -1, :]
arrived = np.linalg.norm(traj[-1] - goal, axis=1) < 0.25
row = {"method": "mmd_native", "n_agents": n, "collision": bool((pair_min < d_safe).any()),
       "safe_completion": bool(arrived.all() and not (pair_min < d_safe).any()),
       "all_arrived": bool(arrived.all()), "min_pair_distance": float(pair_min.min()),
       "duration_s": duration, "max_speed": float(np.linalg.norm(np.diff(traj, axis=0) / dt, axis=-1).max())}
out = Path(r"D:\paper2\_mmd_pilot"); out.mkdir(parents=True, exist_ok=True)
with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
(out / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
print(row)
