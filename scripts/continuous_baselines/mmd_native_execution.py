"""Execute official MMD native 64-point trajectories with a double-integrator tracker.

Native MMD radius is 0.05 m (contact distance 0.10 m), duration 5.0 s,
horizon 64 -> dt=5/64. No First-Hit/W2+ is added.
"""

from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fhd.adapters.mmd_adapter import load_mmd_paths  # noqa: E402

plan = load_mmd_paths(ROOT / "results/Continuous_Baseline_Comparison/native_reproduction/mmd/mmd_paths.json")
native = np.stack(plan.native_paths)
n = len(native)
dt_mmd = 5.0 / native[0].shape[0]      # official 5.0 s / 64
duration = 5.0
dt = 0.1
steps = int(duration / dt)
vmax = 1.5
amax = 2.0
native_safe = 0.10                     # official radius 0.05 contact distance
rng = np.random.default_rng(20281501)
events = [(slot, int(rng.integers(n)), 0.4) for slot in range(30) if rng.random() < 0.1]
pos = native[0][0, :2].copy()
vel = np.zeros((n, 2))
pos = native[:, 0, :2].copy()
pause_until = np.full(n, -1.0)
traj = [pos.copy()]
vel_t = [vel.copy()]
times = [0.0]
for s in range(steps):
    t = s * dt
    for slot, rid, dur in events:
        if abs(slot * 0.4 - t) < 1e-9:
            pause_until[rid] = t + dur
    k = min(int(t / dt_mmd), native[0].shape[0] - 1)
    k2 = min(k + 1, native[0].shape[0] - 1)
    for i in range(n):
        if t < pause_until[i] - 1e-9:
            vdes = np.zeros(2)
        else:
            # derivative of the original 64-point trajectory
            vdes = (native[i][k2, :2] - native[i][k, :2]) / dt_mmd + 4.0 * (native[i][k, :2] - pos[i])
            sp = np.linalg.norm(vdes)
            if sp > vmax:
                vdes *= vmax / sp
        acc = np.clip((vdes - vel[i]) / dt, -amax, amax)
        vel[i] += acc * dt
        sp = np.linalg.norm(vel[i])
        if sp > vmax:
            vel[i] *= vmax / sp
        pos[i] += vel[i] * dt
    traj.append(pos.copy()); vel_t.append(vel.copy()); times.append((s + 1) * dt)
traj = np.stack(traj); vel_t = np.stack(vel_t)
d = np.linalg.norm(traj[:, :, None, :] - traj[:, None, :, :], axis=-1)
iu = np.triu_indices(n, 1)
pair_min = d[:, iu[0], iu[1]].min(axis=0)
goal_err = np.linalg.norm(traj[-1] - native[:, -1, :2], axis=1)
row = {"method": "mmd_native", "n_agents": n, "duration_s": duration,
       "dt_mmd": dt_mmd, "collision_at_010": bool((pair_min < native_safe).any()),
       "collision_at_035": bool((pair_min < 0.35).any()),
       "all_arrived": bool((goal_err < 0.25).all()),
       "min_pair_distance": float(pair_min.min()),
       "goal_error_max": float(goal_err.max()),
       "max_speed": float(np.linalg.norm(vel_t, axis=-1).max())}
out = Path(r"D:\paper2\_mmd_pilot2"); out.mkdir(parents=True, exist_ok=True)
with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
(out / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
print(row)
