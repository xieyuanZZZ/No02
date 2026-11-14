"""ORCA/RVO2 adapter: official RVO2 core + acceleration-limited velocity tracking.

The official RVO2 library computes the ORCA safe velocity; this module does not
teleport agents. A separate step integrates the actual double-integrator state
with |a| <= a_max and feeds it back to RVO2 at every 0.1 s.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

EXE = str(Path(__file__).resolve().parents[3] / "third_party" / "orca_runner.exe")


def run_orca(paths: np.ndarray, events=(), dt: float = 0.1,
             steps: int = 960, vmax: float = 1.5, amax: float = 2.0,
             radius: float = 0.175, neighbor: float = 3.0,
             horizon: float = 2.0, exe: str = EXE) -> dict:
    paths = np.asarray(paths, dtype=float)
    n, h, _ = paths.shape
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{n} {dt} {steps} {vmax} {amax} {radius} {neighbor} {horizon}\n")
        for p in paths:
            f.write(str(h) + " " + " ".join(f"{x:.9f} {y:.9f}" for x, y in p) + "\n")
        evs = list(events)
        f.write(str(len(evs)) + "\n")
        for slot, rid, dur in evs:
            f.write(f"{slot * 0.4:.6f} {int(rid)} {float(dur):.6f}\n")
        path = f.name
    try:
        proc = subprocess.run([exe, path], capture_output=True, text=True,
                              timeout=120)
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ORCA runner failed: {proc.stderr[:500]}")
    rows = [ln.split(",") for ln in proc.stdout.strip().splitlines()[1:] if ln]
    arr = np.asarray(rows, dtype=float)
    # columns: step, robot, x,y,vx,vy,target,paused
    traj = np.zeros((steps + 1, n, 2))
    vel = np.zeros((steps + 1, n, 2))
    target = np.zeros((steps + 1, n), dtype=int)
    paused = np.zeros((steps + 1, n), dtype=bool)
    for r in arr:
        k, i = int(r[0]), int(r[1])
        traj[k, i] = r[2:4]
        vel[k, i] = r[4:6]
        target[k, i] = int(r[6])
        paused[k, i] = bool(r[7])
    iu = np.triu_indices(n, 1)
    pair_min = np.full(len(iu[0]), np.inf)
    for k in range(steps):
        for sub in range(6):
            a = sub / 5.0
            p = traj[k] * (1.0 - a) + traj[k + 1] * a
            dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            if len(iu[0]):
                pair_min = np.minimum(pair_min, dd[iu])
    pair_coll = (pair_min < 0.35)
    finished = (target >= h).all(axis=1)
    if finished.any():
        finish_step = int(np.argmax(finished))
        completion = float(finish_step * dt)
        timeout = False
    else:
        completion = float(steps * dt)
        timeout = True
    speed = np.linalg.norm(vel, axis=-1)
    return {
        "traj": traj, "vel": vel, "target": target, "paused": paused,
        "pair_min": pair_min, "collision": bool(pair_coll.any()),
        "completion_s": completion, "timeout": timeout,
        "all_arrived": bool(not timeout),
        "max_speed": float(speed.max()), "steps": steps,
        "online_requests": int(paused.sum()),
    }


def path_deviation(traj: np.ndarray, paths: np.ndarray) -> float:
    vals = []
    for i in range(paths.shape[0]):
        a = traj[:, i, :]
        r = paths[i]
        best = np.full(len(a), np.inf)
        for k in range(len(r) - 1):
            ab = r[k + 1] - r[k]
            den = float(ab @ ab)
            if den < 1e-12:
                continue
            u = np.clip(((a - r[k]) @ ab) / den, 0.0, 1.0)
            q = r[k] + u[:, None] * ab[None, :]
            best = np.minimum(best, np.linalg.norm(a - q, axis=1))
        vals.append(float(best.mean()))
    return float(np.mean(vals))
