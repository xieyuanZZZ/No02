"""Multi-robot MPC-CBF for No02 double-integrator dynamics.

This is an independent controller (not the frozen No02 MPC). It follows the
official MPC-CBF idea: receding-horizon double-integrator optimization with
discrete-time control-barrier-function constraints on every robot pair.
"""

from __future__ import annotations

import time

import casadi as ca
import numpy as np


def run_mpc_cbf(paths: np.ndarray, events=(), dt: float = 0.1,
                steps: int = 384, vmax: float = 1.5, amax: float = 2.0,
                d_safe: float = 0.35, gamma: float = 0.1, horizon: int = 4,
                q_pos: float = 10.0, r_u: float = 1.0) -> dict:
    paths = np.asarray(paths, dtype=float)
    n, h, _ = paths.shape
    evs = list(events)
    pause_until = np.full(n, -1.0)
    pos = paths[:, 0, :].copy()
    vel = np.zeros((n, 2))
    idx = np.ones(n, dtype=int)
    traj = np.zeros((steps + 1, n, 2))
    vel_t = np.zeros((steps + 1, n, 2))
    fails = 0
    solve_ms = []
    completion = None

    def target_for(i):
        k = min(idx[i], h - 1)
        # look a few points ahead so the optimizer does not stop at a waypoint
        k2 = min(k + 2, h - 1)
        return paths[i, k2]

    def solve(x):
        opti = ca.Opti()
        X = opti.variable(4 * n, horizon + 1)
        U = opti.variable(2 * n, horizon)
        opti.subject_to(X[:, 0] == ca.DM(x.reshape(-1)))
        for k in range(horizon):
            for i in range(n):
                p = X[4 * i:4 * i + 2, k]
                v = X[4 * i + 2:4 * i + 4, k]
                a = U[2 * i:2 * i + 2, k]
                p2 = p + v * dt + 0.5 * a * dt * dt
                v2 = v + a * dt
                opti.subject_to(X[4 * i:4 * i + 4, k + 1] == ca.vertcat(p2, v2))
                opti.subject_to(ca.sumsqr(v2) <= vmax * vmax)
                opti.subject_to(ca.sumsqr(a) <= amax * amax)
            for i in range(n):
                for j in range(i + 1, n):
                    pi = X[4 * i:4 * i + 2, k]
                    pj = X[4 * j:4 * j + 2, k]
                    pi2 = X[4 * i:4 * i + 2, k + 1]
                    pj2 = X[4 * j:4 * j + 2, k + 1]
                    hk = ca.sumsqr(pi - pj) - d_safe * d_safe
                    hk2 = ca.sumsqr(pi2 - pj2) - d_safe * d_safe
                    opti.subject_to(hk2 >= (1.0 - gamma) * hk)
        cost = 0
        for k in range(horizon + 1):
            for i in range(n):
                p = X[4 * i:4 * i + 2, k]
                v = X[4 * i + 2:4 * i + 4, k]
                pref = ca.DM(target_for(i))
                cost += q_pos * ca.sumsqr(p - pref) + 0.1 * ca.sumsqr(v)
                if k < horizon:
                    cost += r_u * ca.sumsqr(U[2 * i:2 * i + 2, k])
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False},
                    {"max_iter": 80, "print_level": 0, "sb": "yes"})
        try:
            sol = opti.solve()
            return np.array(sol.value(U[:, 0])).reshape(n, 2), True
        except Exception:
            return np.zeros((n, 2)), False

    for s in range(steps + 1):
        t = s * dt
        for slot, rid, dur in evs:
            if abs(slot * 0.4 - t) < 1e-9:
                pause_until[int(rid)] = t + float(dur)
        traj[s] = pos
        vel_t[s] = vel
        if s == steps:
            break
        x = np.concatenate([np.concatenate([pos[i], vel[i]]) for i in range(n)])
        t0 = time.perf_counter()
        u, ok = solve(x)
        solve_ms.append((time.perf_counter() - t0) * 1e3)
        if not ok:
            fails += 1
            u[:] = 0.0
        for i in range(n):
            if t < pause_until[i] - 1e-9:
                u[i] = 0.0
            vel[i] += np.clip(u[i], -amax, amax) * dt
            sp = np.linalg.norm(vel[i])
            if sp > vmax:
                vel[i] *= vmax / sp
            pos[i] += vel[i] * dt
            if t >= pause_until[i] - 1e-9:
                nearest = int(np.argmin(np.linalg.norm(paths[i] - pos[i], axis=1)))
                idx[i] = max(idx[i], min(nearest + 1, h))
                while idx[i] < h and np.linalg.norm(pos[i] - paths[i, idx[i]]) < 0.15:
                    idx[i] += 1
        if completion is None and (idx >= h).all():
            completion = float(s * dt)
            break
    d = np.linalg.norm(traj[:, :, None, :] - traj[:, None, :, :], axis=-1)
    iu = np.triu_indices(n, 1)
    pair_min = d[:, iu[0], iu[1]].min(axis=0) if n > 1 else np.array([])
    done = (idx >= h).all()
    completion = float(completion if completion is not None else steps * dt)
    return {"traj": traj, "vel": vel_t, "pair_min": pair_min,
            "collision": bool((pair_min < d_safe).any()),
            "all_arrived": bool(done), "completion_s": completion,
            "timeout": bool(not done), "fails": int(fails),
            "solve_p95_ms": float(np.percentile(solve_ms, 95)) if solve_ms else 0.0,
            "solve_mean_ms": float(np.mean(solve_ms)) if solve_ms else 0.0}
