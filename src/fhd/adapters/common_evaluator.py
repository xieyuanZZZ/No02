"""Independent evaluator used by every external baseline and by Ours.

It reads actual trajectories only; it does not call any controller.  This avoids
method-specific collision rules (sample counts, target-index arrival, etc.).
"""

from __future__ import annotations

import numpy as np


def _rect_dist(p, r):
    x1, y1, x2, y2 = r
    dx = max(x1 - p[0], 0.0, p[0] - x2)
    dy = max(y1 - p[1], 0.0, p[1] - y2)
    return float(np.hypot(dx, dy))


def _seg_min_pair(a0, a1, b0, b1, n=21):
    """Dense sub-step fallback for a pair of linear intervals."""
    u = np.linspace(0.0, 1.0, n)[:, None]
    a = a0[None] * (1 - u) + a1[None] * u
    b = b0[None] * (1 - u) + b1[None] * u
    return float(np.linalg.norm(a - b, axis=1).min())


def evaluate_run(times, positions, velocities, paths, *,
                 rects=(), circles=(), d_safe=0.35, robot_radius=0.15,
                 clearance=0.03, goal_tol=0.25, sprint_tol=0.25) -> dict:
    times = np.asarray(times, float)
    pos = np.asarray(positions, float)
    vel = np.asarray(velocities, float)
    paths = np.asarray(paths, float)
    if len(times) != len(pos) or len(pos) != len(vel):
        raise ValueError("times/positions/velocities length mismatch")
    if len(times) < 2:
        raise ValueError("trajectory too short")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    n = pos.shape[1]
    iu = np.triu_indices(n, 1)
    pair_min = np.full(len(iu[0]), np.inf)
    first_hit = np.full(len(iu[0]), np.inf)
    for k in range(len(times) - 1):
        for q, (i, j) in enumerate(zip(*iu)):
            d = _seg_min_pair(pos[k, i], pos[k + 1, i],
                              pos[k, j], pos[k + 1, j])
            if d < pair_min[q]:
                pair_min[q] = d
            if d < d_safe and not np.isfinite(first_hit[q]):
                first_hit[q] = float(times[k])
    static_min = np.inf
    for i in range(n):
        for k in range(len(times) - 1):
            for f in np.linspace(0, 1, 11):
                p = pos[k, i] * (1 - f) + pos[k + 1, i] * f
                for r in rects:
                    static_min = min(static_min, _rect_dist(p, r) - robot_radius - clearance)
                for c in circles:
                    static_min = min(static_min, np.hypot(p[0] - c[0], p[1] - c[1])
                                     - c[2] - robot_radius - clearance)
    goal_err = np.linalg.norm(pos[-1] - paths[:, -1, :], axis=1)
    end_speed = np.linalg.norm(vel[-1], axis=1)
    arrived = bool((goal_err < goal_tol).all() and (end_speed < sprint_tol).all())
    inter = bool((pair_min < d_safe).any())
    static = bool(static_min < 0.0) if (len(rects) or len(circles)) else None
    return {
        "inter_robot_violation": inter,
        "static_obstacle_violation": static,
        "static_status": "evaluated" if static is not None else "not_evaluated",
        "min_pair_distance": float(pair_min.min()) if len(pair_min) else None,
        "pair_min": pair_min.tolist(),
        "first_pair_violation_times": first_hit.tolist(),
        "min_static_clearance": float(static_min) if np.isfinite(static_min) else None,
        "all_arrived": arrived,
        "goal_error_max": float(goal_err.max()),
        "end_speed_max": float(end_speed.max()),
        "safe_completion": bool(arrived and not inter and static is not True),
        "n_steps": len(times),
    }
