"""MMD path import adapter.

MMD is a full planning system, not an online fixed-path coordinator. This module
only converts MMD's native per-agent trajectories into the frozen No02 32-waypoint
representation; it does not add No02 online waiting to MMD.

Native MMD trajectories are (T, 4): ``[x, y, vx, vy]``. We resample x/y by
arc length to H=32 so they can enter the common execution simulator. The
original trajectory and planning time are kept in metadata for separate reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MmdPlan:
    paths: np.ndarray              # [N, 32, 2]
    starts: np.ndarray             # [N, 2]
    goals: np.ndarray              # [N, 2]
    native_paths: list[np.ndarray]  # native [T,4]
    planning_time_s: float
    metadata: dict


def _resample_polyline(points: np.ndarray, h: int = 32) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or p.shape[0] < 2:
        raise ValueError(f"expected polyline [T,>=2], got {p.shape}")
    seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(p[:1, :2], h, axis=0)
    targets = np.linspace(0.0, total, h)
    out = np.zeros((h, 2), dtype=np.float64)
    for k, s in enumerate(targets):
        j = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
        frac = (s - cum[j]) / max(seg[j], 1e-12)
        out[k] = p[j, :2] + frac * (p[j + 1, :2] - p[j, :2])
    return out


def load_mmd_paths(path: str | Path, h: int = 32) -> MmdPlan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    agents = data.get("agents")
    if not agents:
        raise ValueError("MMD JSON has no 'agents'")
    native = [np.asarray(a, dtype=np.float64) for a in agents]
    paths = np.stack([_resample_polyline(a, h) for a in native], axis=0)
    starts = np.asarray(data.get("start_state_pos_l", paths[:, 0, :]), dtype=np.float64)
    goals = np.asarray(data.get("goal_state_pos_l", paths[:, -1, :]), dtype=np.float64)
    starts = starts[:, :2]
    goals = goals[:, :2]
    return MmdPlan(
        paths=paths, starts=starts, goals=goals, native_paths=native,
        planning_time_s=float(data.get("planning_time_s", float("nan"))),
        metadata={k: v for k, v in data.items() if k != "agents"},
    )


def validate_plan(plan: MmdPlan, start_tol: float = 0.25,
                  goal_tol: float = 0.25) -> dict:
    """Check that resampling preserved native start/goal and path endpoints."""
    issues = []
    for i, native in enumerate(plan.native_paths):
        if np.linalg.norm(plan.paths[i, 0] - native[0, :2]) > start_tol:
            issues.append(f"agent{i}_start_shift")
        if np.linalg.norm(plan.paths[i, -1] - native[-1, :2]) > goal_tol:
            issues.append(f"agent{i}_goal_shift")
        if np.linalg.norm(plan.paths[i, 0] - plan.starts[i]) > start_tol:
            issues.append(f"agent{i}_metadata_start")
        if np.linalg.norm(plan.paths[i, -1] - plan.goals[i]) > goal_tol:
            issues.append(f"agent{i}_metadata_goal")
    return {"ok": not issues, "issues": issues,
            "n_agents": len(plan.paths), "h": plan.paths.shape[1]}


def path_deviation(path: np.ndarray, reference: np.ndarray) -> float:
    """Mean nearest-point distance from a trajectory to a reference polyline."""
    a = np.asarray(path, dtype=np.float64)[:, :2]
    r = np.asarray(reference, dtype=np.float64)[:, :2]
    d = np.linalg.norm(a[:, None, :] - r[None, :, :], axis=-1)
    return float(d.min(axis=1).mean())
