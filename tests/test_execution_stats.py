"""阶段一 §1.4 的定向回归：统计口径、record 一致性、逐对/全队关系。"""

from __future__ import annotations

import numpy as np
import pytest

from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc
from fhd.geometry.primitives import Obstacles
from fhd.geometry.static_maps import load_obstacles_for_map

N = 32


def _paths(n=32, length=8.0):
    t = np.linspace(0, 1, n)
    a = np.stack([length * t, np.ones_like(t)], axis=-1)
    b = np.stack([4.0 * np.ones_like(t), -3.0 + length * t], axis=-1)
    return np.stack([a, b])


def test_record_true_false_have_identical_metrics():
    paths = _paths()
    w = np.zeros((2, N - 1))
    w[1, 0] = 3.0
    cfg = MPCSimConfig()
    r1 = simulate_mpc(paths, w, 0.2, cfg, seed=11, record=True)
    r0 = simulate_mpc(paths, w, 0.2, cfg, seed=11, record=False)
    assert r1.trajectories is not None and r0.trajectories is None
    for name in ("collision", "all_arrived", "time_censored", "timeout",
                 "safe_completion", "completion_time", "tracking_rmse",
                 "max_speed", "max_accel", "infeasible_stops", "mpc_failures",
                 "constraint_violations", "pause_requests", "pause_accepted"):
        assert getattr(r1, name) == getattr(r0, name), name
    assert np.allclose(r1.pair_min_distance, r0.pair_min_distance)
    assert np.allclose(r1.pair_first_time, r0.pair_first_time, equal_nan=True)
    # 墙钟时间不可能逐位一致；计数/状态必须一致
    timing_keys = {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"}
    assert {k: v for k, v in r1.solve_stats.items() if k not in timing_keys} == \
           {k: v for k, v in r0.solve_stats.items() if k not in timing_keys}


def test_team_collision_is_min_of_pair_events():
    paths = _paths()
    r = simulate_mpc(paths, np.zeros((2, N - 1)), 0.0, MPCSimConfig(), seed=3, record=False)
    assert r.collision == bool(r.pair_collision.any())
    finite = r.pair_first_time[np.isfinite(r.pair_first_time)]
    if finite.size:
        assert r.first_collision_time == pytest.approx(float(finite.min()))
        assert r.first_collision_pair is not None
        assert r.pairs[r.pair_collision.tolist().index(True)] == r.first_collision_pair


def test_solve_stats_aggregate_all_mpc_calls_not_last():
    paths = _paths()
    r = simulate_mpc(paths, np.zeros((2, N - 1)), 0.0,
                     MPCSimConfig(horizon_steps=10), seed=0, record=False)
    ss = r.solve_stats
    assert ss["n_solves"] == r.n_steps_executed * 2
    assert ss["calls"] == ss["n_solves"]
    assert ss["mean_ms"] > 0
    assert ss["p50_ms"] <= ss["p95_ms"] <= ss["p99_ms"]
    assert ss["failure_rate"] == pytest.approx(ss["failed"] / max(ss["calls"], 1))


def test_safe_completion_is_no_collision_and_all_arrived():
    paths = _paths()
    w = np.zeros((2, N - 1))
    w[1, 0] = 12.0
    r = simulate_mpc(paths, w, 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.all_arrived and not r.collision
    assert r.safe_completion


def test_static_obstacle_status_not_evaluated_without_map():
    paths = _paths()
    r = simulate_mpc(paths, np.zeros((2, N - 1)), 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.static_obstacle_status == "not_evaluated"
    assert r.obstacle_collision is None
    assert r.min_obstacle_clearance is None


def test_static_obstacle_clearance_computed_when_obstacles_given():
    import torch
    paths = _paths()
    obs = Obstacles(circles=torch.tensor([[100.0, 100.0, 0.2]]),
                    rects=torch.zeros(0, 4))
    r = simulate_mpc(paths, np.zeros((2, N - 1)), 0.0, MPCSimConfig(), seed=0,
                     record=False, obstacles=obs, obstacle_meta={"source": "unit-test"})
    assert r.static_obstacle_status == "evaluated"
    assert r.obstacle_collision is False
    assert r.min_obstacle_clearance > 0


def test_map_id_catalog_resolves_plan_bank_maps():
    obs, meta = load_obstacles_for_map("open_crossing_300000")
    assert meta["status"] == "evaluated"
    assert obs is not None
    obs2, meta2 = load_obstacles_for_map("does_not_exist_999999")
    assert meta2["status"] == "not_evaluated" and obs2 is None
