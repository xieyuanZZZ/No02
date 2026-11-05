"""在线协调接口、B/C 决策可复现性与动作合法性测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fhd.execution.coordinator import build_observation
from fhd.execution.mpc_executor import SingleRobotExecutor
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc
from fhd.control.linear_mpc import MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams
from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams
from fhd.optim.online_baseline import SimpleConflictCoordinator
from fhd.optim.online_firsthit import OnlineFirstHitCoordinator
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config


def _execs(paths, waits, v0=None, steps=0):
    di = DoubleIntegrator(DoubleIntegratorParams())
    out = []
    for i in range(paths.shape[0]):
        w = np.zeros(32)
        w[:31] = waits[i]
        tp = TimeParameterizedPath(paths[i], w, TimingParams())
        ex = SingleRobotExecutor(tp, di, MPCParams(horizon=15), dt_ctrl=0.1)
        vel = v0[i] if v0 is not None else (0.0, 0.0)
        ex.reset(np.array([paths[i, 0, 0], paths[i, 0, 1], vel[0], vel[1]]))
        for _ in range(steps):
            ex.step()
        out.append(ex)
    return out


def _cross_paths(n=32):
    t = np.linspace(0, 1, n)
    a = np.stack([1.0 + 8.0 * t, 5.0 * np.ones_like(t)], axis=-1)
    b = np.stack([5.0 * np.ones_like(t), 1.0 + 8.0 * t], axis=-1)
    return np.stack([a, b])


def _risk():
    cfg = load_config("configs/base.yaml")
    return FirstHitRisk.from_config(cfg), float(cfg["execution"]["horizon_steps"])


def test_observation_contains_no_future_events():
    paths = _cross_paths()
    waits = np.zeros((2, 31))
    execs = _execs(paths, waits, steps=5)
    obs = build_observation(execs, 0.5, 0.35, 0.4, 3.0, 0.1)
    assert not hasattr(obs, "events")
    assert len(obs.robots) == 2
    for rb in obs.robots:
        assert rb.stop_candidates is not None
        assert "mode" in rb.__dict__


def test_b_no_conflict_does_not_stop():
    t = np.linspace(0, 1, 32)
    paths = np.stack([
        np.stack([1 + 8 * t, 1 + 0 * t], axis=-1),
        np.stack([1 + 8 * t, 8 + 0 * t], axis=-1),
    ])
    waits = np.zeros((2, 31))
    execs = _execs(paths, waits, steps=5)
    obs = build_observation(execs, 0.5, 0.35, 0.4, 3.0, 0.0)
    co = SimpleConflictCoordinator(paths, waits, 0.35, 0.4, 3.0, 0.0, {})
    actions = co.decide(obs)
    assert actions == []


def test_b_conflict_issues_legal_stop_action():
    paths = _cross_paths()
    waits = np.zeros((2, 31))
    execs = _execs(paths, waits, v0=[(1.0, 0.0), (0.0, 1.0)], steps=2)
    obs = build_observation(execs, 0.2, 0.35, 0.4, 3.0, 0.0)
    co = SimpleConflictCoordinator(paths, waits, 0.35, 0.4, 6.0, 0.0,
                                   {"hold_slots": 1})
    actions = co.decide(obs)
    assert actions
    for a in actions:
        assert a.robot in (0, 1)
        assert a.s_target is not None and a.s_target > obs.robots[a.robot].s
        assert a.duration_s == pytest.approx(0.4)


def test_b_no_feasible_stop_records_and_returns_empty():
    paths = _cross_paths()
    waits = np.zeros((2, 31))
    execs = _execs(paths, waits, v0=[(1.5, 0.0), (0.0, 1.5)], steps=1)
    obs = build_observation(execs, 0.1, 0.35, 0.4, 3.0, 0.0)
    co = SimpleConflictCoordinator(paths, waits, 0.35, 0.4, 3.0, 0.0,
                                   {"lookahead_m": 0.05})
    actions = co.decide(obs)
    assert actions == []
    assert co.last_diagnostics["no_feasible_stop"] >= 1


def test_c_residual_path_starts_at_current_progress():
    paths = _cross_paths()
    waits = np.zeros((2, 31))
    execs = _execs(paths, waits, steps=20)
    obs = build_observation(execs, 2.0, 0.35, 0.4, 3.2, 0.1)
    risk, T = _risk()
    co = OnlineFirstHitCoordinator(paths, waits, 0.35, 0.4, 3.2, 0.1,
                                   risk_model=risk,
                                   config={"hold_slots": 1})
    pr, wr, next_idx = co._residual_model(obs)
    for i in range(2):
        cur = co.s_to_position(paths[i], obs.robots[i].s)
        assert np.linalg.norm(pr[i, 0].numpy() - cur) < 1e-9
        assert next_idx[i] >= 1


def test_c_same_observation_is_reproducible():
    paths = _cross_paths()
    waits = np.zeros((2, 31))
    execs = _execs(paths, waits, v0=[(1.0, 0.0), (0.0, 1.0)], steps=2)
    obs = build_observation(execs, 0.2, 0.35, 0.4, 3.2, 0.0)
    risk, T = _risk()
    cfg = {"hold_slots": 1, "max_candidates_per_robot": 2}
    a = OnlineFirstHitCoordinator(paths, waits, 0.35, 0.4, 3.2, 0.0, risk, cfg).decide(obs)
    b = OnlineFirstHitCoordinator(paths, waits, 0.35, 0.4, 3.2, 0.0, risk, cfg).decide(obs)
    assert [(x.robot, x.s_target, x.duration_s) for x in a] == \
           [(x.robot, x.s_target, x.duration_s) for x in b]


def test_c_no_conflict_no_action():
    t = np.linspace(0, 1, 32)
    paths = np.stack([
        np.stack([1 + 8 * t, 1 + 0 * t], axis=-1),
        np.stack([1 + 8 * t, 8 + 0 * t], axis=-1),
    ])
    waits = np.zeros((2, 31))
    execs = _execs(paths, waits, steps=5)
    obs = build_observation(execs, 0.5, 0.35, 0.4, 3.2, 0.0)
    risk, T = _risk()
    co = OnlineFirstHitCoordinator(paths, waits, 0.35, 0.4, 3.2, 0.0, risk, {})
    assert co.decide(obs) == []


def test_simulator_coordinator_none_matches_no_commands():
    paths = _cross_paths()
    waits = np.zeros((2, 31))
    r = simulate_mpc(paths, waits, 0.0, MPCSimConfig(), seed=1, record=False)
    assert r.coordinator_name is None
    assert r.online_requests == 0
    assert r.coordinator_stats["n_decisions"] == 0


def test_simulator_runs_b_and_c_and_logs_runtime():
    paths = _cross_paths()
    waits = np.zeros((2, 31))
    risk, T = _risk()
    cfg = MPCSimConfig()
    b = SimpleConflictCoordinator(paths, waits, cfg.d_safe, cfg.dt_slot, 3.0, 0.1,
                                  {"hold_slots": 1})
    r_b = simulate_mpc(paths, waits, 0.1, cfg, seed=2, record=True, coordinator=b)
    assert r_b.coordinator_name == "B_simple_online"
    assert r_b.coordinator_stats["n_decisions"] > 0
    assert r_b.coordinator_stats["mean_ms"] >= 0
    c = OnlineFirstHitCoordinator(paths, waits, cfg.d_safe, cfg.dt_slot, 3.2, 0.1,
                                  risk_model=risk, config={"hold_slots": 1})
    r_c = simulate_mpc(paths, waits, 0.1, cfg, seed=2, record=True, coordinator=c)
    assert r_c.coordinator_name == "C_online_firsthit"
    assert r_c.coordinator_stats["n_decisions"] > 0


def test_coordinator_exception_is_recorded_not_crashing():
    paths = _cross_paths()
    waits = np.zeros((2, 31))

    class Boom:
        name = "boom"

        def observe(self, obs):
            pass

        def decide(self, obs):
            raise RuntimeError("boom")

    r = simulate_mpc(paths, waits, 0.0, MPCSimConfig(), seed=0, record=True,
                     coordinator=Boom())
    assert r.coordinator_stats["errors"] >= 1
    assert r.all_arrived
