"""阶段一 §1.3 的定向回归：连续相对运动的首次越界时刻与模拟器碰撞统计。"""

from __future__ import annotations

import numpy as np
import pytest

from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc
from fhd.geometry.continuous_collision import (all_pair_events, first_breach_alpha,
                                               pair_continuous_collision,
                                               relative_min_distance)


def test_relative_min_distance_analytic():
    d, a = relative_min_distance(np.array([-1.0, 0.0]), np.array([2.0, 0.0]))
    assert d == pytest.approx(0.0)
    assert a == pytest.approx(0.5)


def test_first_breach_analytic_time():
    # |(-1,0) + a (2,0)| = 0.35  ->  a = 0.325
    a = first_breach_alpha(np.array([-1.0, 0.0]), np.array([2.0, 0.0]), 0.35)
    assert a == pytest.approx(0.325)
    ev = pair_continuous_collision(
        np.array([[-1.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([0.0, 1.0]), 0.35)
    assert ev["collision"] and ev["first_time"] == pytest.approx(0.325)
    assert 0.0 <= ev["first_time"] <= 1.0


def test_mid_interval_crossing_but_endpoints_safe():
    """两端点都安全、中途交换位置：必须判为碰撞。"""
    r0 = np.array([-1.0, 1.0])
    dr = np.array([2.0, -2.0])
    d, _ = relative_min_distance(r0, dr)
    assert d < 1e-9
    a = first_breach_alpha(r0, dr, 0.35)
    assert a is not None and 0.0 < a < 1.0
    # 解析：min ||r0+a dr|| = sqrt(2)*(1-2a) = 0.35 -> a = (1-0.35/sqrt2)/2
    assert a == pytest.approx((1.0 - 0.35 / np.sqrt(2.0)) / 2.0, abs=1e-12)


def test_parallel_motion_never_collides():
    assert first_breach_alpha(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.35) is None


def test_initial_overlap_detected_at_alpha_zero():
    assert first_breach_alpha(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.35) == 0.0
    assert first_breach_alpha(np.array([0.1, 0.0]), np.array([0.0, 0.0]), 0.35) == 0.0


def test_exact_tangency_is_not_a_collision():
    """恰好切阈值不算越界（严格 < d_safe）。"""
    assert first_breach_alpha(np.array([0.0, 0.35]), np.array([1.0, 0.0]), 0.35) is None
    ev = pair_continuous_collision(
        np.array([[0.0, 0.35], [1.0, 0.35]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([0.0, 1.0]), 0.35)
    assert not ev["collision"]
    assert ev["min_distance"] == pytest.approx(0.35, abs=1e-12)


def test_static_relative_motion():
    assert first_breach_alpha(np.array([0.1, 0.0]), np.array([0.0, 0.0]), 0.35) == 0.0
    assert first_breach_alpha(np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.35) is None


def _crossing_paths(n=32, length=8.0):
    t = np.linspace(0, 1, n)
    a = np.stack([length * t, np.ones_like(t)], axis=-1)             # y=1
    b = np.stack([4.0 * np.ones_like(t), -3.0 + length * t], axis=-1)  # x=4
    return np.stack([a, b])


def test_simulator_collision_time_within_horizon_and_matches_pair_event():
    paths = _crossing_paths()
    r = simulate_mpc(paths, np.zeros((2, 31)), 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.collision
    T = MPCSimConfig().horizon_steps * MPCSimConfig().dt_slot
    assert 0.0 <= r.first_collision_time <= T
    finite = r.pair_first_time[np.isfinite(r.pair_first_time)]
    assert r.first_collision_time == pytest.approx(float(finite.min()))
    assert r.first_collision_pair is not None


def test_simulator_initial_overlap_detected_at_t0():
    p = _crossing_paths()
    p[1] = p[0].copy()
    r = simulate_mpc(p, np.zeros((2, 31)), 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.collision and r.first_collision_time == 0.0
    assert r.pair_collision[0] and r.pair_first_time[0] == 0.0


def test_continuous_detection_stable_across_dt_dyn():
    paths = _crossing_paths()
    r1 = simulate_mpc(paths, np.zeros((2, 31)), 0.0,
                      MPCSimConfig(dt_dyn=0.05), seed=0, record=False)
    r2 = simulate_mpc(paths, np.zeros((2, 31)), 0.0,
                      MPCSimConfig(dt_dyn=0.025), seed=0, record=False)
    assert r1.collision and r2.collision
    assert abs(r1.first_collision_time - r2.first_collision_time) < 0.1


def test_n4_six_pairs_are_independent():
    paths = _crossing_paths()
    far = paths[0] + np.array([30.0, 30.0])
    p4 = np.stack([paths[0], paths[1], far, far + np.array([5.0, 0.0])])
    w = np.zeros((4, 31))
    w[2, 0] = 20.0
    w[3, 0] = 20.0
    r = simulate_mpc(p4, w, 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.pair_collision.shape == (6,)
    assert r.pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    flags = {pair: bool(v) for pair, v in zip(r.pairs, r.pair_collision)}
    assert flags[(0, 1)]
    assert not any(flags[p] for p in [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def test_all_pair_events_uses_same_time_grid():
    pi = np.array([[0.0, 0.0], [1.0, 0.0]])
    pj = np.array([[1.0, 0.0], [0.0, 0.0]])
    pos = np.stack([pi, pj], axis=1)
    evs = all_pair_events(pos, np.array([0.0, 0.5]), [(0, 1)], 0.35)
    assert evs[0]["collision"]
    assert evs[0]["first_time"] == pytest.approx(0.325 * 0.5)


def test_no_collision_reports_finite_min_distance():
    paths = _crossing_paths()
    w = np.zeros((2, 31))
    w[1, 0] = 12.0
    r = simulate_mpc(paths, w, 0.0, MPCSimConfig(), seed=0, record=False)
    assert not r.collision
    assert np.all(np.isfinite(r.pair_min_distance))
    assert r.pair_min_distance[0] > MPCSimConfig().d_safe
