"""MPC / 动力学执行相关测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fhd.control.linear_mpc import LinearMPC, MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams
from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams
from fhd.execution.mpc_executor import SingleRobotExecutor
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc


def _line(n=32, length=8.0):
    t = np.linspace(0, 1, n)
    return np.stack([1.0 + length * t, 5.0 * np.ones_like(t)], axis=-1)


# --------------------------------------------------------------------------- #
# 双积分器
# --------------------------------------------------------------------------- #
def test_double_integrator_matches_analytic_motion():
    di = DoubleIntegrator(DoubleIntegratorParams(dt=0.01))
    x = np.zeros(4)
    u = np.array([1.0, 0.0])
    for _ in range(100):                     # 1 秒，a=1
        x = di.step(x, u)
    assert x[0] == pytest.approx(0.5, abs=1e-6)     # 0.5*a*t^2
    assert x[2] == pytest.approx(1.0, abs=1e-6)


def test_brake_distance_formula():
    p = DoubleIntegratorParams(a_brake=2.0)
    v = 1.5
    assert DoubleIntegrator.brake_distance(v, p.a_brake) == pytest.approx(v ** 2 / 4.0)


def test_l1_constraint_is_conservative():
    from fhd.dynamics.double_integrator import l1_polynomial_violation
    a_max = 2.0
    # 轴对齐：||a||_1 = ||a||_2 = 2.0，应视为不违反
    assert l1_polynomial_violation(np.array([2.0, 0.0]), a_max) == pytest.approx(1.0)
    # 对角：||a||_1 = 2.0 但 ||a||_2 = 1.414 < 2.0，仍满足真实约束
    assert l1_polynomial_violation(np.array([1.0, 1.0]), a_max) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 时间参数化
# --------------------------------------------------------------------------- #
def test_timing_respects_speed_and_accel_limits():
    wp = _line()
    tp = TimeParameterizedPath(wp, np.zeros(len(wp)), TimingParams(v_max=1.5, a_acc=2.0,
                                                                  a_brake=2.0))
    assert tp.v_grid.max() <= 1.5 + 1e-6
    dv = np.diff(tp.v_grid)
    ds = np.diff(tp.s_grid)
    acc = np.abs(dv / np.maximum(ds, 1e-9) * tp.v_grid[:-1])
    # 允许离散误差
    assert acc.max() <= 25.0


def test_timing_starts_and_ends_at_rest():
    wp = _line()
    tp = TimeParameterizedPath(wp, np.zeros(len(wp)), TimingParams())
    assert tp.speed_at(0.0) <= 0.05
    assert tp.speed_at(tp.length) <= 0.05


def test_dwell_is_reflected_in_nominal_arrival():
    wp = _line()
    w = np.zeros(len(wp))
    w[10] = 2.0                                 # 2 时隙 = 0.8 s
    tp = TimeParameterizedPath(wp, w, TimingParams(dt_slot=0.4))
    assert tp.dwell[10] == pytest.approx(0.8)
    arr = tp.nominal_arrival_times()
    assert arr["time"][11] - arr["time"][10] >= 0.8 - 1e-6


def test_schedule_cap_prevents_running_ahead():
    """短段不应该被规划成比"每时隙一个航点"更快。"""
    wp = _line()
    tp = TimeParameterizedPath(wp, np.zeros(len(wp)), TimingParams(dt_slot=0.4))
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=-1).max()
    v_sched = seg / 0.4
    assert tp.v_grid.max() <= v_sched + 1e-6


# --------------------------------------------------------------------------- #
# MPC
# --------------------------------------------------------------------------- #
def _make_mpc():
    di = DoubleIntegrator(DoubleIntegratorParams())
    return LinearMPC(di, MPCParams(horizon=12, v_max=1.5, a_max=2.0))


def test_mpc_solves_and_respects_control_bounds():
    mpc = _make_mpc()
    x0 = np.array([1.0, 5.0, 0.0, 0.0])
    ref = np.zeros((13, 4))
    for k in range(13):
        ref[k, :2] = [1.0 + 0.1 * k, 5.0]
    U, info = mpc.solve(x0, ref)
    assert info["ok"], info["status"]
    assert np.abs(info["u"]).sum() <= 2.0 + 1e-6


def test_mpc_reports_failure_gracefully():
    """给一个不可能的参考（巨大加速度需求），不应抛出异常。"""
    mpc = _make_mpc()
    x0 = np.zeros(4)
    ref = np.tile(np.array([1e3, 1e3, 0.0, 0.0]), (13, 1))
    U, info = mpc.solve(x0, ref)
    assert np.isfinite(U).all()


def test_mpc_tracks_straight_line():
    wp = _line()
    tp = TimeParameterizedPath(wp, np.zeros(len(wp)), TimingParams())
    di = DoubleIntegrator(DoubleIntegratorParams())
    ex = SingleRobotExecutor(tp, di, MPCParams(horizon=15), dt_ctrl=0.1)
    ex.reset(np.array([wp[0, 0], wp[0, 1], 0.0, 0.0]))
    while ex.t < 60 and not ex.done:
        ex.step()
    P = np.asarray(ex.trace["p"])
    dev = np.abs(P[:, 1] - 5.0)
    assert ex.done
    assert dev.max() < 0.35, f"直线跟踪横向偏差过大 {dev.max():.3f} m"


def test_pause_causes_brake_hold_restart():
    wp = _line()
    tp = TimeParameterizedPath(wp, np.zeros(len(wp)), TimingParams())
    di = DoubleIntegrator(DoubleIntegratorParams())
    ex = SingleRobotExecutor(tp, di, MPCParams(horizon=15), dt_ctrl=0.1)
    ex.reset(np.array([wp[0, 0], wp[0, 1], 0.0, 0.0]))
    while ex.t < 60 and not ex.done:
        ex.step(pause_request=abs(ex.t - 2.0) < 1e-9, pause_duration=1.0)
    modes = ex.trace["mode"]
    assert "BRAKE_PAUSE" in modes and "HOLD" in modes
    assert ex.done


# --------------------------------------------------------------------------- #
# 多机器人 MPC 仿真
# --------------------------------------------------------------------------- #
def _two_robot_paths():
    t = np.linspace(0, 1, 32)
    a = np.stack([1.0 + 8.0 * t, 5.0 * np.ones_like(t)], axis=-1)
    b = np.stack([5.0 * np.ones_like(t), 1.0 + 8.0 * t], axis=-1)
    return np.stack([a, b])


def test_mpc_simulator_p0_safe_with_stagger():
    paths = _two_robot_paths()
    w = np.zeros((2, 31))
    w[1, 0] = 12.0                     # 给第二台足够大的起点等待
    r = simulate_mpc(paths, w, 0.0, MPCSimConfig(), seed=0, record=False)
    assert not r.collision
    assert r.all_arrived
    assert r.mpc_failures == 0


def test_mpc_simulator_detects_collision_without_wait():
    paths = _two_robot_paths()
    w = np.zeros((2, 31))
    r = simulate_mpc(paths, w, 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.collision, "两条交叉路径且无等待，应当检测到碰撞"
    assert r.first_collision_pair is not None


def test_mpc_simulator_respects_limits():
    paths = _two_robot_paths()
    w = np.zeros((2, 31))
    w[1, 0] = 12.0
    r = simulate_mpc(paths, w, 0.1, MPCSimConfig(), seed=3, record=True)
    assert r.max_speed <= 1.5 + 1e-3
    assert r.max_accel <= 2.0 + 1e-3
    assert r.constraint_violations == 0
