"""P0 修复的回归测试：预测矩阵、等待时序、逐对碰撞、跟踪误差、事件一致性。"""

from __future__ import annotations

import numpy as np
import pytest

from fhd.control.linear_mpc import LinearMPC, MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams
from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams
from fhd.execution.mpc_executor import SingleRobotExecutor
from fhd.execution.mpc_simulator import MPCSimConfig, make_pause_events, simulate_mpc


def _line(n=32, length=6.0):
    t = np.linspace(0, 1, n)
    return np.stack([1.0 + length * t, 5.0 * np.ones_like(t)], axis=-1)


def _two_robot_paths():
    t = np.linspace(0, 1, 32)
    a = np.stack([1.0 + 8.0 * t, 5.0 * np.ones_like(t)], axis=-1)
    b = np.stack([5.0 * np.ones_like(t), 1.0 + 8.0 * t], axis=-1)
    return np.stack([a, b])


# --------------------------------------------------------------------------- #
# 3.1 预测矩阵
# --------------------------------------------------------------------------- #
def test_prediction_matrix_matches_stepwise_simulation():
    """X = F x0 + G U 必须与逐步仿真完全一致（旧实现漏掉了 Ac 的一次幂）。"""
    di = DoubleIntegrator(DoubleIntegratorParams())
    mpc = LinearMPC(di, MPCParams(horizon=10))
    rng = np.random.default_rng(0)
    for _ in range(5):
        x0 = rng.normal(size=4)
        U = rng.normal(size=(10, 2))
        X = [x0.copy()]
        for k in range(10):
            X.append(di.Ac @ X[-1] + di.Bc @ U[k])
        X = np.array(X)
        Xm = (mpc.F @ x0 + mpc.G @ U.reshape(-1)).reshape(11, 4)
        assert np.abs(Xm - X).max() < 1e-10


def test_x0_does_not_depend_on_u0():
    """旧实现把 u_0 混进了 x_0；这里回归检查 G 的前 4 行必须为 0。"""
    di = DoubleIntegrator(DoubleIntegratorParams())
    mpc = LinearMPC(di, MPCParams(horizon=6))
    assert np.allclose(mpc.G[:4, :], 0.0)


def test_first_control_affects_exactly_a_k1_b_u0():
    """x_1 = Ac x_0 + Bc u_0：单独验证第一行块。"""
    di = DoubleIntegrator(DoubleIntegratorParams())
    mpc = LinearMPC(di, MPCParams(horizon=4))
    x0 = np.array([1.0, 2.0, 0.3, -0.2])
    U = np.zeros((4, 2))
    U[0] = [0.7, -0.4]
    Xall = mpc.F @ x0 + mpc.G @ U.reshape(-1)
    x1_mat = Xall[4:8]        # 第 1 个状态块（x_0 占 [0:4]）
    x1_step = di.Ac @ x0 + di.Bc @ U[0]
    assert np.abs(x1_mat - x1_step).max() < 1e-12


# --------------------------------------------------------------------------- #
# 3.2 等待执行
# --------------------------------------------------------------------------- #
def test_dwell_starts_only_after_full_stop():
    """计划等待必须在真正停稳后才开始计时。"""
    wp = _line(n=32, length=6.0)
    w = np.zeros(32)
    w[20] = 2.0                       # 航点 20 停 2 时隙 = 0.8 s
    tp = TimeParameterizedPath(wp, w, TimingParams())
    di = DoubleIntegrator(DoubleIntegratorParams())
    ex = SingleRobotExecutor(tp, di, MPCParams(horizon=15), dt_ctrl=0.1)
    ex.reset(np.array([wp[0, 0], wp[0, 1], 0.0, 0.0]))
    while ex.t < 120 and not ex.done:
        ex.step()
    assert ex.done
    assert len(ex.rec.dwell_done) == 1
    k, _, held = ex.rec.dwell_done[0]
    assert k == 20
    assert held == pytest.approx(0.8, abs=0.15), f"实测等待 {held:.3f}s，计划 0.8s"
    assert ex.actual_wait_time == pytest.approx(held, abs=0.25)
    # 停稳时刻速度接近 0（由 V_STOP 阈值保证）
    assert ex.rec.brake_stop, "没有记录到停稳时刻"


def test_no_double_counting_of_wait():
    """制动时间不得被重复计入等待时长。"""
    wp = _line(n=32, length=6.0)
    w = np.zeros(32)
    w[20] = 2.0
    tp = TimeParameterizedPath(wp, w, TimingParams())
    di = DoubleIntegrator(DoubleIntegratorParams())
    ex = SingleRobotExecutor(tp, di, MPCParams(horizon=15), dt_ctrl=0.1)
    ex.reset(np.array([wp[0, 0], wp[0, 1], 0.0, 0.0]))
    while ex.t < 120 and not ex.done:
        ex.step()
    assert ex.actual_wait_time <= ex.t
    assert ex.actual_wait_time + ex.extra_brake_time <= ex.t + 1e-6


# --------------------------------------------------------------------------- #
# 3.3 逐对碰撞统计
# --------------------------------------------------------------------------- #
def test_per_pair_collision_stats():
    """N=4 必须分别记录 6 对，而不是只记全队第一次。"""
    t = np.linspace(0, 1, 32)
    a = np.stack([1.0 + 8.0 * t, 5.0 * np.ones_like(t)], axis=-1)
    b = np.stack([5.0 * np.ones_like(t), 1.0 + 8.0 * t], axis=-1)
    c = a.copy()                       # 与 a 轨迹重合
    d = np.stack([5.0 * np.ones_like(t), 5.0 + 0.3 * t], axis=-1)
    paths = np.stack([a, b, c, d])
    r = simulate_mpc(paths, np.zeros((4, 31)), 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.pair_collision.shape == (6,)
    assert r.pair_first_time.shape == (6,)
    assert r.pair_min_distance.shape == (6,)
    assert r.pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert r.pair_collision[1], "机器人 0 与 2 轨迹重合，该对应被判为碰撞"


# --------------------------------------------------------------------------- #
# 3.4 跟踪误差
# --------------------------------------------------------------------------- #
def test_tracking_metrics_recorded_without_full_trace():
    """record=False 也必须给出真实跟踪误差，不能显示为 0。"""
    paths = _two_robot_paths()
    w = np.zeros((2, 31))
    w[1, 0] = 12.0
    r = simulate_mpc(paths, w, 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.trajectories is None
    assert r.tracking_rmse > 0.0
    assert r.tracking_max >= r.tracking_rmse


# --------------------------------------------------------------------------- #
# 3.5 暂停事件
# --------------------------------------------------------------------------- #
def test_pause_events_are_method_independent():
    """同一 (seed, p) 下事件序列必须完全一致，保证不同方法用同一串扰动。"""
    cfg = MPCSimConfig()
    e1 = make_pause_events(123, 0.2, 4, 96, cfg)
    e2 = make_pause_events(123, 0.2, 4, 96, cfg)
    assert e1 == e2
    assert e1 != make_pause_events(124, 0.2, 4, 96, cfg)
    assert make_pause_events(123, 0.0, 4, 96, cfg) == []


def test_segment_times_available_for_calibration():
    """M1/M2 的标定依赖每段实测通行时间。"""
    paths = _two_robot_paths()
    w = np.zeros((2, 31))
    w[1, 0] = 12.0
    r = simulate_mpc(paths, w, 0.0, MPCSimConfig(), seed=0, record=False)
    assert r.segment_times is not None and len(r.segment_times) == 2
    for robot_segs in r.segment_times:
        assert len(robot_segs) == 31
        assert "dt" in robot_segs[0] and "length" in robot_segs[0]
