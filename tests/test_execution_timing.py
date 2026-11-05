"""阶段一 §1.2 的定向回归：等待/暂停/停车/到达事件与时间账本。"""

from __future__ import annotations

import numpy as np
import pytest

from fhd.control.linear_mpc import MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams
from fhd.dynamics.trajectory_timing import (MonotoneProgress, TimeParameterizedPath,
                                            TimingParams)
from fhd.execution.mpc_executor import SingleRobotExecutor
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc

N = 32


def _line(length=8.0, n=N):
    t = np.linspace(0, 1, n)
    return np.stack([1.0 + length * t, 5.0 * np.ones_like(t)], axis=-1)


def _executor(wait=None, length=8.0, horizon=15):
    wp = _line(length)
    w = np.zeros(N) if wait is None else np.asarray(wait, dtype=np.float64)
    tp = TimeParameterizedPath(wp, w, TimingParams())
    di = DoubleIntegrator(DoubleIntegratorParams())
    ex = SingleRobotExecutor(tp, di, MPCParams(horizon=horizon), dt_ctrl=0.1)
    ex.reset(np.array([wp[0, 0], wp[0, 1], 0.0, 0.0]))
    return ex, tp


def _run(ex, tmax=80.0, pause_at=None, pause_dur=0.4):
    while ex.t < tmax and not ex.done:
        req = pause_at is not None and abs(ex.t - pause_at) < 1e-9
        ex.step(pause_request=req, pause_duration=pause_dur)


def test_no_wait_line_ledger_has_no_negative_durations():
    ex, tp = _executor()
    _run(ex)
    assert ex.done
    assert ex.actual_wait_time == pytest.approx(0.0, abs=1e-9)
    assert ex.extra_brake_time == pytest.approx(0.0, abs=1e-9)
    segs = ex.segment_stats()
    assert all(s["valid"] for s in segs)
    dts = [s["dt"] for s in segs]
    assert min(dts) > 0.0
    # 时间守恒：最后到达 = 松驰段行驶时间之和 + 计划等待
    assert ex.finish_time == pytest.approx(sum(dts) + ex.actual_wait_time, abs=0.11)
    # 无等待航点：depart == arrival
    for k in range(1, tp.n_wp - 1):
        assert ex.wp_depart[k] == pytest.approx(ex.wp_arrival[k], abs=1e-9)


def test_start_wait_counts_from_t0_and_departs_after_hold():
    w = np.zeros(N)
    w[0] = 3.0                       # 1.2 s 出发前等待
    ex, tp = _executor(wait=w)
    _run(ex)
    assert ex.done
    assert ex.wp_arrival[0] == 0.0
    assert ex.wp_depart[0] == pytest.approx(1.2, abs=0.11)
    assert ex.actual_wait_time == pytest.approx(1.2, abs=0.11)
    assert not ex.rec.brake_start, "起点等待不需要制动"
    # 第一段从等待结束后才开始计时
    assert ex.segment_stats()[0]["depart_time"] == pytest.approx(1.2, abs=0.11)


def test_mid_waypoint_wait_brake_hold_resume_order_and_no_double_count():
    w = np.zeros(N)
    w[10] = 1.0                      # 0.4 s
    ex, tp = _executor(wait=w)
    _run(ex)
    assert ex.done
    k, t_resume, held = ex.rec.dwell_done[0]
    assert k == 10
    assert held == pytest.approx(0.4, abs=0.11)
    bs = [e for e in ex.rec.events if e["type"] == "brake_start"]
    hs = [e for e in ex.rec.events if e["type"] == "hold_start"]
    rs = [e for e in ex.rec.events if e["type"] == "resume"]
    assert bs and hs and rs
    assert bs[0]["t"] < hs[0]["t"] <= rs[0]["t"]
    assert ex.wp_arrival[10] <= hs[0]["t"] + 1e-9
    assert ex.wp_depart[10] == pytest.approx(rs[0]["t"])
    assert ex.actual_wait_time == pytest.approx(held, abs=0.11)
    assert ex.extra_brake_time > 0.0
    # 制动与等待不得重叠：两者之和不超过总时长
    assert ex.actual_wait_time + ex.extra_brake_time <= ex.t + 1e-9
    # 每段都应是有效账本
    segs = ex.segment_stats()
    assert all(s["valid"] and s["dt"] is not None and s["dt"] > 0 for s in segs)


def test_two_consecutive_waits_keep_order():
    w = np.zeros(N)
    w[10] = 1.0
    w[11] = 1.0
    ex, tp = _executor(wait=w)
    _run(ex)
    assert ex.done
    done = [d[0] for d in ex.rec.dwell_done]
    assert done == [10, 11]
    assert ex.rec.dwell_done[0][1] < ex.rec.dwell_done[1][1]
    assert ex.actual_wait_time == pytest.approx(0.8, abs=0.15)


def test_pause_request_during_braking_is_rejected_and_logged():
    w = np.zeros(N)
    w[12] = 2.0
    ex, tp = _executor(wait=w)
    while ex.mode != "BRAKE_DWELL" and ex.t < 40:
        ex.step()
    assert ex.mode == "BRAKE_DWELL"
    n_before = ex.pause_rejected
    ok = ex.request_pause(0.4)
    assert ok is False
    assert ex.pause_rejected == n_before + 1
    assert ex.mode == "BRAKE_DWELL"
    assert ex.rec.pause_rejected


def test_pause_duration_starts_after_full_stop():
    ex, tp = _executor()
    _run(ex, pause_at=2.0, pause_dur=1.0)
    assert ex.done
    assert ex.pause_count == 1
    bs = [e for e in ex.rec.events if e["type"] == "brake_stop" and e.get("kind") == "pause"]
    rs = [e for e in ex.rec.events if e["type"] == "resume" and e.get("kind") == "pause"]
    assert bs and rs
    assert rs[0]["t"] - bs[0]["t"] == pytest.approx(1.0, abs=0.11)
    # 暂停请求发生在 t=2.0，但保持从停稳后才开始，故 resume 明显晚于 3.0
    assert rs[0]["t"] > 3.0
    assert ex.random_hold_time == pytest.approx(1.0, abs=0.11)


def test_pause_after_dwell_clears_stale_dwell_index():
    """一次 dwell 后请求暂停，不得复用旧的 _dwell_idx。"""
    w = np.zeros(N)
    w[8] = 1.0
    ex, tp = _executor(wait=w)
    # 先完成一次 dwell
    while len(ex.rec.dwell_done) == 0 and ex.t < 40:
        ex.step()
    assert ex._dwell_idx is None
    ok = ex.request_pause(0.4)
    assert ok
    assert ex._dwell_idx is None
    assert ex._hold_kind == "pause"
    _run(ex, tmax=80)
    assert ex.done
    assert ex.pause_count == 1


def test_multi_waypoint_crossing_in_one_step_records_all():
    ex, tp = _executor()
    ex._record_arrivals_crossing(0.0, 0.0, 3.0)
    arrived = [k for k in range(tp.n_wp) if np.isfinite(ex.wp_arrival[k]) and k > 0]
    assert len(arrived) >= 10, f"一次跨越应记录多个航点，实际 {arrived}"
    times = [ex.wp_arrival[k] for k in arrived]
    assert all(0.0 <= t <= ex.dt + 1e-9 for t in times)
    assert times == sorted(times)


def test_corner_projection_stays_monotone():
    """U 形回程贴近去程时，全局最近点会倒跳，邻域跟踪必须保持单调。"""
    out = np.stack([np.linspace(0, 5, 16), np.zeros(16)], axis=-1)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 8)
    turn = np.stack([5 + 0.05 * np.cos(theta), 0.05 + 0.05 * np.sin(theta)], axis=-1)
    back = np.stack([np.linspace(5, 0, 8), 0.1 * np.ones(8)], axis=-1)
    wp = np.vstack([out, turn, back])
    tp = TimeParameterizedPath(wp, np.zeros(len(wp)), TimingParams())
    prog = MonotoneProgress(tp)
    # 已经走到回程中段（离终点 3 m）
    s_far = tp.length - 3.0
    prog.s = s_far
    s_new, lat = prog.update(np.array([3.0, 0.05]))
    assert s_new >= s_far, "进度不得因近邻平行段而倒跳"
    assert s_new > 6.0, f"回程应投影到后半段，实际 s={s_new:.3f}"
    assert lat < 0.1


def test_timeout_is_censored_not_reported_as_completion():
    wp = _line(length=8.0)
    tp = TimeParameterizedPath(wp, np.zeros(N), TimingParams())
    di = DoubleIntegrator(DoubleIntegratorParams())
    ex = SingleRobotExecutor(tp, di, MPCParams(horizon=15), dt_ctrl=0.1)
    ex.reset(np.array([wp[0, 0], wp[0, 1], 0.0, 0.0]))
    horizon = 4 * 0.4
    for _ in range(4):
        ex.step()
    assert not ex.done
    ex.mark_censored(horizon)
    assert ex.time_censored is True
    segs = ex.segment_stats()
    assert any(not s["valid"] for s in segs)
    assert all(s["dt"] is None for s in segs if not s["valid"])


def test_simulator_marks_timeout_and_censoring():
    paths = np.stack([_line(), _line()])
    w = np.zeros((2, N - 1))
    r = simulate_mpc(paths, w, 0.0,
                     MPCSimConfig(horizon_steps=3), seed=0, record=False)
    assert r.all_arrived is False
    assert r.timeout is True
    assert r.time_censored is True
    assert r.completion_time == pytest.approx(3 * 0.4)


def test_rejected_pause_when_done():
    ex, tp = _executor(length=2.0)
    _run(ex)
    assert ex.done
    assert ex.request_pause(0.4) is False
    assert ex.pause_rejected == 1
