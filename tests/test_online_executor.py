"""在线协调所需的执行器接口测试：目标停车点、来源分离、候选点与拒绝原因。"""

from __future__ import annotations

import numpy as np
import pytest

from fhd.control.linear_mpc import MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams
from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams
from fhd.execution.mpc_executor import SingleRobotExecutor

N = 32


def _line(length=8.0, y=5.0):
    t = np.linspace(0, 1, N)
    return np.stack([1.0 + length * t, y * np.ones_like(t)], axis=-1)


def _ex(wait=None, v0=(0.0, 0.0)):
    wp = _line()
    w = np.zeros(N) if wait is None else np.asarray(wait, dtype=np.float64)
    tp = TimeParameterizedPath(wp, w, TimingParams())
    di = DoubleIntegrator(DoubleIntegratorParams())
    ex = SingleRobotExecutor(tp, di, MPCParams(horizon=15), dt_ctrl=0.1)
    ex.reset(np.array([wp[0, 0], wp[0, 1], v0[0], v0[1]]))
    return ex


def test_request_stop_at_feasible_target_is_accepted():
    ex = _ex()
    # 让机器人先动起来
    for _ in range(8):
        ex.step()
    s_now = ex.s_act
    target = min(s_now + 1.0, ex.path.length)
    ok = ex.request_stop_at(target, 0.4, source="online", reason="unit")
    assert ok
    assert ex.mode == "BRAKE_PAUSE"
    assert ex.target_s == pytest.approx(target)
    assert ex.pause_source_counts["online"] == 1
    assert ex.online_pause_count == 1


def test_high_speed_short_target_is_rejected_cannot_stop():
    ex = _ex(v0=(1.5, 0.0))
    s_now = ex.s_act
    ok = ex.request_stop_at(s_now + 0.2, 0.4, source="online")
    assert ok is False
    assert ex.last_stop_reject_reason == "cannot_stop"
    assert ex.pause_reject_reasons["cannot_stop"] == 1
    assert ex.mode == "NORMAL"


def test_target_behind_or_beyond_path_rejected():
    ex = _ex()
    assert ex.request_stop_at(ex.s_act - 0.1, 0.4) is False
    assert ex.last_stop_reject_reason == "target_behind"
    assert ex.request_stop_at(ex.path.length + 1.0, 0.4) is False
    assert ex.last_stop_reject_reason == "target_beyond_path"


def test_second_source_while_braking_is_rejected_not_overwritten():
    ex = _ex()
    for _ in range(8):
        ex.step()
    target = min(ex.s_act + 1.0, ex.path.length)
    assert ex.request_stop_at(target, 0.4, source="online")
    t_before = ex.target_s
    # 随机暂停不能覆盖在线命令
    assert ex.request_pause(0.4) is False
    assert ex.mode == "BRAKE_PAUSE"
    assert ex.target_s == pytest.approx(t_before)
    assert ex.pause_source_counts["online"] == 1
    assert ex.pause_source_counts["random"] == 0


def test_online_hold_time_is_counted_separately_from_random():
    ex = _ex()
    for _ in range(6):
        ex.step()
    target = min(ex.s_act + 1.0, ex.path.length)
    assert ex.request_stop_at(target, 0.4, source="online")
    while ex.t < 30 and not ex.done:
        ex.step()
        if ex.mode == "NORMAL" and ex.online_wait_time > 0:
            break
    assert ex.online_wait_time == pytest.approx(0.4, abs=0.12)
    assert ex.random_hold_time == 0.0
    assert ex.actual_wait_time == 0.0


def test_request_stop_at_while_hold_rejected():
    ex = _ex(wait=[3.0] + [0.0] * (N - 1))       # 起点等待 1.2 s
    assert ex.mode == "HOLD"
    ok = ex.request_stop_at(min(1.0, ex.path.length), 0.4, source="online")
    assert ok is False
    assert ex.last_stop_reject_reason == "mode_HOLD"


def test_feasible_stop_candidates_skip_dwell_and_goal():
    wait = np.zeros(N)
    wait[10] = 1.0
    ex = _ex(wait=wait)
    # 先推进到起点之后
    for _ in range(8):
        ex.step()
    cands = ex.feasible_stop_candidates(lookahead_m=2.0)
    assert cands
    assert all(c["wp"] < 10 for c in cands)       # 不越过下一个计划等待点
    assert all(c["wp"] != N - 1 for c in cands)   # 不在终点停车
    assert all(c["available"] >= 0.05 for c in cands)


def test_feasible_candidates_empty_when_moving_too_fast_for_lookahead():
    ex = _ex(v0=(1.5, 0.0))
    cands = ex.feasible_stop_candidates(lookahead_m=0.2)
    assert cands == []


def test_random_pause_target_uses_brake_distance_not_instant_freeze():
    ex = _ex(v0=(1.2, 0.0))
    s0 = ex.s_act
    ok = ex.request_pause(0.4)
    assert ok
    assert ex.target_s > s0
    assert ex.target_s - s0 >= 1.2 ** 2 / (2 * 2.0) - 1e-9
