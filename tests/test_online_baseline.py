"""修复后的简单在线冲突预测：ETA/让行/停车点/前瞻/计划等待恢复的定向测试。"""

from __future__ import annotations

import numpy as np
import pytest

from fhd.execution.coordinator import Observation, RobotObservation
from fhd.optim.online_baseline import SimpleConflictCoordinator


def _line(length=8.0, y=5.0, n=32):
    t = np.linspace(0, 1, n)
    return np.stack([1.0 + length * t, y * np.ones_like(t)], axis=-1)


def _cross():
    t = np.linspace(0, 1, 32)
    a = np.stack([1.0 + 8.0 * t, 5.0 * np.ones_like(t)], axis=-1)
    b = np.stack([5.0 * np.ones_like(t), 1.0 + 8.0 * t], axis=-1)
    return np.stack([a, b])


def _rb(i, s, speed, path, candidates=None):
    p = _pos(path, s)
    return RobotObservation(
        robot_id=i, x=float(p[0]), y=float(p[1]), vx=float(speed), vy=0.0,
        speed=float(speed), s=float(s), mode="NORMAL", hold_remaining=0.0,
        hold_source=None, online_wait_time=0.0, last_online_stop_t=-1e9,
        length=float(_length(path)), stop_candidates=list(candidates or []))


def _length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum())


def _pos(path, s):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = float(np.clip(s, 0, cum[-1]))
    k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    frac = (s - cum[k]) / max(seg[k], 1e-12)
    return path[k] + frac * (path[k + 1] - path[k])


def _obs(paths, robots, t=0.0, pause_p=0.1):
    return Observation(t=t, robots=robots, d_safe=0.35, dt_slot=0.4,
                       horizon_s=3.0, pause_p=pause_p, config={})


def _cands(path, s_now, s_conf, n=4):
    out = []
    seg = np.linspace(s_now + 0.4, s_conf - 0.15, n)
    for k, s in enumerate(seg):
        if s <= s_now:
            continue
        out.append({"wp": k + 1, "s": float(s), "dist": float(s - s_now),
                    "brake_distance": 0.0, "lead_distance": 0.0,
                    "available": float(s - s_now)})
    return out


def test_yielder_uses_later_eta_not_robot_id():
    paths = _cross()
    c = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0, {})
    i, j, rule = c._yielder({"pair": (0, 1), "eta_i": 1.0, "eta_j": 2.5})
    assert (i, j) == (1, 0) and rule.startswith("later_eta")
    # 交换编号但 ETA 关系不变：后到者仍然让行
    i, j, rule = c._yielder({"pair": (1, 0), "eta_i": 1.0, "eta_j": 2.5})
    assert i == 0 and j == 1


def test_tie_uses_only_robot_id():
    paths = _cross()
    c = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0, {})
    i, j, rule = c._yielder({"pair": (0, 1), "eta_i": 1.0, "eta_j": 1.02})
    assert (i, j) == (1, 0) and rule.startswith("tie_eta")
    i, j, _ = c._yielder({"pair": (2, 1), "eta_i": 1.0, "eta_j": 1.02})
    assert (i, j) == (2, 1)


def test_conflict_locations_are_per_path_not_midpoint():
    paths = _cross()
    # 0 号先到交叉区，1 号后到；s_conf 应分别来自各自路径
    r0 = _rb(0, 3.2, 1.0, paths[0], _cands(paths[0], 3.2, 4.0))
    r1 = _rb(1, 3.5, 0.5, paths[1], _cands(paths[1], 3.5, 4.0))
    co = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0, {})
    times, prog, pos, conflicts = co._conflict_scan(_obs(paths, [r0, r1]))
    assert conflicts
    c = conflicts[0]
    assert c["s_conf_i"] > r0.s - 1e-6
    assert c["s_conf_j"] > r1.s - 1e-6
    assert c["eta_i"] != c["eta_j"] or abs(c["eta_i"] - c["eta_j"]) < 1e-9
    # 两者位置不同，s_conf 不是同一个几何中点
    assert abs(c["s_conf_i"] - c["s_conf_j"]) > 1e-6


def test_no_unnecessary_stop_for_parallel_safe_paths():
    paths = np.stack([_line(y=1.0), _line(y=8.0)])
    r0 = _rb(0, 2.0, 0.8, paths[0])
    r1 = _rb(1, 2.0, 0.8, paths[1])
    co = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0, {})
    assert co.decide(_obs(paths, [r0, r1])) == []


def test_planned_wait_recovers_in_prediction():
    paths = np.stack([_line(), _line(y=8.0)])
    waits = np.zeros((2, 31))
    waits[0, 10] = 1.0                 # 0.4 s
    co = SimpleConflictCoordinator(paths, waits, 0.35, 0.4, 6.0, 0.0, {})
    r0 = _rb(0, 1.5, 1.0, paths[0])
    r1 = _rb(1, 2.0, 1.0, paths[1])
    times = np.arange(0, 5.01, 0.1)
    s = co.future_progress(_obs(paths, [r0, r1]), 0, times)
    stop_s = co.s_wp[0, 10]
    hold_region = np.abs(s - stop_s) < 1e-9
    assert hold_region.any(), "预测应在计划等待点停住"
    # 等待结束后必须继续前进，而不是一直静止
    assert s[-1] > stop_s + 0.5


def test_candidate_before_conflict_and_behind_stop_is_required():
    paths = _cross()
    # 后到的 1 号只有冲突区之后的候选点 -> 不得发出命令
    bad = [{"wp": 20, "s": 6.0, "dist": 4.0, "brake_distance": 0.0,
            "lead_distance": 0.0, "available": 4.0}]
    r0 = _rb(0, 3.0, 1.0, paths[0], _cands(paths[0], 3.0, 4.0))
    r1 = _rb(1, 1.0, 0.5, paths[1], bad)
    co = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0,
                                   {"lookahead_m": 10.0})
    actions = co.decide(_obs(paths, [r0, r1]))
    # 首选 1 号没有冲突前候选，应尝试 0 号；不能直接接受越界候选
    assert all(a.s_target < 4.0 for a in actions)


def test_both_no_candidate_records_no_solution():
    paths = _cross()
    r0 = _rb(0, 3.5, 1.0, paths[0], [])
    r1 = _rb(1, 3.7, 0.8, paths[1], [])
    co = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0, {})
    assert co.decide(_obs(paths, [r0, r1])) == []
    assert co.last_diagnostics["no_feasible_stop"] >= 1


def test_p0_keeps_offline_plan_without_stop():
    paths = _cross()
    r0 = _rb(0, 3.2, 1.0, paths[0], _cands(paths[0], 3.2, 4.0))
    r1 = _rb(1, 3.5, 0.5, paths[1], _cands(paths[1], 3.5, 4.0))
    co = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0, {})
    assert co.decide(_obs(paths, [r0, r1], pause_p=0.0)) == []
    assert co.last_diagnostics.get("p0_no_action") is True


def test_same_input_is_deterministic():
    paths = _cross()
    r0 = _rb(0, 3.0, 1.0, paths[0], _cands(paths[0], 3.0, 4.0))
    r1 = _rb(1, 1.0, 0.5, paths[1], _cands(paths[1], 1.0, 4.0))
    co1 = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0, {})
    co2 = SimpleConflictCoordinator(paths, np.zeros((2, 31)), 0.35, 0.4, 3.0, 0.0, {})
    a1 = co1.decide(_obs(paths, [r0, r1]))
    a2 = co2.decide(_obs(paths, [r0, r1]))
    assert [(a.robot, a.s_target, a.duration_s) for a in a1] == \
           [(a.robot, a.s_target, a.duration_s) for a in a2]
