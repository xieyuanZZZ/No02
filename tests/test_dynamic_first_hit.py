"""阶段二 §2.3/§2.5 E1：M3 状态模型、首次碰撞与采样接口的数学/边界测试。"""

from __future__ import annotations

import numpy as np
import pytest

from fhd.risk.conditional_execution_model import (
    ConditionalExecutionModel, ExecutionModelParams, PathGeometry,
    compute_pair_first_hits, fit_execution_model,
)
from fhd.risk.dynamic_first_hit import deterministic_first_hit, predict_pair_risks


def _line(n=24, length=6.0, y=1.0):
    t = np.linspace(0, 1, n)
    return np.stack([1.0 + length * t, y * np.ones_like(t)], axis=-1)


def _params(**kw):
    base = dict(a_acc=2.0, a_brake=2.0, a_restart=1.5, restart_time=0.2,
                v_scale=1.0, pause_durations=[0.4], dt=0.1, dt_slot=0.4)
    base.update(kw)
    return ExecutionModelParams(**base)


def test_deterministic_first_hit_matches_analytic():
    # 两条直线在 t=0.5 交换位置，最小距离 0
    k = 11
    t = np.linspace(0, 1, k)
    a = np.stack([t, np.zeros_like(t)], axis=-1)
    b = np.stack([1.0 - t, np.zeros_like(t)], axis=-1)
    pos = np.stack([a, b], axis=1)
    out = deterministic_first_hit(pos, t, d_safe=0.35)
    assert out["collision"]
    # |1-2s| = 0.35 -> s = 0.325
    assert out["pair_first_time"][0] == pytest.approx(0.325, abs=1e-9)


def test_deterministic_first_hit_no_crossing():
    t = np.linspace(0, 1, 11)
    a = np.stack([t, np.zeros_like(t)], axis=-1)
    b = np.stack([t, 2.0 * np.ones_like(t)], axis=-1)
    out = deterministic_first_hit(np.stack([a, b], axis=1), t, d_safe=0.35)
    assert not out["collision"]
    assert out["pair_first_time"][0] == float("inf")


def test_m3_zero_pause_is_deterministic():
    paths = np.stack([_line(), _line(y=5.0)])
    wait = np.zeros((2, 23))
    model = ConditionalExecutionModel(_params())
    out = model.rollout(paths, wait, 0.0, 10.0, n_samples=8, seed=3)
    assert out["pause_starts"] == 0
    assert np.allclose(out["positions"][:, 0], out["positions"][:, 1])


def test_m3_start_wait_delays_departure():
    paths = np.stack([_line(), _line(y=5.0)])
    wait = np.zeros((2, 23))
    wait[1, 0] = 5.0
    model = ConditionalExecutionModel(_params())
    out = model.rollout(paths, wait, 0.0, 10.0, n_samples=4, seed=1)
    s0 = out["s_hist"][:, 0, 0]
    s1 = out["s_hist"][:, 0, 1]
    # 第二台在前 2 s 应基本停住，第一台已明显前进
    idx = int(2.0 / 0.1)
    assert s1[idx] < 0.05
    assert s0[idx] > s1[idx]


def test_m3_coincident_paths_risk_one_and_team_not_sum():
    paths = np.stack([_line(), _line(), _line()])
    wait = np.zeros((3, 23))
    model = ConditionalExecutionModel(_params())
    out = model.predict(paths, wait, 0.0, 8.0, d_safe=0.35, n_samples=20, seed=2)
    assert np.allclose(out["pair_risks"], 1.0)
    assert out["team_collision_estimate"] == 1.0
    assert out["team_collision_estimate"] <= np.sum(out["pair_risks"])


def test_m3_separated_paths_risk_zero():
    paths = np.stack([_line(), _line(y=5.0)])
    wait = np.zeros((2, 23))
    model = ConditionalExecutionModel(_params())
    out = model.predict(paths, wait, 0.0, 8.0, d_safe=0.35, n_samples=20, seed=2)
    assert out["pair_risks"][0] == 0.0
    assert out["team_collision_estimate"] == 0.0


def test_m3_reproducible_with_same_seed():
    paths = np.stack([_line(), _line(y=5.0)])
    wait = np.zeros((2, 23))
    model = ConditionalExecutionModel(_params())
    a = model.predict(paths, wait, 0.2, 8.0, d_safe=0.35, n_samples=30, seed=7)
    b = model.predict(paths, wait, 0.2, 8.0, d_safe=0.35, n_samples=30, seed=7)
    assert np.array_equal(a["pair_risks"], b["pair_risks"])
    assert a["team_first_hit_times"] == b["team_first_hit_times"]


def test_m3_pause_changes_trajectory():
    paths = np.stack([_line(), _line(y=5.0)])
    wait = np.zeros((2, 23))
    model = ConditionalExecutionModel(_params())
    base = model.rollout(paths, wait, 0.0, 8.0, n_samples=50, seed=1)
    paused = model.rollout(paths, wait, 0.3, 8.0, n_samples=50, seed=1)
    assert paused["pause_starts"] > 0
    # 至少有一个样本的位置与无暂停情形不同
    assert not np.allclose(base["positions"], paused["positions"])


def test_m3_interface_keys_and_ranges():
    paths = np.stack([_line(), _line(y=5.0)])
    wait = np.zeros((2, 23))
    out = predict_pair_risks(paths, wait, 0.1, 8.0, _params(), d_safe=0.35,
                             n_samples=20, seed=3)
    assert set(out) >= {"pair_index", "pair_risks", "first_hit_time_distribution",
                        "team_collision_estimate", "runtime_ms", "model_info"}
    assert np.all((out["pair_risks"] >= 0) & (out["pair_risks"] <= 1))
    assert 0 <= out["team_collision_estimate"] <= 1
    assert out["model_info"]["is_exact_probability"] is False


def test_m3_ablation_flags_run():
    paths = np.stack([_line(), _line(y=5.0)])
    wait = np.zeros((2, 23))
    wait[0, 8] = 2.0
    model = ConditionalExecutionModel(_params())
    for kwargs in ({"use_braking": False}, {"use_modes": False},
                   {"use_speed": False}):
        out = model.predict(paths, wait, 0.0, 8.0, d_safe=0.35, n_samples=10,
                            seed=1, **kwargs)
        assert np.all(np.isfinite(out["pair_risks"]))


def test_m3_prediction_does_not_read_execution_truth(monkeypatch):
    import fhd.execution.mpc_simulator as sim

    def boom(*a, **k):
        raise AssertionError("M3 预测阶段不得读取测试任务的 MPC 真值")

    monkeypatch.setattr(sim, "simulate_mpc", boom)
    paths = np.stack([_line(), _line(y=5.0)])
    wait = np.zeros((2, 23))
    out = predict_pair_risks(paths, wait, 0.1, 8.0, _params(), d_safe=0.35,
                             n_samples=10, seed=1)
    assert np.all(np.isfinite(out["pair_risks"]))


def test_params_roundtrip(tmp_path):
    p = _params(v_scale=0.77, pause_durations=[0.4, 0.8])
    p.save(tmp_path / "p.json")
    q = ExecutionModelParams.load(tmp_path / "p.json")
    assert q.v_scale == pytest.approx(0.77)
    assert q.pause_durations == pytest.approx([0.4, 0.8])


def test_path_geometry_next_stop_skips_current_point():
    """刚离开等待点时 s 仍等于该航点，不能立刻再次选择它。"""
    wp = _line(n=11, length=5.0)
    wait = np.zeros((1, 10))
    wait[0, 0] = 2.0
    geo = PathGeometry(np.stack([wp]), wait)
    s = np.zeros((3, 1))
    target, idx = geo.next_stop(s)
    assert idx[0, 0] == -1, "s=0 时起点等待不应再被选为下一个停车点"
    assert target[0, 0] == pytest.approx(geo.length[0])


def test_fit_execution_model_recovers_plausible_params():
    dt = 0.1
    T, N = 40, 1
    s = np.zeros((T, N))
    v = np.zeros((T, N))
    modes = np.zeros((T, N), dtype=np.int64)
    for t in range(1, T):
        v[t] = min(v[t - 1] + 2.0 * dt, 1.5)
        s[t] = s[t - 1] + v[t] * dt
    vp = np.full((T, N), 1.5)
    traces = [{"s": s, "v": v, "v_plan": vp, "modes": modes, "T_valid": T}]
    events = [[[]]]
    p = fit_execution_model(traces, events, dt=dt)
    assert 1.0 <= p.a_acc <= 2.5
    assert p.v_scale > 0.5
