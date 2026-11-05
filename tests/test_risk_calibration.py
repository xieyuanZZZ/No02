"""阶段二 §2.1 的定向回归：时长语义、M2 抽样索引、无泄漏预测入口。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fhd.execution.progress import release_times, release_times_with_durations
from fhd.risk.baselines_m0m1m2 import predict_m0m1m2
from fhd.risk.duration_model import DurationModel, segment_features
from fhd.risk.multi_pair import FirstHitRisk


def _static_inputs(n=2, h=12, seed=0):
    rng = np.random.default_rng(seed)
    paths = np.zeros((n, h, 2))
    for i in range(n):
        t = np.linspace(0, 1, h)
        paths[i, :, 0] = 1.0 + 5.0 * t
        paths[i, :, 1] = 1.0 + i * 2.0 + 0.3 * np.sin(3 * t)
    wait = np.zeros((n, h - 1))
    wait[0, 0] = 2.0
    wait[1, 4] = 1.0
    return torch.tensor(paths, dtype=torch.float64), torch.tensor(wait, dtype=torch.float64)


def test_durations_equal_one_is_exactly_m0():
    rng = np.random.default_rng(1)
    wait = torch.tensor(rng.integers(0, 4, size=(2, 7)).astype(np.float64))
    ones = torch.ones_like(wait)
    assert torch.allclose(release_times(wait), release_times_with_durations(wait, ones))


def test_release_times_wait_attribution_matches_discrete_model():
    """tau_m = Σ_{j<m} d_j + Σ_{h<=m} w_h，等待属于当前航点。"""
    wait = torch.tensor([[0.0, 0.0, 3.0, 0.0]])
    dur = torch.tensor([[1.0, 2.0, 1.0, 1.0]])
    tau = release_times_with_durations(wait, dur)
    # m=0: 0 + 0; m=1: 1 + 0; m=2: 1+2 + 3 = 6; m=3: 1+2+1 + 3 = 7
    assert torch.allclose(tau[0], torch.tensor([0.0, 1.0, 6.0, 7.0]))


def test_release_times_rejects_censored_or_negative():
    wait = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        release_times_with_durations(wait, torch.tensor([[1.0, float("nan"), 1.0]]))
    with pytest.raises(ValueError):
        release_times_with_durations(wait, torch.tensor([[1.0, -1.0, 1.0]]))


def _toy_model():
    """pred=0、残差池第 0 行=1、第 1 行=2 的玩具模型。"""
    h1 = 5
    m = DurationModel()
    m.coef = np.zeros(11)
    m.mean = np.zeros(11)
    m.scale = np.ones(11)
    m.residual_pool = [np.full(h1, 1.0), np.full(h1, 2.0)]
    return m


def test_m2_sampling_uses_per_robot_index_not_robot0():
    """旧实现 samp[idx[:,None,None]][:,0,:] 固定拿 robot 0；这里必须各机器人独立。"""
    m = _toy_model()
    feats = np.zeros((2, 5, 11))

    class FakeRng:
        def integers(self, low, high, size=None):
            assert size == (3, 2)
            return np.array([[0, 1], [1, 0], [1, 1]])

    dur = m.sample(feats, FakeRng(), n_draws=3)
    assert dur.shape == (3, 2, 5)
    assert np.allclose(dur[0, 0], 1.0) and np.allclose(dur[0, 1], 2.0)
    assert np.allclose(dur[1, 0], 2.0) and np.allclose(dur[1, 1], 1.0)
    assert np.allclose(dur[2], 2.0)
    # 旧 bug 会让所有机器人都等于 robot0 的抽样值
    assert not np.allclose(dur[0, 0], dur[0, 1])


def test_m2_zero_residual_equals_m1():
    rng = np.random.default_rng(2)
    paths, wait = _static_inputs()
    risk = FirstHitRisk(d_safe=0.35)
    m = DurationModel()
    h1 = paths.shape[1] - 1
    m.coef = np.zeros(11)
    m.mean = np.zeros(11)
    m.scale = np.ones(11)
    m.residual_pool = [np.zeros(h1), np.zeros(h1)]
    out = predict_m0m1m2(risk, paths, wait, torch.full((2,), 0.1, dtype=torch.float64),
                         time_steps=20, duration_model=m, n_draws=5, rng=rng)
    assert np.allclose(out["m2"], out["m1"])
    assert out["samples"].shape == (5, 2, h1)


def test_prediction_entry_does_not_read_execution_truth(monkeypatch):
    """预测入口只接收静态输入；把 simulate_mpc 替换为报错也必须能算。"""
    import fhd.execution.mpc_simulator as sim

    def boom(*a, **k):
        raise AssertionError("预测阶段不得读取测试任务的 MPC 真实执行结果")

    monkeypatch.setattr(sim, "simulate_mpc", boom)
    paths, wait = _static_inputs()
    risk = FirstHitRisk(d_safe=0.35)
    out = predict_m0m1m2(risk, paths, wait, torch.zeros(2, dtype=torch.float64),
                         20, None, n_draws=3)
    assert set(out) >= {"m0", "m1", "m2"}
    assert np.allclose(out["m0"], out["m1"])


def test_segment_features_are_static_and_finite():
    paths, wait = _static_inputs(n=4, h=16)
    feats = segment_features(paths.numpy(), wait.numpy(), dt_slot=0.4)
    assert feats.shape == (4, 15, 11)
    assert np.isfinite(feats).all()
    assert (feats[..., 0] > 0).all()


def test_duration_model_fit_predict_shapes():
    rng = np.random.default_rng(3)
    X = np.abs(rng.normal(size=(6, 10, 11))) + 0.1
    y = 1.0 + 0.2 * X[..., 0] + 0.1 * rng.normal(size=(6, 10))
    y[0, 3] = np.nan                       # 右删失段
    m = DurationModel(alpha=1e-3).fit(X, y)
    assert m.n_train_segments == 59
    pred = m.predict(X)
    assert pred.shape == (6, 10)
    s = m.sample(X, np.random.default_rng(0), 4)
    assert s.shape == (4, 6, 10)
    assert (s >= 0).all()


def test_including_pause_durations_is_rejected():
    risk = FirstHitRisk(d_safe=0.35)
    paths, wait = _static_inputs()
    dur = torch.ones(2, paths.shape[1] - 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        risk(paths, wait, torch.full((2,), 0.1, dtype=torch.float64), 10,
             segment_durations=dur, duration_semantics="includes_pause")
