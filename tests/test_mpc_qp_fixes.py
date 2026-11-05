"""阶段一 §1.1 的定向回归：增量代价、QP 与显式目标一致、求解统计与安全回退。"""

from __future__ import annotations

import numpy as np
import pytest

from fhd.control.linear_mpc import LinearMPC, MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams


def _mpc(horizon=8, **kw):
    di = DoubleIntegrator(DoubleIntegratorParams())
    p = MPCParams(horizon=horizon, **kw)
    return LinearMPC(di, p), di


def _random_problem(seed=0):
    rng = np.random.default_rng(seed)
    mpc, di = _mpc()
    x0 = np.array([0.4, -0.2, 0.5, 0.1])
    Np = mpc.Np
    U = rng.normal(scale=0.3, size=(Np, 2))
    x_ref = np.zeros((Np + 1, 4))
    x_ref[:, :2] = np.linspace([0.4, -0.2], [2.0, 1.0], Np + 1)
    x_ref[:, 2:] = np.linspace([0.5, 0.1], [0.0, 0.0], Np + 1)
    last = np.array([0.2, -0.4])
    mpc._last_u = last.copy()
    return mpc, x0, U, x_ref, last


def test_qp_objective_matches_explicit_cost():
    """0.5 U'HU + f'U 必须等于显式目标减去常数项（逐项一致）。"""
    mpc, x0, U, x_ref, _ = _random_problem(1)
    H, f, A_c, lo, hi = mpc._assemble_qp(x0, x_ref)
    qp = 0.5 * U.reshape(-1) @ H @ U.reshape(-1) + f @ U.reshape(-1)
    n = mpc.Np
    bias = mpc.F @ x0
    Q, R = mpc._weights()
    d_prev = np.zeros(2 * n)
    d_prev[:2] = mpc._last_u
    const = (bias - x_ref.reshape(-1)) @ Q @ (bias - x_ref.reshape(-1)) \
        + d_prev @ mpc.S_delta @ d_prev
    # H 上有 1e-8 的正则项，允许 ~1e-8*||U||² 的差异
    assert qp + const == pytest.approx(mpc.explicit_cost(U, x0, x_ref), abs=1e-6)


def test_qp_gradient_matches_explicit_cost_finite_difference():
    mpc, x0, U, x_ref, _ = _random_problem(2)
    H, f, _, _, _ = mpc._assemble_qp(x0, x_ref)
    grad = H @ U.reshape(-1) + f
    eps = 1e-6
    fd = np.zeros_like(grad)
    for k in range(grad.size):
        Up = U.copy().reshape(-1)
        Um = U.copy().reshape(-1)
        Up[k] += eps
        Um[k] -= eps
        fd[k] = (mpc.explicit_cost(Up.reshape(-1, 2), x0, x_ref)
                 - mpc.explicit_cost(Um.reshape(-1, 2), x0, x_ref)) / (2 * eps)
    assert np.abs(grad - fd).max() < 1e-5


def test_first_increment_penalty_is_separate_from_adjacent_differences():
    """u_{-1} 只作用于第一个控制；相邻差分不应再减 u_{-1}。

    构造 U 使所有相邻差分恰好为 0：旧实现会得到 (Np-1)*λ||u0-u_prev||²
    的错误惩罚，新实现只保留第一项。
    """
    mpc, x0, _, _, last = _random_problem(3)
    U = np.tile(np.array([0.3, -0.2]), (mpc.Np, 1))
    Q, R = mpc._weights()
    bias = mpc.F @ x0
    # 只关心 Δu 项：把跟踪权重置零后直接算显式代价里的平滑项
    x_ref = (bias + mpc.G @ U.reshape(-1)).reshape(mpc.Np + 1, 4)
    J = mpc.explicit_cost(U, x0, x_ref)
    du0 = U[0] - last
    expected = mpc.p.s_du * float(du0 @ du0)
    # 跟踪项为 0、控制项为 R；把 R 的贡献补回来再比较
    u_vec = U.reshape(-1)
    control = float(u_vec @ R @ u_vec)
    assert J - control == pytest.approx(expected, abs=1e-9)

    H, f, _, _, _ = mpc._assemble_qp(x0, x_ref)
    qp = 0.5 * U.reshape(-1) @ H @ U.reshape(-1) + f @ U.reshape(-1)
    d_prev = np.zeros(2 * mpc.Np)
    d_prev[:2] = last
    const = (bias - x_ref.reshape(-1)) @ Q @ (bias - x_ref.reshape(-1)) \
        + d_prev @ mpc.S_delta @ d_prev
    assert qp + const == pytest.approx(J, abs=1e-6)


def test_solver_stats_aggregate_all_calls():
    mpc, di = _mpc(horizon=6)
    x0 = np.zeros(4)
    ref = np.zeros((7, 4))
    ref[:, 0] = np.linspace(0, 0.6, 7)
    for _ in range(4):
        mpc.solve(x0, ref)
    assert mpc.stats.n_solves == 4
    assert len(mpc.stats.solve_times) == 4
    s = mpc.stats.summary()
    assert s["n_solves"] == 4
    assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"] + 1e-12


def test_solver_cache_rebuilt_only_when_matrices_change():
    mpc, di = _mpc(horizon=6)
    x0 = np.zeros(4)
    ref = np.zeros((7, 4))
    mpc.solve(x0, ref)
    first_solver = mpc._solver
    mpc.solve(x0, ref)
    assert mpc._solver is first_solver, "P/A 不变时应复用 solver（只 update q/l/u）"
    mpc.p.s_du = 0.9                      # 改变权重 -> H 变 -> 必须重建
    mpc.solve(x0, ref)
    assert mpc._solver is not first_solver


def test_failed_solve_uses_feasible_braking_fallback(monkeypatch):
    mpc, di = _mpc(horizon=6)
    x0 = np.array([0.0, 0.0, 1.2, 0.0])
    ref = np.zeros((7, 4))
    ref[:, 0] = np.linspace(0.0, 0.6, 7)

    def boom(*a, **k):
        raise RuntimeError("solver down")

    monkeypatch.setattr(mpc, "_ensure_solver", boom)
    U, info = mpc.solve(x0, ref)
    assert not info["ok"] and info["fallback"] == "brake"
    assert info["usable"] is True
    assert info["u"][0] < 0.0, "速度朝 +x，制动应朝 -x"
    assert np.abs(info["u"]).sum() <= mpc.p.a_max + 1e-9
    assert np.linalg.norm(x0[2:] + mpc.di.p.dt_ctrl * info["u"]) <= np.linalg.norm(x0[2:]) + 1e-9
    assert mpc.stats.n_failed == 1 and mpc.stats.n_fallback == 1


def test_infeasible_fallback_is_reported_not_silently_fixed(monkeypatch):
    """初速超过 v_max 且制动一拍内停不下来：必须显式 usable=False。"""
    mpc, di = _mpc(horizon=6)
    x0 = np.array([0.0, 0.0, 0.0, 3.0])       # 远超 v_max=1.5
    ref = np.zeros((7, 4))

    def boom(*a, **k):
        raise RuntimeError("solver down")

    monkeypatch.setattr(mpc, "_ensure_solver", boom)
    U, info = mpc.solve(x0, ref)
    assert not info["ok"]
    assert info["usable"] is False
    assert mpc.stats.n_fallback_infeasible == 1


def test_stopping_feasibility_helper():
    assert LinearMPC.stopping_feasible(1.5, 1.0, 2.0)
    assert not LinearMPC.stopping_feasible(1.5, 0.3, 2.0)


def test_solve_does_not_clip_to_velocity_limit_in_postprocessing():
    """速度界由 QP 约束保证；求解器返回的 u 不应被事后裁剪掩盖。"""
    mpc, di = _mpc(horizon=6)
    x0 = np.array([0.0, 0.0, 1.4, 0.0])
    ref = np.zeros((7, 4))
    ref[:, 0] = np.linspace(0.0, 0.2, 7)
    U, info = mpc.solve(x0, ref)
    assert info["ok"]
    assert info["u"][0] <= 0.0 + 1e-6       # 需要减速而不是继续加速
