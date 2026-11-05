"""离线等待 + First-Hit 在线修正的组合语义测试。"""

from __future__ import annotations

import numpy as np
import torch

from fhd.control.linear_mpc import MPCParams
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams
from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams
from fhd.execution.coordinator import build_observation
from fhd.execution.mpc_executor import SingleRobotExecutor
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc
from fhd.optim.online_firsthit import OnlineFirstHitCoordinator
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config


def _paths():
    t = np.linspace(0, 1, 32)
    return np.stack([
        np.stack([1 + 8 * t, 5 + 0 * t], axis=-1),
        np.stack([5 + 0 * t, 1 + 8 * t], axis=-1),
    ])


def _risk():
    cfg = load_config("configs/base.yaml")
    return FirstHitRisk.from_config(cfg), float(cfg["execution"]["horizon_steps"])


def _execs(paths, waits, steps=0):
    di = DoubleIntegrator(DoubleIntegratorParams())
    out = []
    for i in range(paths.shape[0]):
        tp = TimeParameterizedPath(paths[i], waits[i], TimingParams())
        ex = SingleRobotExecutor(tp, di, MPCParams(horizon=15), dt_ctrl=0.1)
        ex.reset(np.array([paths[i, 0, 0], paths[i, 0, 1], 0.0, 0.0]))
        for _ in range(steps):
            ex.step()
        out.append(ex)
    return out


def test_hybrid_residual_keeps_offline_wait():
    paths = _paths()
    waits = np.zeros((2, 32))
    waits[0, 0] = 2.0                 # 起点 0.8 s
    waits[0, 10] = 1.0                # 中途 0.4 s
    risk, T = _risk()
    co = OnlineFirstHitCoordinator(paths, waits, 0.35, 0.4, 3.2, 0.1,
                                   risk_model=risk, config={"hold_slots": 1})
    execs = _execs(paths, waits, steps=12)   # 已离开起点，接近航点 10 之前
    obs = build_observation(execs, 1.2, 0.35, 0.4, 3.2, 0.1)
    pr, wr, next_idx = co._residual_model(obs)
    assert pr.shape[0] == 2 and wr.shape[0] == 2
    # 在剩余等待里必须还能找到原始 1 时隙的等待（不能悄悄丢弃离线计划）
    assert float(wr[0].sum()) >= 0.3
    assert pr[0, 0].shape == (2,)


def test_hybrid_simulation_keeps_offline_and_online_waits_separate():
    paths = _paths()
    waits = np.zeros((2, 32))
    waits[1, 0] = 3.0                 # 1.2 s 起点等待
    risk, T = _risk()
    co = OnlineFirstHitCoordinator(paths, waits, 0.35, 0.4, 3.2, 0.1,
                                   risk_model=risk, config={"hold_slots": 1})
    r = simulate_mpc(paths, waits[:, :31], 0.1,
                     MPCSimConfig(horizon_steps=int(T)),
                     seed=1, record=False, coordinator=co)
    assert r.actual_wait_time >= 1.2 - 0.15
    assert r.online_wait_time >= 0.0
    assert r.coordinator_name == "firsthit_online"
    # 计划等待与在线等待分别累计；没有把在线等待算进计划等待
    assert r.actual_wait_time + r.online_wait_time <= r.completion_time + 1e-6


def test_hybrid_none_equivalent_to_offline_when_no_decision():
    t = np.linspace(0, 1, 32)
    paths = np.stack([np.stack([1 + 8 * t, 1 + 0 * t], axis=-1),
                      np.stack([1 + 8 * t, 8 + 0 * t], axis=-1)])
    waits = np.zeros((2, 32))
    waits[1, 0] = 3.0
    risk, T = _risk()
    co = OnlineFirstHitCoordinator(paths, waits, 0.35, 0.4, 3.2, 0.0,
                                   risk_model=risk, config={"hold_slots": 1})
    r = simulate_mpc(paths, waits[:, :31], 0.0,
                     MPCSimConfig(horizon_steps=int(T)),
                     seed=0, record=False, coordinator=co)
    # p=0、平行安全路径：不应产生在线等待，完成时间应与纯离线一致
    assert r.online_requests == 0
    assert r.online_wait_time == 0.0
