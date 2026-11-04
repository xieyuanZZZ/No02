"""N=4 轨迹库与等待策略的验收测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fhd.execution.simulator import simulate_execution
from fhd.optim.wait_policies import (
    WaitBudget,
    allowed_positions,
    make_mask,
    policy_w2_enum,
    policy_w3_optimize,
)
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "mpd_delay_plan_bank"
CFG = load_config(ROOT / "configs" / "base.yaml")

pytestmark = pytest.mark.skipif(not (BANK / "plan_bank_n4.json").exists(),
                                reason="尚未构建 N=4 轨迹库")

T_STEPS = int(CFG["execution"]["horizon_steps"])
H = int(CFG["trajectory"]["horizon"])
DT = float(CFG["trajectory"]["dt_s"])
V_MAX = float(CFG["trajectory"]["max_speed_mps"])


def _bank():
    bank = json.loads((BANK / "plan_bank_n4.json").read_text(encoding="utf-8"))
    trajs = np.load(BANK / "plan_bank_n4_trajs.npz")
    return bank, trajs


def test_n4_bank_has_enough_tasks():
    bank, _ = _bank()
    assert bank["n_joint_tasks"] >= 10, "N=4 任务太少，统计功效不足"
    assert all(t["n_agents"] == 4 for t in bank["tasks"])


def test_n4_p0_collision_free_and_all_arrive():
    bank, trajs = _bank()
    d_safe = FirstHitRisk.from_config(CFG).d_safe
    for jt in bank["tasks"]:
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        wait0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        sim = simulate_execution(paths, wait0, torch.zeros(4, dtype=torch.float64),
                                 d_safe, T_STEPS, DT, n_samples=1,
                                 generator=torch.Generator().manual_seed(0))
        assert not bool(sim.collision.any()), f"{jt['task_id']} 在 p=0 下碰撞"
        assert bool(sim.all_arrived.all())


def test_n4_speed_and_endpoints():
    bank, trajs = _bank()
    for jt in bank["tasks"]:
        paths = np.asarray(trajs[jt["task_id"]])
        assert paths.shape[1] == H and paths.shape[0] == 4
        seg = np.linalg.norm(np.diff(paths, axis=1), axis=-1)
        assert seg.max() <= V_MAX * DT + 1e-6
        assert np.allclose(paths[:, 0, :], np.asarray(jt["starts_m"]), atol=1e-4)
        assert np.allclose(paths[:, -1, :], np.asarray(jt["goals_m"]), atol=1e-4)


def test_n4_w2_w3_keep_p0_safe_and_window():
    """W2/W3 的等待必须保持 p=0 安全，且不越过执行窗口。"""
    bank, trajs = _bank()
    risk = FirstHitRisk.from_config(CFG)
    budget = WaitBudget(time_steps=T_STEPS, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), 4)
    pause = torch.full((4,), 0.1, dtype=torch.float64)
    zero = torch.zeros(4, dtype=torch.float64)
    for jt in bank["tasks"][:4]:
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        for r in (policy_w2_enum(w0, mask, budget, risk, paths, pause, T_STEPS, lam_T=0.05),
                  policy_w3_optimize(w0, mask, budget, risk, paths, pause, T_STEPS, steps=40)):
            with torch.no_grad():
                h0 = risk(paths, r.wait, zero, T_STEPS, collision_mode="hard",
                          gate_mode="hard", return_diagnostics=False)
            assert float(h0["pair_sum_risk"]) <= 1e-9, \
                f"{jt['task_id']} 的等待破坏了 p=0 安全"
            completion = (H - 1) + r.wait.sum(dim=-1)
            assert float(completion.max()) < T_STEPS
