"""W2+ 的验收测试：与 W3 同约束、p=0 安全、预算合法、无梯度法不劣于 W0。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fhd.optim.wait_policies import (
    WaitBudget,
    allowed_positions,
    make_mask,
    policy_w0,
    policy_w2plus,
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


def _task(i: int = 0):
    bank = json.loads((BANK / "plan_bank_n4.json").read_text(encoding="utf-8"))
    trajs = np.load(BANK / "plan_bank_n4_trajs.npz")
    jt = bank["tasks"][i]
    paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
    w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
    return jt, paths, w0


def _setup():
    risk = FirstHitRisk.from_config(CFG)
    budget = WaitBudget(time_steps=T_STEPS, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), 4)
    pause = torch.full((4,), 0.1, dtype=torch.float64)
    zero = torch.zeros(4, dtype=torch.float64)
    return risk, budget, mask, pause, zero


def test_w2plus_uses_same_variables_as_w3():
    """W2+ 与 W3 的变量集合与上界必须完全一致。"""
    _, budget, mask, pause, _ = _setup()
    assert mask.shape == (4, H - 1)
    positions = [(i, k) for i in range(4) for k in range(H - 1) if float(mask[i, k]) > 0]
    assert len(positions) == 4 * len(allowed_positions(H, 5))  # 28 个变量
    assert budget.max_added_per_point == 6.0
    assert budget.max_added_total_per_robot == 12.0


def test_w2plus_respects_budget_and_integer_slots():
    jt, paths, w0 = _task(0)
    risk, budget, mask, pause, _ = _setup()
    r = policy_w2plus(w0, mask, budget, risk, paths, pause, T_STEPS,
                      max_evals=200, restarts=2)
    added = r.wait - w0
    assert torch.all(added >= -1e-9)
    assert torch.allclose(added, added.round()), "必须是整数时隙"
    assert float(added.max()) <= budget.max_added_per_point + 1e-9
    cap = min(budget.max_added_total_per_robot, budget.max_added_total_allowed())
    assert float(added.sum(dim=-1).max()) <= cap + 1e-9
    completion = (H - 1) + r.wait.sum(dim=-1)
    assert float(completion.max()) < T_STEPS


def test_w2plus_keeps_p0_safe():
    jt, paths, w0 = _task(0)
    risk, budget, mask, pause, zero = _setup()
    r = policy_w2plus(w0, mask, budget, risk, paths, pause, T_STEPS,
                      max_evals=200, restarts=2)
    with torch.no_grad():
        h0 = risk(paths, r.wait, zero, T_STEPS, collision_mode="hard",
                  gate_mode="hard", return_diagnostics=False)
    assert float(h0["pair_sum_risk"]) <= 1e-9
    assert r.p0_safe


def test_w2plus_not_worse_than_w0():
    jt, paths, w0 = _task(1)
    risk, budget, mask, pause, _ = _setup()
    r0 = policy_w0(w0, mask, budget, risk, paths, pause, T_STEPS)
    rp = policy_w2plus(w0, mask, budget, risk, paths, pause, T_STEPS,
                       max_evals=300, restarts=2)
    assert rp.hard_risk <= r0.hard_risk + 1e-9


def test_w2plus_reports_no_gradient_steps():
    jt, paths, w0 = _task(0)
    risk, budget, mask, pause, _ = _setup()
    r = policy_w2plus(w0, mask, budget, risk, paths, pause, T_STEPS,
                      max_evals=120, restarts=1)
    assert r.grad_steps == 0, "无梯度方法不应记录梯度步"
    assert r.dp_evals > 0


def test_w3_reports_gradient_and_dp_counts():
    jt, paths, w0 = _task(0)
    risk, budget, mask, pause, _ = _setup()
    r = policy_w3_optimize(w0, mask, budget, risk, paths, pause, T_STEPS, steps=20)
    assert r.grad_steps == 20
    assert r.dp_evals >= 0
