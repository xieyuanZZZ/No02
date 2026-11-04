"""阶段三（等待优化）的验收测试：等待合法性、执行窗口、DP/MC 一致、回退机制。"""

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
    discretize,
    make_mask,
    policy_w0,
    policy_w2_enum,
    policy_w3_optimize,
    project_wait,
)
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "mpd_delay_plan_bank"
CFG = load_config(ROOT / "configs" / "base.yaml")

pytestmark = pytest.mark.skipif(not (BANK / "plan_bank.json").exists(),
                                reason="尚未构建 MPD 轨迹库")

T_STEPS = int(CFG["execution"]["horizon_steps"])
H = int(CFG["trajectory"]["horizon"])
DT = float(CFG["trajectory"]["dt_s"])


def _bank():
    bank = json.loads((BANK / "plan_bank.json").read_text(encoding="utf-8"))
    trajs = np.load(BANK / "plan_bank_trajs.npz")
    return bank, trajs


def _task(i: int = 0):
    bank, trajs = _bank()
    jt = bank["tasks"][i]
    paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
    w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
    return jt, paths, w0


def test_wait_projection_never_removes_stagger():
    """等待只能增加：投影后每台机器人的等待不得小于 baseline（含 3 时隙发车间隔）。"""
    _, paths, w0 = _task()
    budget = WaitBudget(time_steps=T_STEPS, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), 2)
    w = project_wait(w0 - 5.0, w0, mask, budget)      # 故意给负增量
    assert torch.all(w >= w0 - 1e-12)
    assert float(w[1, 0]) >= float(w0[1, 0])
    assert abs(float(w0[1, 0]) - 3.0) < 1e-9, "冻结任务的发车间隔应为 3 时隙"


def test_wait_respects_caps_and_integer_slots():
    _, paths, w0 = _task()
    budget = WaitBudget(time_steps=T_STEPS, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), 2)
    w = w0 + 50.0 * torch.ones_like(w0)               # 远超预算
    d = discretize(project_wait(w, w0, mask, budget), w0, mask, budget)
    added = d - w0
    assert torch.all(added >= -1e-12)
    assert float(added.max()) <= budget.max_added_per_point + 1e-9
    assert float(added.sum(dim=-1).max()) <= budget.max_added_total_per_robot + 1e-9
    assert torch.allclose(added, added.round()), "离散化后必须是整数时隙"


def test_wait_keeps_completion_inside_execution_window():
    _, paths, w0 = _task()
    budget = WaitBudget(time_steps=T_STEPS, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), 2)
    pause = torch.full((2,), 0.1, dtype=torch.float64)
    risk = FirstHitRisk.from_config(CFG)
    r = policy_w3_optimize(w0, mask, budget, risk, paths, pause, T_STEPS, steps=40)
    completion = (H - 1) + r.wait.sum(dim=-1)
    assert float(completion.max()) < T_STEPS, "新增等待不得让机器人越过执行窗口"
    assert torch.all(r.wait >= w0 - 1e-12)


def test_p0_execution_still_safe_after_waiting():
    """加等待后 p=0 仍必须无碰撞且全部到达（几何未变）。"""
    _, paths, w0 = _task()
    budget = WaitBudget(time_steps=T_STEPS, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), 2)
    pause = torch.full((2,), 0.1, dtype=torch.float64)
    risk = FirstHitRisk.from_config(CFG)
    r = policy_w2_enum(w0, mask, budget, risk, paths, pause, T_STEPS, lam_T=0.05)
    sim = simulate_execution(paths, r.wait, torch.zeros(2, dtype=torch.float64),
                             risk.d_safe, T_STEPS, DT, n_samples=1,
                             generator=torch.Generator().manual_seed(0))
    assert not bool(sim.collision.any())
    assert bool(sim.all_arrived.all())


def test_w2_never_worse_than_w0_in_dp_hard_risk():
    """W2 在 DP 硬风险 + 时间代价下不得劣于 W0（Δ=0 一定在候选里）。"""
    bank, trajs = _bank()
    risk = FirstHitRisk.from_config(CFG)
    budget = WaitBudget(time_steps=T_STEPS, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), 2)
    pause = torch.full((2,), 0.1, dtype=torch.float64)
    for jt in bank["tasks"][:6]:
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        r0 = policy_w0(w0, mask, budget, risk, paths, pause, T_STEPS)
        r2 = policy_w2_enum(w0, mask, budget, risk, paths, pause, T_STEPS, lam_T=0.05)
        j0 = r0.hard_risk + 0.05 * 0.0
        j2 = r2.hard_risk + 0.05 * (r2.finish_s - r0.finish_s)
        assert j2 <= j0 + 1e-9, f"{jt['task_id']} 的 W2 劣于 W0"


def test_w3_fallback_is_reported():
    """构造一个已经全安全的等待（w0 即最优），优化应回退或不劣于 w0。"""
    bank, trajs = _bank()
    risk = FirstHitRisk.from_config(CFG)
    budget = WaitBudget(time_steps=T_STEPS, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), 2)
    pause = torch.full((2,), 0.1, dtype=torch.float64)
    # 找一个初始硬风险最低的任务
    best = None
    for jt in bank["tasks"]:
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        r0 = policy_w0(w0, mask, budget, risk, paths, pause, T_STEPS)
        if best is None or r0.hard_risk < best[0]:
            best = (r0.hard_risk, jt, paths, w0)
    h0, jt, paths, w0 = best
    r3 = policy_w3_optimize(w0, mask, budget, risk, paths, pause, T_STEPS, steps=30)
    assert r3.hard_risk <= h0 + 1e-9, "W3 结果不得比 W0 更差（有回退保护）"
    assert isinstance(r3.fallback, bool)
