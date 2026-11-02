"""R7 风险—模拟一致性：硬 DP 与独立 Monte Carlo 的二项检验对照。

判定规则（批复 §30）：使用精确二项检验（等价于 Wilson/Clopper-Pearson 区间），
而不是机械的固定误差阈值；对 p 接近 0/1 的极端案例同样适用。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from scipy.stats import binomtest

from fhd.execution.simulator import simulate_execution
from fhd.maps.scenarios import crossing
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml")
D_SAFE = 0.35
ALPHA = 0.001
N_MC = 10000


def _random_crossing_case(seed: int, h: int):
    """构造一对交叉路径：H 点、起终点安全、几何必然交叉。"""
    g = torch.Generator().manual_seed(seed)
    y1 = 1.0 + 2.0 * float(torch.rand((), generator=g))
    x1 = 1.0 + 2.0 * float(torch.rand((), generator=g))
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    a = torch.tensor([0.2, y1], dtype=torch.float64) + t * torch.tensor([3.6, 0.0], dtype=torch.float64)
    b = torch.tensor([x1, 0.2], dtype=torch.float64) + t * torch.tensor([0.0, 3.6], dtype=torch.float64)
    return torch.stack([a, b])


def _dp_risk(paths, wait, p, t_steps, model) -> float:
    out = model(paths, wait, p, time_steps=t_steps, collision_mode="hard", gate_mode="hard",
                return_diagnostics=False)
    return float(out["pair_risks"][0])


def test_r7_dp_vs_monte_carlo(tmp_path):
    cases = []
    n_cases = 20
    failures = 0
    for k in range(n_cases):
        h = 6 + (k % 4)
        t_steps = 20
        paths = _random_crossing_case(seed=k, h=h)
        wait = torch.zeros(2, h - 1, dtype=torch.float64)
        if k % 3 == 0:  # 部分案例加入等待，覆盖门槛非平凡的情形
            wait[1, 1] = 1.0
        p_val = [0.05, 0.10, 0.20][k % 3]
        pause = torch.tensor([p_val, p_val], dtype=torch.float64)
        model = FirstHitRisk(d_safe=D_SAFE, beta=0.25, offset=0.5, temperature_m=0.025, dt_s=0.4)
        dp = _dp_risk(paths, wait, pause, t_steps, model)

        mc = simulate_execution(
            paths, wait, pause, D_SAFE, t_steps, 0.4,
            n_samples=N_MC, chunk_size=5000,
            generator=torch.Generator().manual_seed(1000 + k),
        )
        hits = int(mc.collision.sum())
        test = binomtest(hits, N_MC, dp if 0.0 < dp < 1.0 else 1e-12, alternative="two-sided")
        pvalue = 1.0 if dp in (0.0, 1.0) and hits in (0, N_MC) else float(test.pvalue)
        record = {
            "case": k, "h": h, "t_steps": t_steps, "pause_p": p_val,
            "dp_risk": dp, "mc_hits": hits, "mc_n": N_MC, "mc_rate": hits / N_MC,
            "binom_pvalue": pvalue, "consistent": pvalue >= ALPHA,
        }
        cases.append(record)
        if pvalue < ALPHA:
            failures += 1
            (tmp_path / f"anomaly_{k}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    (tmp_path / "r7_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    assert failures == 0, (
        f"{failures}/{n_cases} 个案例的 DP 与 Monte Carlo 在 alpha={ALPHA} 下不一致；"
        f"异常案例已写入 {tmp_path}"
    )


def test_r7_known_scenarios_extremes():
    """极端情形：必然碰撞风险=1 与必然安全风险=0 都要与模拟完全一致。"""
    from fhd.maps.scenarios import head_on

    sc = head_on(CFG, h=8)
    wait = torch.zeros(2, 7, dtype=torch.float64)
    pause = torch.zeros(2, dtype=torch.float64)
    model = FirstHitRisk(d_safe=D_SAFE)
    dp_hit = _dp_risk(sc.paths, wait, pause, 7, model)
    mc_hit = simulate_execution(sc.paths, wait, pause, D_SAFE, 7, 0.4, n_samples=500, chunk_size=500,
                                generator=torch.Generator().manual_seed(3))
    assert dp_hit == 1.0 and mc_hit.collision_rate() == 1.0

    sc2 = crossing(CFG, h=16, offset=1.2)
    wait2 = torch.zeros(2, 15, dtype=torch.float64)
    dp_safe = _dp_risk(sc2.paths, wait2, pause, 15, model)
    mc_safe = simulate_execution(sc2.paths, wait2, pause, D_SAFE, 15, 0.4, n_samples=500,
                                 chunk_size=500, generator=torch.Generator().manual_seed(4))
    assert dp_safe == 0.0 and mc_safe.collision_rate() == 0.0
