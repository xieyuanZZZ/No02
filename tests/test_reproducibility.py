"""端到端一致性：端点固定、场景可复现、模拟器可复现。"""

from __future__ import annotations

import torch

from fhd.execution.simulator import simulate_execution
from fhd.maps.generate import generate_map
from fhd.maps.scenarios import build_scenario
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config
from fhd.utils.seed import set_seed

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml")


def test_scenario_reproducible():
    sc1 = build_scenario("crossing", CFG, offset=0.5)
    sc2 = build_scenario("crossing", CFG, offset=0.5)
    assert torch.allclose(sc1.paths, sc2.paths)
    assert torch.allclose(sc1.starts, sc2.starts)
    assert torch.allclose(sc1.goals, sc2.goals)


def test_map_reproducible():
    a = generate_map("bottleneck", CFG, 12345)
    b = generate_map("bottleneck", CFG, 12345)
    assert a.rects.shape == b.rects.shape and (a.rects == b.rects).all()
    assert a.circles.shape == b.circles.shape and (a.circles == b.circles).all()
    c = generate_map("bottleneck", CFG, 12346)
    different = (a.rects.shape != c.rects.shape) or not (a.rects == c.rects).all() \
        or (a.circles.shape != c.circles.shape) or not (a.circles == c.circles).all()
    assert different, "不同种子应生成不同地图"


def test_monte_carlo_reproducible():
    sc = build_scenario("crossing", CFG, offset=0.5)
    wait = torch.zeros(sc.n_agents, sc.horizon - 1, dtype=torch.float64)
    pause = torch.full((sc.n_agents,), 0.1, dtype=torch.float64)
    b1 = simulate_execution(sc.paths, wait, pause, 0.35, 96, 0.4, n_samples=200,
                            chunk_size=200, generator=torch.Generator().manual_seed(5))
    b2 = simulate_execution(sc.paths, wait, pause, 0.35, 96, 0.4, n_samples=200,
                            chunk_size=200, generator=torch.Generator().manual_seed(5))
    assert torch.equal(b1.collision, b2.collision)
    assert torch.equal(b1.makespan_steps, b2.makespan_steps)


def test_risk_deterministic_given_inputs():
    sc = build_scenario("head_on", CFG, h=8)
    wait = torch.zeros(2, 7, dtype=torch.float64)
    pause = torch.zeros(2, dtype=torch.float64)
    model = FirstHitRisk(d_safe=0.35)
    r1 = model(sc.paths, wait, pause, 7, collision_mode="hard", gate_mode="hard",
               return_diagnostics=False)["pair_risks"]
    r2 = model(sc.paths, wait, pause, 7, collision_mode="hard", gate_mode="hard",
               return_diagnostics=False)["pair_risks"]
    assert torch.equal(r1, r2)


def test_set_seed_reproducibility():
    set_seed(42)
    a = torch.randn(5, 3)
    set_seed(42)
    b = torch.randn(5, 3)
    assert torch.equal(a, b)


def test_endpoints_never_move_in_risk():
    """风险模块不修改输入张量（纯函数）。"""
    sc = build_scenario("crossing", CFG, offset=0.5)
    paths = sc.paths.clone()
    before = paths.clone()
    wait = torch.zeros(2, sc.horizon - 1, dtype=torch.float64)
    pause = torch.full((2,), 0.1, dtype=torch.float64)
    FirstHitRisk(d_safe=0.35)(paths, wait, pause, 96, collision_mode="soft", gate_mode="soft",
                              return_diagnostics=False)
    assert torch.equal(paths, before)
