"""S1 首次碰撞 DP 正确性测试 R1–R6（文档 §5.3）+ 独立暴力枚举对照。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_dp import brute_force_first_hit  # noqa: E402

from fhd.execution.simulator import simulate_deterministic  # noqa: E402
from fhd.maps.scenarios import crossing, head_on, terminal_occupancy  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "base.yaml")
D_SAFE = 2.0 * float(CFG["world"]["robot_radius_m"]) + float(CFG["world"]["safety_margin_m"])


def make_risk(**kwargs) -> FirstHitRisk:
    params = dict(d_safe=D_SAFE, beta=0.25, offset=0.5, temperature_m=0.025, kernel="logistic", dt_s=0.4,
                  lambda_tail_s=2.0)
    params.update(kwargs)
    return FirstHitRisk(**params)


def run(paths, wait, p, t_steps, collision="hard", gate="hard", **risk_kwargs):
    model = make_risk(**risk_kwargs)
    n = paths.shape[0]
    if isinstance(p, torch.Tensor):
        pause = p.to(paths.dtype).reshape(n)
    else:
        pause = torch.full((n,), float(p), dtype=paths.dtype)
    return model(paths, wait, pause, time_steps=t_steps, collision_mode=collision, gate_mode=gate)


# --------------------------------------------------------------------------- #
# R1 概率守恒
# --------------------------------------------------------------------------- #
def test_r1_probability_conservation():
    """H=3, p=0.1，两路径相距很远（无碰撞）时，硬 DP 总概率恒为 1。"""
    h, t_steps = 3, 4
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    p0 = torch.tensor([0.0, 0.0], dtype=torch.float64) + t * torch.tensor([0.0, 2.0], dtype=torch.float64)
    p1 = torch.tensor([5.0, 5.0], dtype=torch.float64) + t * torch.tensor([0.0, 2.0], dtype=torch.float64)
    paths = torch.stack([p0, p1])
    wait = torch.zeros(2, h - 1, dtype=torch.float64)
    out = run(paths, wait, 0.1, t_steps)
    survival = out["diagnostics"]["survival"]  # [P,T+1]
    assert survival.shape == (1, t_steps + 1)
    assert torch.allclose(survival, torch.ones_like(survival), atol=1e-12), survival
    assert float(out["pair_risks"][0]) == pytest.approx(0.0, abs=1e-12)
    # 单机进度分布与手工二项分布一致（H=3 时进度在 2 处饱和）
    from math import comb

    from fhd.execution.progress import move_probabilities, propagate_progress, release_times

    a = move_probabilities(
        release_times(wait), torch.tensor([0.1, 0.1], dtype=torch.float64), t_steps, gate_mode="hard"
    )
    pi = propagate_progress(a)
    p = 0.1
    exact = [comb(4, k) * (1 - p) ** k * p ** (4 - k) for k in range(5)]
    exact_h = [exact[0], exact[1], exact[2] + exact[3] + exact[4]]
    assert pi[0, 4].tolist() == pytest.approx(exact_h, abs=1e-12)


# --------------------------------------------------------------------------- #
# R2 确定性极限
# --------------------------------------------------------------------------- #
def test_r2_deterministic_limit_collision():
    """p=0、硬门槛、无等待：对向换位必然在中途相撞，风险严格为 1。"""
    sc = head_on(CFG, h=8)
    wait = torch.zeros(2, 7, dtype=torch.float64)
    out = run(sc.paths, wait, 0.0, t_steps=7)
    assert float(out["pair_risks"][0]) == pytest.approx(1.0, abs=1e-12)
    det = simulate_deterministic(sc.paths, wait, torch.zeros(2, dtype=torch.float64), D_SAFE, 7, 0.4)
    assert det["collision"] is True


def test_r2_deterministic_limit_safe():
    """p=0、硬门槛、无等待：错峰交叉的安全轨迹，风险严格为 0。"""
    sc = crossing(CFG, h=16, offset=0.9)
    wait = torch.zeros(2, 15, dtype=torch.float64)
    out = run(sc.paths, wait, 0.0, t_steps=15)
    assert float(out["pair_risks"][0]) == pytest.approx(0.0, abs=1e-12)
    det = simulate_deterministic(sc.paths, wait, torch.zeros(2, dtype=torch.float64), D_SAFE, 15, 0.4)
    assert det["collision"] is False


# --------------------------------------------------------------------------- #
# R3 首次碰撞只计一次
# --------------------------------------------------------------------------- #
def test_r3_first_hit_counted_once():
    """两机器人路径完全重合、持续处于碰撞状态，风险应为 1 而不是随时间累加。"""
    h, t_steps = 6, 30
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    p0 = torch.tensor([1.0, 1.0], dtype=torch.float64) + t * torch.tensor([3.0, 0.0], dtype=torch.float64)
    paths = torch.stack([p0, p0.clone()])
    wait = torch.zeros(2, h - 1, dtype=torch.float64)
    out = run(paths, wait, 0.2, t_steps)
    risk = float(out["pair_risks"][0])
    assert risk == pytest.approx(1.0, abs=1e-12)
    survival = out["diagnostics"]["survival"][0]
    assert float(survival[0]) == pytest.approx(1.0)  # t=0 时尚未吸收
    assert float(survival[-1]) == pytest.approx(0.0, abs=1e-12)
    assert torch.all(survival <= 1.0 + 1e-12)


# --------------------------------------------------------------------------- #
# R4 交叉段碰撞（同一时隙对向交换）
# --------------------------------------------------------------------------- #
def test_r4_swap_collision():
    # 两条交叉线段在 (1,0) 相交，起终点各相距约 1.118 m；只有同时前进才相撞
    qi = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float64)
    qj = torch.tensor([[[1.0, -0.5], [1.0, 0.5]]], dtype=torch.float64)
    paths = torch.cat([qi, qj], dim=0)
    wait = torch.zeros(2, 1, dtype=torch.float64)
    out = run(paths, wait, 0.0, t_steps=1)
    assert float(out["pair_risks"][0]) == pytest.approx(1.0)
    # 让机器人 0 在起点等待 1 个时隙（tau_0 = 1 > t = 0），则不再同时前进
    wait_wait = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    out2 = run(paths, wait_wait, 0.0, t_steps=1)
    assert float(out2["pair_risks"][0]) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# R5 终点占据
# --------------------------------------------------------------------------- #
def test_r5_terminal_occupancy():
    sc = terminal_occupancy(CFG, h=32)
    wait_safe = torch.zeros(2, 31, dtype=torch.float64)
    out_safe = run(sc.paths, wait_safe, 0.0, t_steps=96)
    assert float(out_safe["pair_risks"][0]) == pytest.approx(0.0, abs=1e-9), "不等待时 j 先穿过，不应碰撞"

    # 让 j 在前 15 个路点每个等 2 个时隙，使其在 i 已停在终点后才穿过
    wait_delay = torch.zeros(2, 31, dtype=torch.float64)
    wait_delay[1, :15] = 2.0
    out_hit = run(sc.paths, wait_delay, 0.0, t_steps=96)
    assert float(out_hit["pair_risks"][0]) == pytest.approx(1.0, abs=1e-9), "j 穿越 i 的终点必须被判为碰撞"


# --------------------------------------------------------------------------- #
# R6 等待行为
# --------------------------------------------------------------------------- #
def test_r6_wait_behaviour():
    """手工指定在安全点等待：硬风险下降、完成时间增大、与 MC 频率一致。"""
    from fhd.execution.simulator import simulate_execution

    sc = crossing(CFG, h=32, offset=0.5)
    wait0 = torch.zeros(2, 31, dtype=torch.float64)
    wait1 = torch.zeros(2, 31, dtype=torch.float64)
    wait1[0, 10:18] = 1.0  # 机器人 0 在进入交叉口前逐点等待

    out0 = run(sc.paths, wait0, 0.1, 96)
    out1 = run(sc.paths, wait1, 0.1, 96)
    r0, r1 = float(out0["pair_risks"][0]), float(out1["pair_risks"][0])
    assert r0 > 0.01, f"基线风险过小，无法验证等待效果: {r0}"
    assert r1 < r0, f"等待后风险应下降: {r0} -> {r1}"
    assert float(out1["finish_cost"]) > float(out0["finish_cost"]), "等待必须让完成时间代价增大"

    mc = simulate_execution(
        sc.paths, wait0, torch.tensor([0.1, 0.1], dtype=torch.float64), D_SAFE, 96, 0.4,
        n_samples=4000, chunk_size=2000, generator=torch.Generator().manual_seed(0),
    )
    freq = mc.collision_rate()
    se = (r0 * (1 - r0) / 4000) ** 0.5
    assert abs(freq - r0) < max(4 * se, 0.01), f"DP={r0:.4f} 与 MC={freq:.4f} 偏差超出统计误差"


# --------------------------------------------------------------------------- #
# 与独立暴力枚举对照
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dp_matches_bruteforce_small(seed):
    """小规模（H=4, T=6）上，DP 与独立暴力枚举的首次碰撞概率一致。"""
    torch.manual_seed(seed)
    h, t_steps = 4, 6
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    qa = torch.tensor([0.0, 0.0], dtype=torch.float64) + t * torch.tensor([2.0, 1.0], dtype=torch.float64)
    qb = torch.tensor([2.0, 1.0], dtype=torch.float64) + t * torch.tensor([-1.5, 1.0], dtype=torch.float64)
    qa = qa + torch.randn_like(qa) * 0.15
    qb = qb + torch.randn_like(qb) * 0.15
    paths = torch.stack([qa, qb])
    wait = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    p = torch.tensor([0.2, 0.3], dtype=torch.float64)
    out = run(paths, wait, p, t_steps)
    ref = brute_force_first_hit(
        qa.tolist(), qb.tolist(), wait[0].tolist(), wait[1].tolist(),
        0.2, 0.3, t_steps, D_SAFE, gate_mode="hard", n_samples=4001,
    )
    assert float(out["pair_risks"][0]) == pytest.approx(ref, abs=1e-9), (
        f"DP={float(out['pair_risks'][0])} 与暴力枚举={ref} 不一致"
    )
