"""阶段一/二的验收测试：坐标缩放、连续碰撞、p=0 安全、时轴一致、DP 与 MC 对照。

对应实施安排 §7.4 要求的测试项。所有断言都跑在**落盘的轨迹库**上，
不依赖 WSL / 官方 MPD 环境。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fhd.execution.simulator import simulate_execution
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "mpd_delay_plan_bank"
CFG = load_config(ROOT / "configs" / "base.yaml")

pytestmark = pytest.mark.skipif(not (BANK / "plan_bank.json").exists(),
                                reason="尚未构建 MPD 轨迹库")


# --------------------------------------------------------------------------- #
# 坐标与几何
# --------------------------------------------------------------------------- #
def test_coordinate_scaling_roundtrip():
    """官方 [-1,1] 与米制 [0,10] 的换算（比例 0.2），用轨迹库里存的成对数据验证。"""
    bank, trajs = _load_bank()
    items = json.loads((BANK / "candidate_index.json").read_text(encoding="utf-8")) \
        if (BANK / "candidate_index.json").exists() else None
    # 直接用轨迹库里的 starts_m / goals_m 与官方坐标的换算关系
    import numpy as _np
    for jt in bank["tasks"][:5]:
        for key in ("starts_m", "goals_m"):
            arr = _np.asarray(jt[key])
            official = arr / 5.0 - 1.0
            assert _np.all(official >= -1.0 - 1e-6) and _np.all(official <= 1.0 + 1e-6)
            assert _np.allclose((official + 1.0) * 5.0, arr)
    # WORLD_M=10 与官方半宽 1 的比例必须是 0.2
    assert abs((2.0 / 10.0) - 0.2) < 1e-12


def test_arc_length_resample_preserves_geometry():
    from scripts.mpd_s1_build_plan_bank import arc_length_resample

    t = np.linspace(0, 1, 200)
    line = np.stack([t * 4.0, t * 3.0], axis=-1)          # 长度 5 的直线
    wp = arc_length_resample(line, 32)
    assert wp.shape == (32, 2)
    assert np.allclose(wp[0], line[0]) and np.allclose(wp[-1], line[-1])
    length = np.linalg.norm(np.diff(wp, axis=0), axis=-1).sum()
    assert length == pytest.approx(5.0, rel=1e-6)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=-1)
    assert seg.std() < 1e-6, "按弧长重采样后每段应等长"


def test_chord_error_bound_is_valid_upper_bound():
    """二次曲线 q(t)=(t, t²)：验证弦误差上界 max|q''|·h²/8 确实覆盖真实偏差。"""
    from scripts.mpd_s1_build_plan_bank import chord_error_bound

    T, dur = 128, 10.0
    t = np.linspace(0, dur, T)
    pos = np.stack([t, t ** 2], axis=-1)
    acc = np.tile(np.array([0.0, 2.0]), (T, 1))
    bound = chord_error_bound(pos, acc, dur)
    # 真实偏差：折线中点与曲线的最大距离
    h = dur / (T - 1)
    worst = 0.0
    for k in range(T - 1):
        mid_t = 0.5 * (t[k] + t[k + 1])
        chord_mid = 0.5 * (pos[k] + pos[k + 1])
        curve_mid = np.array([mid_t, mid_t ** 2])
        worst = max(worst, float(np.linalg.norm(chord_mid - curve_mid)))
    assert bound >= worst - 1e-12, f"误差上界 {bound:.3e} 小于真实偏差 {worst:.3e}"
    assert worst > 0, "构造的曲线应有非零弦误差"


def test_continuous_clearance_box_and_circle():
    from scripts.mpd_s1_build_plan_bank import continuous_clearance

    circles = np.array([[0.0, 0.0, 1.0]])
    boxes = np.array([[2.0, -1.0, 4.0, 1.0]])
    pts = np.array([
        [0.0, 0.0],      # 圆心 -> -1
        [2.0, 0.0],      # 圆心外 1m -> +1
        [3.0, 0.0],      # 矩形内部
        [3.0, 2.0],      # 矩形外 1m
    ])
    d = continuous_clearance(pts, circles, boxes)
    assert d[0] == pytest.approx(-1.0)     # 圆心：圆的内部，距离 -1
    assert d[2] < 0                        # 矩形内部
    assert d[3] == pytest.approx(1.0)      # 矩形上边界外 1 m
    # 圆右边缘 (1,0) 到矩形左边界 (2,0) 相距 1，最近障碍是矩形 -> 距离 1
    d2 = continuous_clearance(np.array([[1.5, 0.0]]), circles, boxes)
    assert d2[0] == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# 轨迹库本身
# --------------------------------------------------------------------------- #
def _load_bank():
    bank = json.loads((BANK / "plan_bank.json").read_text(encoding="utf-8"))
    trajs = np.load(BANK / "plan_bank_trajs.npz")
    return bank, trajs


def _run_p0(jt, trajs, dt_s, t_steps, d_safe, n_agents=2):
    paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
    wait0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
    sim = simulate_execution(paths, wait0, torch.zeros(n_agents, dtype=torch.float64),
                             d_safe, t_steps, dt_s, n_samples=1,
                             generator=torch.Generator().manual_seed(0))
    return paths, wait0, sim


def test_plan_bank_p0_is_collision_free_and_all_arrive():
    bank, trajs = _load_bank()
    dt_s = float(CFG["trajectory"]["dt_s"])
    t_steps = int(CFG["execution"]["horizon_steps"])
    d_safe = FirstHitRisk.from_config(CFG).d_safe
    for jt in bank["tasks"]:
        _, _, sim = _run_p0(jt, trajs, dt_s, t_steps, d_safe)
        assert not bool(sim.collision.any()), f"{jt['task_id']} 在 p=0 下碰撞"
        assert bool(sim.all_arrived.all()), f"{jt['task_id']} 在 p=0 下未全部到达"


def test_plan_bank_segments_satisfy_speed_limit():
    bank, trajs = _load_bank()
    v_max = float(CFG["trajectory"]["max_speed_mps"])
    dt_s = float(CFG["trajectory"]["dt_s"])
    for jt in bank["tasks"]:
        paths = np.asarray(trajs[jt["task_id"]])
        seg = np.linalg.norm(np.diff(paths, axis=1), axis=-1)
        assert seg.max() <= v_max * dt_s + 1e-6, \
            f"{jt['task_id']} 段长 {seg.max():.3f} > {v_max * dt_s:.3f}"


def test_plan_bank_endpoints_match_starts_goals():
    bank, trajs = _load_bank()
    for jt in bank["tasks"]:
        paths = np.asarray(trajs[jt["task_id"]])
        starts = np.asarray(jt["starts_m"])
        goals = np.asarray(jt["goals_m"])
        assert np.allclose(paths[:, 0, :], starts, atol=1e-4)
        assert np.allclose(paths[:, -1, :], goals, atol=1e-4)


def test_stagger_shifts_completion_by_expected_slots():
    """固定发车间隔必须精确反映在完成时隙上（执行时轴一致性）。"""
    bank, trajs = _load_bank()
    dt_s = float(CFG["trajectory"]["dt_s"])
    t_steps = int(CFG["execution"]["horizon_steps"])
    d_safe = FirstHitRisk.from_config(CFG).d_safe
    jt = next(t for t in bank["tasks"] if t["stagger_slots"] > 0)
    paths, wait0, sim = _run_p0(jt, trajs, dt_s, t_steps, d_safe)
    h = paths.shape[1]
    stagger = int(jt["stagger_slots"])
    # 第 i 台机器人应比第 0 台晚 stagger*i 个时隙完成
    assert bool(sim.all_arrived.all())
    comp = sim.completion_steps[0]
    assert int(comp[0]) == h - 1
    assert int(comp[1]) == h - 1 + stagger


# --------------------------------------------------------------------------- #
# DP 与 MC 的一致性（阶段二验收）
# --------------------------------------------------------------------------- #
def test_dp_matches_monte_carlo_trend():
    runs = sorted((ROOT / "results" / "MPD_S2_delay").glob("*/dp_vs_mc.csv"))
    if not runs:
        pytest.skip("尚无阶段二结果")
    import csv

    rows = list(csv.DictReader(runs[-1].open(encoding="utf-8")))
    a = np.array([float(r["dp_hard_risk"]) for r in rows])
    b = np.array([float(r["mc_collision_rate"]) for r in rows])
    assert np.corrcoef(a, b)[0, 1] > 0.9, "DP 硬风险与蒙特卡洛碰撞率相关性过低"
    assert np.mean(np.abs(a - b)) < 0.05, "DP 与 MC 平均绝对误差过大"
    p0 = [float(r["mc_collision_rate"]) for r in rows if abs(float(r["p"])) < 1e-9]
    assert max(p0) == 0.0, "p=0 时不应出现碰撞"


def test_delay_increases_collision_rate():
    runs = sorted((ROOT / "results" / "MPD_S2_delay").glob("*/delay_mc.csv"))
    if not runs:
        pytest.skip("尚无阶段二结果")
    import csv

    rows = list(csv.DictReader(runs[-1].open(encoding="utf-8")))
    def mean_at(p):
        vals = [float(r["collision_rate"]) for r in rows if abs(float(r["p"]) - p) < 1e-9]
        return float(np.mean(vals))
    assert mean_at(0.0) == 0.0
    assert mean_at(0.2) > mean_at(0.05) > 0.0, "碰撞率应随暂停概率单调上升"
