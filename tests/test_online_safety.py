"""在线安全的定向测试：死锁/超时、静态障碍分离、连续碰撞与等待上限。"""

from __future__ import annotations

import json

import numpy as np

from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc
from fhd.geometry.static_maps import load_obstacles_for_map
from fhd.optim.online_baseline import SimpleConflictCoordinator
from fhd.risk.multi_pair import FirstHitRisk
from fhd.utils.config import load_config


def _paths4():
    t = np.linspace(0, 1, 32)
    return np.stack([
        np.stack([1 + 8 * t, 5 + 0 * t], axis=-1),
        np.stack([5 + 0 * t, 1 + 8 * t], axis=-1),
        np.stack([1 + 8 * t, 2 + 0 * t], axis=-1),
        np.stack([2 + 0 * t, 1 + 8 * t], axis=-1),
    ])


def test_b_online_does_not_deadlock_and_completes():
    paths = _paths4()
    waits = np.zeros((4, 31))
    cfg = MPCSimConfig()
    co = SimpleConflictCoordinator(paths, waits, cfg.d_safe, cfg.dt_slot, 3.0, 0.1,
                                   {"hold_slots": 1, "max_online_waits": 6})
    r = simulate_mpc(paths, waits, 0.1, cfg, seed=5, record=False, coordinator=co)
    assert r.all_arrived
    assert not r.timeout
    assert r.coordinator_stats["errors"] == 0
    # 等待不能无限增长
    assert r.online_wait_time <= 4 * 6 * 0.4 + 1e-6


def test_static_obstacle_and_pair_collision_are_separate_metrics():
    cfg = load_config("configs/base.yaml")
    T = int(cfg["execution"]["horizon_steps"])
    bank = json.loads((__import__("pathlib").Path("data/mpd_delay_plan_bank/plan_bank_n4.json")
                       .read_text(encoding="utf-8")))
    trajs = np.load("data/mpd_delay_plan_bank/plan_bank_n4_trajs.npz")
    jt = [t for t in bank["tasks"] if t["task_id"] == "single_obstacle_300004_n4_0"][0]
    paths = np.asarray(trajs[jt["task_id"]])
    waits = np.asarray(jt["baseline_wait"])
    obs, meta = load_obstacles_for_map(jt["map_id"])
    r = simulate_mpc(paths, waits, 0.0,
                     MPCSimConfig(horizon_steps=T), seed=1, record=False,
                     obstacles=obs, obstacle_meta=meta)
    assert r.static_obstacle_status == "evaluated"
    assert r.obstacle_collision is True
    assert r.min_obstacle_clearance < 0
    # 机器人间碰撞与障碍碰撞是两个字段
    assert isinstance(r.collision, bool)


def test_w2plus_and_online_share_static_metrics():
    cfg = load_config("configs/base.yaml")
    T = int(cfg["execution"]["horizon_steps"])
    bank = json.loads((__import__("pathlib").Path("data/mpd_delay_plan_bank/plan_bank_n4.json")
                       .read_text(encoding="utf-8")))
    trajs = np.load("data/mpd_delay_plan_bank/plan_bank_n4_trajs.npz")
    jt = bank["tasks"][0]
    paths = np.asarray(trajs[jt["task_id"]])
    waits = np.asarray(jt["baseline_wait"])
    obs, meta = load_obstacles_for_map(jt["map_id"])
    r = simulate_mpc(paths, waits, 0.1, MPCSimConfig(horizon_steps=T), seed=2,
                     record=False, obstacles=obs, obstacle_meta=meta,
                     coordinator=SimpleConflictCoordinator(
                         paths, waits, 0.35, 0.4, 3.0, 0.1, {}))
    assert r.static_obstacle_status == "evaluated"
    assert r.coordinator_stats["n_decisions"] > 0
    assert r.solve_stats["n_solves"] > 0
