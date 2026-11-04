"""N=4 联合轨迹库：用**逐对增量构造**代替随机抽样。

思路：
  * p=0 的碰撞是逐对（pairwise）的，两机的相对发车间隔只取决于 |i-j|·stagger；
  * 先对每张地图的所有候选两两预计算"在给定相对间隔下是否 p=0 安全"（带缓存）；
  * 再在 4 个机器人任务上做回溯搜索，只在兼容集合里扩展。

输出（与 N=2 的库并列，不覆盖）：
  data/mpd_delay_plan_bank/plan_bank_n4.json
  data/mpd_delay_plan_bank/plan_bank_n4_trajs.npz
  data/mpd_delay_plan_bank/planning_coverage_n4.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

from scripts.mpd_s1_build_plan_bank import audit_candidate, candidate_waypoints  # noqa: E402


def pair_safe(paths_a, paths_b, delta: int, d_safe, t_steps, dt_s) -> bool:
    """两台机器人、相对发车间隔 delta 时的 p=0 安全性。"""
    paths = torch.stack([torch.as_tensor(paths_a, dtype=torch.float64),
                         torch.as_tensor(paths_b, dtype=torch.float64)])
    wait = torch.zeros(2, paths.shape[1] - 1, dtype=torch.float64)
    wait[1, 0] = float(delta)
    sim = simulate_execution(paths, wait, torch.zeros(2, dtype=torch.float64),
                             d_safe, t_steps, dt_s, n_samples=1,
                             generator=torch.Generator().manual_seed(0))
    return (not bool(sim.collision.any())) and bool(sim.all_arrived.all())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(ROOT / "data" / "mpd_delay_plan_bank"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--stagger-slots", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=20)
    ap.add_argument("--max-cands-per-task", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    h_wp = int(cfg["trajectory"]["horizon"])
    v_max = float(cfg["trajectory"]["max_speed_mps"])
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe
    n_agents = int(args.n_agents)
    stagger = int(args.stagger_slots)

    bank_dir = Path(args.bank)
    run_dir = new_run_dir("MPD_N4_plan_bank", f"n{n_agents}_greedy")
    logger = get_logger("fhd.mpd_n4", run_dir / "logs" / "n4.log")
    log_section(logger, f"构建 N={n_agents} 联合轨迹库（逐对增量构造）")

    maps_all = {m["map_id"] if "map_id" in m else f"{m['kind']}_{m['seed']}": m
                for m in json.loads((Path("D:/paper2/data/revised_s1/maps_test.json"))
                                    .read_text(encoding="utf-8"))}
    duration = float(json.loads((bank_dir / "candidate_index.json")
                                .read_text(encoding="utf-8"))["trajectory_duration"])

    # ---- 1) 审计全部候选 ----
    valid: dict[tuple[str, str], list[int]] = defaultdict(list)
    n_audit = 0
    for f in sorted((bank_dir / "candidates").glob("cand_*.npz")):
        z = np.load(f, allow_pickle=True)
        map_id = str(z["map_id"][0])
        if map_id not in maps_all:
            continue
        ti = f.stem.split("_t")[-1]
        sc = maps_all[map_id]
        rects = np.asarray(sc["rects"], dtype=np.float64).reshape(-1, 4)
        circ = np.asarray(sc["circles"], dtype=np.float64).reshape(-1, 3)
        pos_m = z["pos_m_raw"].astype(np.float64)
        acc = z["acc_raw"].astype(np.float64)
        start_m = z["start_m"].astype(np.float64)
        goal_m = z["goal_m"].astype(np.float64)
        for k in range(pos_m.shape[0]):
            n_audit += 1
            a = audit_candidate(pos_m[k], acc[k], duration, start_m, goal_m,
                                circ, rects, v_max, dt_s, h_wp)
            if a["ok"]:
                valid[(map_id, ti)].append(k)
    logger.info("审计 %d 条候选，有效 %d 条", n_audit, sum(len(v) for v in valid.values()))

    per_map: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for (map_id, ti), cands in valid.items():
        per_map[map_id][ti] = cands

    # ---- 2) 逐对增量构造 ----
    joint = []
    stats = {"maps_eligible": 0, "maps_solved": 0, "pair_checks": 0}
    cache: dict[tuple, bool] = {}

    def compatible(map_id, items):
        for a, b in itertools.combinations(range(len(items)), 2):
            delta = stagger * abs(a - b)
            key = (map_id, items[a][0], items[a][1], items[b][0], items[b][1], delta)
            if key in cache:
                if not cache[key]:
                    return False
                continue
            stats["pair_checks"] += 1
            ok = pair_safe(items[a][2], items[b][2], delta, d_safe, t_steps, dt_s)
            cache[key] = ok
            if not ok:
                return False
        return True

    for map_id, tasks_map in sorted(per_map.items()):
        if len(tasks_map) < n_agents:
            continue
        stats["maps_eligible"] += 1
        task_ids = sorted(tasks_map)
        solved = []
        for combo in itertools.combinations(task_ids, n_agents):
            if len(solved) >= 2:
                break
            pool = []
            for ti in combo:
                entries = []
                for c in tasks_map[ti][:args.max_cands_per_task]:
                    wp, start_m, goal_m, info = candidate_waypoints(bank_dir, map_id, ti, c, h_wp)
                    entries.append((ti, c, wp, start_m, goal_m, info))
                pool.append(entries)
            if any(len(e) == 0 for e in pool):
                continue

            found = []

            def dfs(i, chosen):
                if found:
                    return True
                if i == n_agents:
                    found.append(list(chosen))
                    return True
                for e in pool[i]:
                    items = [(c[0], c[1], c[2]) for c in chosen] + [(e[0], e[1], e[2])]
                    if compatible(map_id, items):
                        chosen.append(e)
                        if dfs(i + 1, chosen):
                            return True
                        chosen.pop()
                return False

            if dfs(0, []):
                ch = found[0]
                paths = torch.tensor(np.stack([c[2] for c in ch]), dtype=torch.float64)
                wait0 = torch.zeros(n_agents, h_wp - 1, dtype=torch.float64)
                for i in range(n_agents):
                    wait0[i, 0] = stagger * i
                sim = simulate_execution(paths, wait0, torch.zeros(n_agents, dtype=torch.float64),
                                         d_safe, t_steps, dt_s, n_samples=1,
                                         generator=torch.Generator().manual_seed(0))
                assert not bool(sim.collision.any()), "构造出的组合必须 p=0 安全"
                solved.append({
                    "task_id": f"{map_id}_n{n_agents}_{len(solved)}",
                    "map_id": map_id, "kind": maps_all[map_id]["kind"],
                    "n_agents": n_agents,
                    "starts_m": [c[3].tolist() for c in ch],
                    "goals_m": [c[4].tolist() for c in ch],
                    "paths_m": paths.numpy().tolist(),
                    "stagger_slots": stagger,
                    "baseline_wait": wait0.numpy().tolist(),
                    "p0_pair_min_distance": float(sim.pair_min_distance.min()),
                    "source": {"files": [f"cand_{map_id}_t{int(c[0]):02d}.npz" for c in ch],
                               "cands": [int(c[1]) for c in ch],
                               "task_indices": [int(c[0]) for c in ch]},
                    "path_lengths_m": [float(c[5]["path_length"]) for c in ch],
                    "planned_completion_slots": int(h_wp - 1 + stagger * (n_agents - 1)),
                })
        if solved:
            stats["maps_solved"] += 1
            joint.extend(solved[:2])
            logger.info("  %s：%d 组入库（最小间距 %.3f）", map_id, len(solved[:2]),
                        min(s["p0_pair_min_distance"] for s in solved))
        else:
            logger.info("  %s：无可行组合（%d 个机器人任务）", map_id, len(tasks_map))
        if len(joint) >= args.max_tasks:
            break

    logger.info("N=%d 联合任务：%d 个（地图 %d/%d，配对检查 %d 次）", n_agents, len(joint),
                stats["maps_solved"], stats["maps_eligible"], stats["pair_checks"])

    trajs = {jt["task_id"]: np.asarray(jt["paths_m"], dtype=np.float32) for jt in joint}
    np.savez_compressed(bank_dir / f"plan_bank_n{n_agents}_trajs.npz", **trajs)
    (bank_dir / f"plan_bank_n{n_agents}.json").write_text(json.dumps({
        "created_by": "scripts/mpd_n4_build_plan_bank.py",
        "config": {"v_max": v_max, "dt_s": dt_s, "H": h_wp, "T": t_steps,
                   "d_safe": d_safe, "stagger_slots": stagger,
                   "n_agents": n_agents, "world_m": 10.0},
        "n_joint_tasks": len(joint),
        "coverage": stats,
        "tasks": joint,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (bank_dir / f"planning_coverage_n{n_agents}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_agents": n_agents, "tasks": len(joint), **stats},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
