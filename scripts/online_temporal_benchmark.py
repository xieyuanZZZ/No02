"""A/B/C 在线时序协调配对实验。

A = 离线 W2+（执行期不看状态）
B = 原始等待 + 简单在线冲突预测/固定优先级让行
C = 原始等待 + 当前状态条件化 First-Hit 有限搜索

公平性：同一冻结路径、同一 MPC 参数、同一 `d_safe`、同一预生成外部暂停事件表、
同一决策周期；A/B/C 的执行器循环完全相同，只有 Coordinator 是否/如何决策不同。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.evaluation.risk_dataset import bootstrap_task_ci  # noqa: E402
from fhd.execution.mpc_simulator import (MPCSimConfig, make_pause_events,  # noqa: E402
                                         simulate_mpc)
from fhd.execution.coordinator import Coordinator  # noqa: E402
from fhd.geometry.static_maps import load_obstacles_for_map  # noqa: E402
from fhd.optim.online_baseline import SimpleConflictCoordinator  # noqa: E402
from fhd.optim.online_firsthit import OnlineFirstHitCoordinator  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget, allowed_positions, make_mask, policy_w0, policy_w2plus,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

import yaml  # noqa: E402


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _w2plus_for_task(jt, n, risk, cfg, ti, budget_cfg):
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    budget = WaitBudget(time_steps=T, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), n)
    paths = torch.tensor(np.asarray(jt["paths_m"]), dtype=torch.float64)
    w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
    pause01 = torch.full((n,), 0.1, dtype=torch.float64)
    t0 = time.perf_counter()
    res = policy_w2plus(w0, mask, budget, risk, paths, pause01, T,
                        lam_T=budget_cfg["lam_T"], max_evals=budget_cfg["max_evals"],
                        restarts=budget_cfg["restarts"], seed=ti)
    elapsed = time.perf_counter() - t0
    return res.wait.numpy(), float(res.hard_risk), elapsed, int(res.dp_evals)


def _make_coordinator(method, paths, wait, d_safe, dt_slot, pause_p, risk, ccfg):
    if method == "B_simple_online":
        return SimpleConflictCoordinator(
            paths, wait, d_safe, dt_slot, ccfg["B_horizon_s"], pause_p,
            {**ccfg})
    if method == "C_online_firsthit":
        return OnlineFirstHitCoordinator(
            paths, wait, d_safe, dt_slot, ccfg["C_horizon_s"], pause_p,
            risk_model=risk, config={**ccfg})
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/Online_Temporal/03_ab_comparison")
    ap.add_argument("--n4-tasks", type=int, default=8)
    ap.add_argument("--n-mc", type=int, default=30)
    ap.add_argument("--ps", default="0,0.05,0.1,0.2")
    ap.add_argument("--seed0", type=int, default=20281115)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task-ids", default="")
    args = ap.parse_args()

    base = load_config(ROOT / "configs" / "base.yaml")
    tcfg = _load_cfg(ROOT / "configs" / "online_temporal.yaml")
    set_seed(int(tcfg["seed"]))
    T = int(base["execution"]["horizon_steps"])
    H = int(base["trajectory"]["horizon"])
    dt_slot = float(base["trajectory"]["dt_s"])
    risk = FirstHitRisk.from_config(base)
    d_safe = risk.d_safe
    ps = [float(x) for x in args.ps.split(",")]
    if args.smoke:
        args.n4_tasks = min(args.n4_tasks, 2)
        args.n_mc = min(args.n_mc, 2)
        ps = [0.1]
    method_names = ["A_w2plus", "B_simple_online", "C_online_firsthit"]
    ccfg = dict(tcfg["coordinator"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.online_bench", out / "online_bench.log")
    log_section(logger, "A/B/C 在线时序协调配对实验")

    bank = json.loads((ROOT / "data" / "mpd_delay_plan_bank" / "plan_bank_n4.json")
                      .read_text(encoding="utf-8"))
    trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / "plan_bank_n4_trajs.npz")
    tasks = bank["tasks"]
    if args.task_ids:
        ids = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t["task_id"] in ids]
    tasks = tasks[:args.n4_tasks]
    logger.info("任务数 %d：%s", len(tasks), [t["task_id"] for t in tasks])

    # 预计算 A 的离线 W2+
    plans = {}
    plan_meta = {}
    for ti, jt in enumerate(tasks):
        w0 = np.asarray(jt["baseline_wait"], dtype=np.float64)
        w2, hard, elapsed, dp_evals = _w2plus_for_task(
            jt, 4, risk, base, ti, tcfg["offline_w2plus"])
        plans[jt["task_id"]] = {"orig": w0, "A_w2plus": w2}
        plan_meta[jt["task_id"]] = {
            "w2plus_hard_risk": hard, "w2plus_elapsed_s": elapsed,
            "w2plus_dp_evals": dp_evals,
        }
        logger.info("[W2+] %s 完成（%.1fs, %d DP）", jt["task_id"], elapsed, dp_evals)

    rows = []
    t_start = time.perf_counter()
    for ti, jt in enumerate(tasks):
        tid = jt["task_id"]
        paths = np.asarray(trajs[tid], dtype=np.float64)
        w0 = plans[tid]["orig"]
        w2 = plans[tid]["A_w2plus"]
        obs, obs_meta = load_obstacles_for_map(str(jt.get("map_id", "")))
        for p in ps:
            for k in range(args.n_mc):
                seed = args.seed0 + ti * 1000 + k
                events = make_pause_events(seed, p, 4, T,
                                           MPCSimConfig(d_safe=d_safe,
                                                        horizon_steps=T))
                for method in method_names:
                    wait = w2 if method == "A_w2plus" else w0
                    coord = _make_coordinator(method, paths, wait, d_safe,
                                              dt_slot, p, risk, ccfg)
                    r = simulate_mpc(paths, wait, p,
                                     MPCSimConfig(d_safe=d_safe, horizon_steps=T),
                                     seed=seed, record=False, events=events,
                                     obstacles=obs, obstacle_meta=obs_meta,
                                     coordinator=coord)
                    rows.append({
                        "method": method, "task_id": tid,
                        "map_id": jt.get("map_id", ""), "p": p, "seed": seed,
                        "n_agents": 4,
                        "collision": bool(r.collision),
                        "safe_completion": bool(r.safe_completion),
                        "timeout": bool(r.timeout),
                        "all_arrived": bool(r.all_arrived),
                        "completion_s": float(r.completion_time),
                        "min_pair_distance": float(np.nanmin(r.pair_min_distance)),
                        "obstacle_collision": r.obstacle_collision,
                        "min_obstacle_clearance": r.min_obstacle_clearance,
                        "infeasible_stops": int(r.infeasible_stops),
                        "mpc_failures": int(r.mpc_failures),
                        "constraint_violations": int(r.constraint_violations),
                        "controller_failed": bool(r.controller_failed),
                        "pause_requests": int(r.pause_requests),
                        "pause_accepted": int(r.pause_accepted),
                        "online_requests": int(r.online_requests),
                        "online_accepted": int(r.online_accepted),
                        "online_rejected": int(r.online_rejected),
                        "online_wait_s": float(r.online_wait_time),
                        "actual_wait_s": float(r.actual_wait_time),
                        "random_hold_s": float(r.random_hold_time),
                        "total_wait_s": float(r.actual_wait_time + r.random_hold_time
                                               + r.online_wait_time),
                        "coord_n_decisions": int(r.coordinator_stats.get("n_decisions", 0)),
                        "coord_mean_ms": float(r.coordinator_stats.get("mean_ms", 0.0)),
                        "coord_p95_ms": float(r.coordinator_stats.get("p95_ms", 0.0)),
                        "coord_p99_ms": float(r.coordinator_stats.get("p99_ms", 0.0)),
                        "coord_max_ms": float(r.coordinator_stats.get("max_ms", 0.0)),
                        "coord_over_budget": int(r.coordinator_stats.get("over_budget", 0)),
                        "coord_errors": int(r.coordinator_stats.get("errors", 0)),
                        "solve_mean_ms": float(r.solve_stats.get("mean_ms", 0.0)),
                        "tracking_rmse": float(r.tracking_rmse),
                    })
        logger.info("[%s] 完成", tid)

    fieldnames = list(rows[0].keys())
    with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # per task/p/method
    keys = sorted({(r["task_id"], r["method"], r["p"]) for r in rows})
    task_rows = []
    for tid, method, p in keys:
        sel = [r for r in rows if r["task_id"] == tid and r["method"] == method
               and abs(r["p"] - p) < 1e-12]
        n = len(sel)
        task_rows.append({
            "task_id": tid, "map_id": sel[0]["map_id"], "method": method, "p": p,
            "n_mc": n,
            "collision_rate": float(np.mean([r["collision"] for r in sel])),
            "safe_rate": float(np.mean([r["safe_completion"] for r in sel])),
            "timeout_rate": float(np.mean([r["timeout"] for r in sel])),
            "completion_mean_s": float(np.mean([r["completion_s"] for r in sel])),
            "total_wait_mean_s": float(np.mean([r["total_wait_s"] for r in sel])),
            "online_wait_mean_s": float(np.mean([r["online_wait_s"] for r in sel])),
            "online_accept_rate": float(np.mean(
                [r["online_accepted"] / max(r["online_requests"], 1) for r in sel])),
            "min_pair_distance": float(np.min([r["min_pair_distance"] for r in sel])),
            "obstacle_collision_rate": float(np.mean(
                [bool(r["obstacle_collision"]) for r in sel
                 if r["obstacle_collision"] is not None])) if any(
                r["obstacle_collision"] is not None for r in sel) else None,
            "infeasible_stops": int(sum(r["infeasible_stops"] for r in sel)),
            "mpc_failures": int(sum(r["mpc_failures"] for r in sel)),
            "coord_mean_ms": float(np.mean([r["coord_mean_ms"] for r in sel])),
            "coord_p95_ms": float(np.mean([r["coord_p95_ms"] for r in sel])),
            "coord_over_budget": int(sum(r["coord_over_budget"] for r in sel)),
            "coord_errors": int(sum(r["coord_errors"] for r in sel)),
        })
    with (out / "per_task.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(task_rows[0].keys()))
        w.writeheader()
        w.writerows(task_rows)

    def cell(method, p):
        sel = [r for r in task_rows if r["method"] == method and abs(r["p"] - p) < 1e-12]
        return {
            "n_tasks": len(sel),
            "collision_rate": float(np.mean([r["collision_rate"] for r in sel])),
            "safe_rate": float(np.mean([r["safe_rate"] for r in sel])),
            "timeout_rate": float(np.mean([r["timeout_rate"] for r in sel])),
            "completion_mean_s": float(np.mean([r["completion_mean_s"] for r in sel])),
            "total_wait_mean_s": float(np.mean([r["total_wait_mean_s"] for r in sel])),
            "online_wait_mean_s": float(np.mean([r["online_wait_mean_s"] for r in sel])),
            "coord_mean_ms": float(np.mean([r["coord_mean_ms"] for r in sel])),
            "coord_p95_ms": float(np.mean([r["coord_p95_ms"] for r in sel])),
            "coord_over_budget": int(sum(r["coord_over_budget"] for r in sel)),
            "infeasible_stops": int(sum(r["infeasible_stops"] for r in sel)),
            "mpc_failures": int(sum(r["mpc_failures"] for r in sel)),
        }

    # paired differences by task (same task/p, mean over seeds)
    def paired(a_method, b_method, p, metric):
        a = {r["task_id"]: r[metric] for r in task_rows
             if r["method"] == a_method and abs(r["p"] - p) < 1e-12}
        b = {r["task_id"]: r[metric] for r in task_rows
             if r["method"] == b_method and abs(r["p"] - p) < 1e-12}
        common = sorted(set(a) & set(b))
        return {t: a[t] - b[t] for t in common}

    summary = {
        "stage": "online_temporal_ab_comparison",
        "n_tasks": len(tasks), "n_mc": args.n_mc, "ps": ps,
        "methods": method_names,
        "plan_meta": plan_meta,
        "by_method_p": {m: {str(p): cell(m, p) for p in ps} for m in method_names},
        "paired": {},
        "runtime": {"total_s": time.perf_counter() - t_start},
    }
    for p in ps:
        summary["paired"][str(p)] = {}
        for a, b in (("B_simple_online", "A_w2plus"),
                     ("C_online_firsthit", "B_simple_online"),
                     ("C_online_firsthit", "A_w2plus")):
            summary["paired"][str(p)][f"{a}_minus_{b}"] = {}
            for metric in ("collision_rate", "safe_rate", "completion_mean_s",
                           "total_wait_mean_s"):
                diffs = paired(a, b, p, metric)
                summary["paired"][str(p)][f"{a}_minus_{b}"][metric] = \
                    bootstrap_task_ci(diffs, seed=7)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps({
        "command": " ".join(sys.argv), "config": tcfg,
        "seed0": args.seed0, "n_mc": args.n_mc, "ps": ps,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({m: {str(p): round(cell(m, p)["collision_rate"], 3) for p in ps}
                      for m in method_names}, ensure_ascii=False, indent=1))
    print(f"输出：{out}  {summary['runtime']['total_s']/60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
