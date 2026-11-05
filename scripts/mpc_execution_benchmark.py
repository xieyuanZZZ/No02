"""阶段一 D1/D2 可信执行基准（Online_FirstHit_v2/01、02）。

E0 原始等待 + 理想离散执行（D1）
E1 W2+ 等待 + 理想离散执行（D1）
E2 原始等待 + MPC 动力学执行（D2）
E3 W2+ 等待 + MPC 动力学执行（D2）

同一任务、同一等待计划、同一种子配对比较。D1 与 D2 是不同随机执行模型，
差异不能归因为算法优劣；两者的事件语义也不同（见 run_meta.json）。

产物（每个输出目录）：
  config.json      完整参数（含 CLI、基线配置快照）
  run_meta.json    git SHA、依赖版本、硬件、耗时、事件语义
  per_run.csv      每条 MC 一行（D1/D2 指标并列）
  per_task.csv     每任务×计划×p 聚合（含 Wilson / 任务级 bootstrap 区间）
  summary.json     分组汇总与配对比较
  figures/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.geometry.static_maps import load_obstacles_for_map  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget,
    allowed_positions,
    make_mask,
    policy_w0,
    policy_w2plus,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

_setup_fonts()
BANK_FILES = {
    2: ("plan_bank.json", "plan_bank_trajs.npz"),
    4: ("plan_bank_n4.json", "plan_bank_n4_trajs.npz"),
}


# --------------------------------------------------------------------------- #
# 统计工具
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(center - half, 0.0)), float(min(center + half, 1.0)))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return None


def _meta(args, cfg, elapsed: float) -> dict:
    import osqp
    import scipy
    return {
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": elapsed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "osqp": osqp.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "base_config": cfg,
        "event_semantics": {
            "D1": "每个时隙对每台机器人独立抽一次暂停硬币（torch.Generator, 同一 seed）",
            "D2": "同一 seed 预生成 (slot, robot, duration) 事件表，在执行器停稳后计时",
            "note": "两种执行模型使用同一随机种子，但扰动语义不同，不能宣称物理扰动完全相同",
        },
        "static_obstacles": "按 map_id 从 data/revised_s1/maps_*.json 载入；载入不到时标 not_evaluated",
    }


def _git_branch() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return None


def load_tasks(n_agents: int):
    bank_json, trajs_npz = BANK_FILES[n_agents]
    bank = json.loads((ROOT / "data" / "mpd_delay_plan_bank" / bank_json)
                      .read_text(encoding="utf-8"))
    trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / trajs_npz)
    return bank["tasks"], trajs


def load_obstacles_for_tasks(tasks) -> dict:
    cache = {}
    for t in tasks:
        mid = str(t.get("map_id", ""))
        if mid not in cache:
            cache[mid] = load_obstacles_for_map(mid)
    return cache


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n2-tasks", type=int, default=8)
    ap.add_argument("--n4-tasks", type=int, default=8)
    ap.add_argument("--n-mc", type=int, default=10)
    ap.add_argument("--ps", default="0,0.1,0.2")
    ap.add_argument("--seed0", type=int, default=20270101)
    ap.add_argument("--budget", type=float, default=12.0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "base.yaml")
    set_seed(int(cfg["seed"]))
    dt_s = float(cfg["trajectory"]["dt_s"])
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe
    ps = [float(x) for x in args.ps.split(",")]

    run_dir = Path(args.out) if args.out else new_run_dir("Online_FirstHit_v2", "01_mpc_benchmark")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.mpc_bench_v2", run_dir / "logs" / "bench.log")
    log_section(logger, "阶段一 MPC 动力学执行基准 v2")

    config = {
        "stage": "phase1_mpc_benchmark_v2",
        "d_safe": d_safe,
        "horizon_steps": T,
        "horizon_waypoints": H,
        "dt_s": dt_s,
        "ps": ps,
        "n_mc": args.n_mc,
        "seed0": args.seed0,
        "n2_tasks": args.n2_tasks,
        "n4_tasks": args.n4_tasks,
        "mpc": {
            "dt_ctrl": 0.1, "dt_dyn": 0.05, "dt_slot": 0.4,
            "horizon": 18, "v_max": 1.5, "a_max": 2.0, "a_brake": 2.0,
            "pause_mode": "fixed", "pause_fixed_s": 0.4,
            "collision": "continuous per dt_dyn substep",
        },
        "plans": ["orig", "w2plus"],
        "w2plus": {"lam_T": 0.05, "max_evals": 300, "restarts": 2},
        "base_config": cfg,
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    rows = []
    t_start = time.perf_counter()
    for n_agents, n_tasks, tag in ((2, args.n2_tasks, "N2"), (4, args.n4_tasks, "N4")):
        tasks, trajs = load_tasks(n_agents)
        tasks = tasks[:n_tasks]
        obs_cache = load_obstacles_for_tasks(tasks)
        budget = WaitBudget(time_steps=T, horizon=H)
        mask = make_mask(H, allowed_positions(H, 5), n_agents)
        pause01 = torch.full((n_agents,), 0.1, dtype=torch.float64)

        for ti, jt in enumerate(tasks):
            paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
            paths_np = paths.numpy()
            w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
            obs, obs_meta = obs_cache[str(jt.get("map_id", ""))]
            r0 = policy_w0(w0, mask, budget, risk, paths, pause01, T)
            r2 = policy_w2plus(w0, mask, budget, risk, paths, pause01, T,
                               lam_T=0.05, max_evals=300, restarts=2, seed=ti)
            plans = {"orig": (r0.wait.numpy(), float(r0.hard_risk)),
                     "w2plus": (r2.wait.numpy(), float(r2.hard_risk))}

            for plan_name, (w_np, dp_risk) in plans.items():
                for p in ps:
                    for k in range(args.n_mc):
                        seed = args.seed0 + ti * 1000 + k
                        gen = torch.Generator().manual_seed(seed)
                        sim = simulate_execution(
                            paths, torch.tensor(w_np, dtype=torch.float64),
                            torch.full((n_agents,), p, dtype=torch.float64),
                            d_safe, T, dt_s, n_samples=1, generator=gen)
                        d1_collision = bool(sim.collision[0])
                        d1_arrived = bool(sim.all_arrived[0])
                        d1_completion = float(sim.makespan_steps[0]) * dt_s
                        d1_min = float(sim.pair_min_distance[0].min())

                        m = simulate_mpc(paths_np, w_np, p,
                                         MPCSimConfig(d_safe=d_safe, horizon_steps=T,
                                                      pause_fixed_s=0.4, pause_mode="fixed"),
                                         seed=seed, record=False, obstacles=obs,
                                         obstacle_meta=obs_meta)
                        rows.append({
                            "group": tag, "task_id": jt["task_id"],
                            "map_id": jt.get("map_id", ""), "kind": jt.get("kind", ""),
                            "n_agents": n_agents, "plan": plan_name, "p": p, "seed": seed,
                            "dp_hard_risk": float(dp_risk),
                            "d1_collision": d1_collision,
                            "d1_safe": bool((not d1_collision) and d1_arrived),
                            "d1_completion": d1_completion,
                            "d1_min_pair_distance": d1_min,
                            "d2_collision": bool(m.collision),
                            "d2_team_collision": bool(m.collision),
                            "d2_first_hit_time": m.first_collision_time,
                            "d2_min_pair_distance": float(np.nanmin(m.pair_min_distance)),
                            "d2_all_arrived": bool(m.all_arrived),
                            "d2_timeout": bool(m.timeout),
                            "d2_censored": bool(m.time_censored),
                            "d2_safe_completion": bool(m.safe_completion),
                            "d2_completion": float(m.completion_time),
                            "d2_infeasible_stops": int(m.infeasible_stops),
                            "d2_constraint_violations": int(m.constraint_violations),
                            "d2_mpc_failures": int(m.mpc_failures),
                            "d2_solve_mean_ms": float(m.solve_stats.get("mean_ms", 0.0)),
                            "d2_solve_p95_ms": float(m.solve_stats.get("p95_ms", 0.0)),
                            "d2_solve_p99_ms": float(m.solve_stats.get("p99_ms", 0.0)),
                            "d2_n_solves": int(m.solve_stats.get("n_solves", 0)),
                            "d2_solve_statuses": json.dumps(
                                m.solve_stats.get("statuses", {}), ensure_ascii=False),
                            "d2_tracking_rmse": float(m.tracking_rmse),
                            "d2_tracking_max": float(m.tracking_max),
                            "d2_max_speed": float(m.max_speed),
                            "d2_max_accel": float(m.max_accel),
                            "d2_pause_requests": int(m.pause_requests),
                            "d2_pause_accepted": int(m.pause_accepted),
                            "d2_pause_rejected": int(m.pause_rejected),
                            "d2_actual_wait_s": float(m.actual_wait_time),
                            "d2_random_hold_s": float(m.random_hold_time),
                            "d2_extra_brake_s": float(m.extra_brake_time),
                            "d2_obstacle_collision": m.obstacle_collision,
                            "d2_min_obstacle_clearance": m.min_obstacle_clearance,
                            "d2_static_status": m.static_obstacle_status,
                        })
            logger.info("[%s] %s 完成：%d plan × %d p × %d MC", tag, jt["task_id"],
                        len(plans), len(ps), args.n_mc)

    total = time.perf_counter() - t_start
    if not rows:
        print("没有产生任何结果", file=sys.stderr)
        return 1

    fieldnames = list(rows[0].keys())
    with (run_dir / "per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---------------- 每任务聚合（Wilson + 任务级 bootstrap） ----------------
    keys = sorted({(r["group"], r["task_id"], r["plan"], r["p"]) for r in rows})
    task_rows = []
    for (g, tid, plan, p) in keys:
        sel = [r for r in rows if r["group"] == g and r["task_id"] == tid
               and r["plan"] == plan and abs(r["p"] - p) < 1e-12]
        n = len(sel)
        d2_c = sum(r["d2_collision"] for r in sel)
        d2_s = sum(r["d2_safe_completion"] for r in sel)
        d1_c = sum(r["d1_collision"] for r in sel)
        d2_c_lo, d2_c_hi = wilson_interval(d2_c, n)
        d2_s_lo, d2_s_hi = wilson_interval(d2_s, n)
        task_rows.append({
            "group": g, "task_id": tid, "map_id": sel[0]["map_id"], "kind": sel[0]["kind"],
            "plan": plan, "p": p, "n_mc": n,
            "dp_hard_risk": float(np.mean([r["dp_hard_risk"] for r in sel])),
            "d1_collision_rate": float(np.mean([r["d1_collision"] for r in sel])),
            "d1_safe_rate": float(np.mean([r["d1_safe"] for r in sel])),
            "d2_collision_rate": d2_c / n, "d2_collision_ci_lo": d2_c_lo,
            "d2_collision_ci_hi": d2_c_hi,
            "d2_safe_rate": d2_s / n, "d2_safe_ci_lo": d2_s_lo, "d2_safe_ci_hi": d2_s_hi,
            "d2_timeout_rate": float(np.mean([r["d2_timeout"] for r in sel])),
            "d2_infeasible_stops": int(sum(r["d2_infeasible_stops"] for r in sel)),
            "d2_constraint_violations": int(sum(r["d2_constraint_violations"] for r in sel)),
            "d2_mpc_failures": int(sum(r["d2_mpc_failures"] for r in sel)),
            "d2_solve_mean_ms": float(np.mean([r["d2_solve_mean_ms"] for r in sel])),
            "d2_solve_p95_ms": float(np.mean([r["d2_solve_p95_ms"] for r in sel])),
            "d2_tracking_rmse": float(np.mean([r["d2_tracking_rmse"] for r in sel])),
            "d2_completion_mean_s": float(np.mean([r["d2_completion"] for r in sel])),
            "d2_min_pair_distance": float(np.min([r["d2_min_pair_distance"] for r in sel])),
            "d2_obstacle_collision_rate": (
                float(np.mean([bool(r["d2_obstacle_collision"])
                               for r in sel if r["d2_obstacle_collision"] is not None]))
                if any(r["d2_obstacle_collision"] is not None for r in sel) else None),
            "d2_static_status": sel[0]["d2_static_status"],
        })
    with (run_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(task_rows[0].keys()))
        w.writeheader()
        w.writerows(task_rows)

    # ---------------- 分组汇总 ----------------
    def cell(g, plan, p):
        sel = [r for r in task_rows if r["group"] == g and r["plan"] == plan
               and abs(r["p"] - p) < 1e-12]
        vals = np.array([r["d2_collision_rate"] for r in sel])
        lo, hi = bootstrap_mean_ci(vals, seed=7)
        d1 = np.array([r["d1_collision_rate"] for r in sel])
        safe = np.array([r["d2_safe_rate"] for r in sel])
        return {
            "n_tasks": len(sel), "n_mc_per_task": args.n_mc,
            "d1_collision_rate": float(d1.mean()) if len(sel) else None,
            "d2_collision_rate": float(vals.mean()) if len(sel) else None,
            "d2_collision_ci_lo": lo, "d2_collision_ci_hi": hi,
            "d2_collision_pooled": (sum(r["d2_collision_rate"] * r["n_mc"] for r in sel)
                                    / max(sum(r["n_mc"] for r in sel), 1)) if sel else None,
            "d2_safe_rate": float(safe.mean()) if len(sel) else None,
            "d2_timeout_rate": float(np.mean([r["d2_timeout_rate"] for r in sel])) if sel else None,
            "d2_infeasible_stops": int(sum(r["d2_infeasible_stops"] for r in sel)),
            "d2_constraint_violations": int(sum(r["d2_constraint_violations"] for r in sel)),
            "d2_mpc_failures": int(sum(r["d2_mpc_failures"] for r in sel)),
            "d2_solve_mean_ms": float(np.mean([r["d2_solve_mean_ms"] for r in sel])) if sel else None,
            "d2_solve_p95_ms": float(np.mean([r["d2_solve_p95_ms"] for r in sel])) if sel else None,
            "d2_tracking_rmse": float(np.mean([r["d2_tracking_rmse"] for r in sel])) if sel else None,
            "d2_min_pair_distance": float(np.min([r["d2_min_pair_distance"] for r in sel])) if sel else None,
        }

    summary = {
        "stage": "phase1_mpc_benchmark_v2",
        "n_mc": args.n_mc, "ps": ps, "d_safe": d_safe,
        "n_tasks": {"N2": args.n2_tasks, "N4": args.n4_tasks},
        "total_time_s": total,
        "by_group": {g: {plan: {str(p): cell(g, plan, p) for p in ps}
                         for plan in ("orig", "w2plus")} for g in ("N2", "N4")},
    }
    # 配对比较：同一任务、p 下 W2+ - orig 的 D2 碰撞率差
    summary["paired_w2plus_minus_orig"] = {}
    for g in ("N2", "N4"):
        summary["paired_w2plus_minus_orig"][g] = {}
        for p in ps:
            o = {r["task_id"]: r["d2_collision_rate"] for r in task_rows
                 if r["group"] == g and r["plan"] == "orig" and abs(r["p"] - p) < 1e-12}
            w2 = {r["task_id"]: r["d2_collision_rate"] for r in task_rows
                  if r["group"] == g and r["plan"] == "w2plus" and abs(r["p"] - p) < 1e-12}
            common = sorted(set(o) & set(w2))
            diffs = np.array([w2[t] - o[t] for t in common])
            lo, hi = bootstrap_mean_ci(diffs, seed=11) if diffs.size else (float("nan"), float("nan"))
            summary["paired_w2plus_minus_orig"][g][str(p)] = {
                "n_tasks": len(common), "mean_diff": float(diffs.mean()) if diffs.size else None,
                "ci_lo": lo, "ci_hi": hi,
            }

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (run_dir / "run_meta.json").write_text(
        json.dumps(_meta(args, cfg, total), ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for tag, ls in (("N2", "-"), ("N4", "--")):
        for plan, c in (("orig", "#4c72b0"), ("w2plus", "#dd8452")):
            ax = axes[0]
            y1 = [summary["by_group"][tag][plan][str(p)]["d1_collision_rate"] for p in ps]
            y2 = [summary["by_group"][tag][plan][str(p)]["d2_collision_rate"] for p in ps]
            ax.plot(ps, y1, ls, marker="o", c=c, alpha=0.5)
            ax.plot(ps, y2, ls, marker="s", c=c, label=f"{tag} {plan}")
            axes[1].plot(ps, [summary["by_group"][tag][plan][str(p)]["d2_safe_rate"]
                              for p in ps], ls, marker="o", c=c)
            axes[2].plot(ps, [summary["by_group"][tag][plan][str(p)]["d2_solve_mean_ms"]
                              for p in ps], ls, marker="o", c=c)
    axes[0].set_title("碰撞率：虚线 D1 / 实线 D2")
    axes[1].set_title("安全完成率（无碰撞且全部到达）")
    axes[2].set_title("MPC 平均求解耗时（全程调用）")
    for ax in axes:
        ax.set_xlabel("暂停概率 p")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("碰撞率")
    axes[1].set_ylabel("安全完成率")
    axes[2].set_ylabel("ms")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "execution_benchmark.png", dpi=130)
    plt.close(fig)

    print(json.dumps({g: {pl: {p: round(v["d2_collision_rate"], 3)
                               for p, v in summary["by_group"][g][pl].items()}
                          for pl in ("orig", "w2plus")} for g in ("N2", "N4")},
                     ensure_ascii=False, indent=1))
    print(f"输出：{run_dir}  总耗时 {total/60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
