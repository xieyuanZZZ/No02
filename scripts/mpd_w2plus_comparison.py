"""W2+ 公平对照：在 20 个 N=4 冻结任务上比较 W0/W1/W2/W2+/W3。

关键设计（对应实施安排 §二）：
  * W2+ 与 W3 使用**完全相同的 28 个等待变量与同一套约束**，区别只在优化器
    （整数坐标下降 vs 可微梯度）；
  * 每个方法**独立计时**，并记录 DP 前向次数与梯度步数；
  * W2+ 提供两个预算：`W2+@120`（与 W3 的 120 步 DP 前向数相近）与
    `W2+@800`（充分搜索）；
  * 所有方法统一用 **Hard First-Hit DP** + 独立 MC 评测；
  * 统计同时按任务配对与按地图聚类。

输出：results/MPD_W2Plus/<run>/（不覆盖既有 final/）
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fhd.evaluation.statistics import paired_bootstrap_diff  # noqa: E402
from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget,
    allowed_positions,
    make_mask,
    policy_w0,
    policy_w1,
    policy_w2_enum,
    policy_w2plus,
    policy_w3_optimize,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

_setup_fonts()
LAM_T, LAM_R, LAM_W = 0.05, 1.0, 0.01
OPT_P = 0.1
METHODS = ("W0", "W1", "W2", "W2+@120", "W2+@800", "W3")


def run_mc(paths, wait, p, d_safe, t_steps, dt_s, n, seed):
    gen = torch.Generator().manual_seed(int(seed))
    pause = torch.full((paths.shape[0],), float(p), dtype=torch.float64)
    sim = simulate_execution(paths, wait, pause, d_safe, t_steps, dt_s,
                             n_samples=n, generator=gen)
    coll = sim.collision.to(torch.float64)
    arrived = sim.all_arrived.to(torch.float64)
    return {
        "collision_rate": float(coll.mean()),
        "safe_completion_rate": float(((1 - coll) * arrived).mean()),
        "all_arrived_rate": float(arrived.mean()),
        "makespan_mean_slots": float(sim.makespan_steps.to(torch.float64).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(ROOT / "data" / "mpd_delay_plan_bank"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--n-mc", type=int, default=300)
    ap.add_argument("--eval-seeds", default="20261130,20261131,20261132")
    ap.add_argument("--ps", default="0,0.05,0.1,0.2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    dt_s = float(cfg["trajectory"]["dt_s"])
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    n_agents = int(args.agents)
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe

    run_dir = Path(args.out) if args.out else new_run_dir("MPD_W2Plus", "w2plus")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.w2plus", run_dir / "logs" / "w2plus.log")
    log_section(logger, "W2+ 公平对照")

    bank = json.loads((Path(args.bank) / f"plan_bank_n{n_agents}.json").read_text(encoding="utf-8"))
    trajs = np.load(Path(args.bank) / f"plan_bank_n{n_agents}_trajs.npz")
    tasks = bank["tasks"]
    ps = [float(x) for x in args.ps.split(",")]
    eval_seeds = [int(x) for x in args.eval_seeds.split(",")]
    logger.info("任务 %d，p=%s，%d 个评测种子 × %d 次 MC", len(tasks), ps,
                len(eval_seeds), args.n_mc)

    budget = WaitBudget(time_steps=T, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), n_agents)
    pause = torch.full((n_agents,), OPT_P, dtype=torch.float64)

    rows = []
    for ti, jt in enumerate(tasks):
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)

        plans = {}
        # 每个方法独立计时
        for name, fn in (
            ("W0", lambda: policy_w0(w0, mask, budget, risk, paths, pause, T)),
            ("W1", lambda: policy_w1(w0, mask, budget, risk, paths, pause, T, 0)),
            ("W2", lambda: policy_w2_enum(w0, mask, budget, risk, paths, pause, T, LAM_T)),
            ("W2+@120", lambda: policy_w2plus(w0, mask, budget, risk, paths, pause, T,
                                              lam_T=LAM_T, max_evals=120, restarts=2,
                                              seed=ti)),
            ("W2+@800", lambda: policy_w2plus(w0, mask, budget, risk, paths, pause, T,
                                              lam_T=LAM_T, max_evals=800, restarts=3,
                                              seed=ti)),
            ("W3", lambda: policy_w3_optimize(w0, mask, budget, risk, paths, pause, T,
                                              lam_R=LAM_R, lam_T=LAM_T, lam_W=LAM_W)),
        ):
            t0 = time.perf_counter()
            r = fn()
            wall = time.perf_counter() - t0
            plans[name] = (r, wall)

        for name, (r, wall) in plans.items():
            for p in ps:
                mets = [run_mc(paths, r.wait, p, d_safe, T, dt_s, args.n_mc,
                               seed=s * 7000 + ti) for s in eval_seeds]
                agg = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
                rows.append({
                    "task_id": jt["task_id"], "map_id": jt["map_id"], "kind": jt["kind"],
                    "method": name, "p": p,
                    "hard_risk": r.hard_risk, "soft_risk": r.soft_risk,
                    "added_slots": float(r.added.sum()),
                    "opt_s": wall, "dp_evals": r.dp_evals, "grad_steps": r.grad_steps,
                    "p0_safe": bool(r.p0_safe), "fallback": bool(r.fallback),
                    **agg,
                })
        if (ti + 1) % 5 == 0:
            logger.info("  %d/%d 任务完成（W2+@800 用时 %.2fs，dp_evals=%d）",
                        ti + 1, len(tasks), plans["W2+@800"][1],
                        plans["W2+@800"][0].dp_evals)

    # ---------------- 汇总 ----------------
    def summarize(p):
        sel = [r for r in rows if abs(r["p"] - p) < 1e-9]
        out = {}
        for m in METHODS:
            s = [r for r in sel if r["method"] == m]
            if not s:
                continue
            out[m] = {
                "collision_rate": float(np.mean([r["collision_rate"] for r in s])),
                "safe_completion_rate": float(np.mean([r["safe_completion_rate"] for r in s])),
                "makespan_mean_slots": float(np.mean([r["makespan_mean_slots"] for r in s])),
                "added_slots_mean": float(np.mean([r["added_slots"] for r in s])),
                "opt_s_mean": float(np.mean([r["opt_s"] for r in s])),
                "dp_evals_mean": float(np.mean([r["dp_evals"] for r in s])),
                "grad_steps_mean": float(np.mean([r["grad_steps"] for r in s])),
                "p0_safe_rate": float(np.mean([r["p0_safe"] for r in s])),
            }
        return out

    summary = {"tasks": len(tasks), "by_p": {str(p): summarize(p) for p in ps}}

    def paired(a, b, key="safe_completion_rate", p=OPT_P, cluster="task_id"):
        da = {r[cluster]: r[key] for r in rows if r["method"] == a and abs(r["p"] - p) < 1e-9}
        db = {r[cluster]: r[key] for r in rows if r["method"] == b and abs(r["p"] - p) < 1e-9}
        keys = sorted(set(da) & set(db))
        return paired_bootstrap_diff(np.array([da[k] for k in keys]),
                                     np.array([db[k] for k in keys]),
                                     np.array(keys), n_boot=2000)

    comps = {}
    for a, b in (("W3", "W2+@800"), ("W3", "W2+@120"), ("W2+@800", "W2+@120"),
                 ("W3", "W2"), ("W2+@800", "W2"), ("W3", "W0"), ("W2+@800", "W0")):
        comps[f"{a}_vs_{b}"] = {
            "by_task": paired(a, b, cluster="task_id"),
            "by_map": paired(a, b, cluster="map_id"),
        }
    summary["paired_p0.1"] = comps

    for k, v in comps.items():
        bt, bm = v["by_task"], v["by_map"]
        logger.info("%-16s 任务级 %+.4f [%+.4f,%+.4f] %s | 地图级 %+.4f [%+.4f,%+.4f] %s",
                    k, bt["diff"], bt["lo"], bt["hi"], "显著" if bt["significant"] else "不显著",
                    bm["diff"], bm["lo"], bm["hi"], "显著" if bm["significant"] else "不显著")

    def dump(name, rs):
        if not rs:
            return
        with (run_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)

    dump("w2plus_eval.csv", rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    colors = {"W0": "#4c72b0", "W1": "#9467bd", "W2": "#55a868",
              "W2+@120": "#ff7f0e", "W2+@800": "#d62728", "W3": "#111111"}
    for m in METHODS:
        ys = [summary["by_p"][str(p)][m]["collision_rate"] for p in ps]
        axes[0].plot(ps, ys, "o-", color=colors[m], label=m)
    axes[0].set_xlabel("暂停概率 p")
    axes[0].set_ylabel("碰撞率")
    axes[0].set_title("碰撞率 vs p（20 个 N=4 任务）", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    ms = [summary["by_p"][str(OPT_P)][m]["makespan_mean_slots"] for m in METHODS]
    cs = [summary["by_p"][str(OPT_P)][m]["collision_rate"] * 100 for m in METHODS]
    ax2 = axes[1]
    ax2.scatter(ms, cs, s=90, c=[colors[m] for m in METHODS])
    for m, x, y in zip(METHODS, ms, cs):
        ax2.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xlabel("makespan（时隙，p=0.1）")
    ax2.set_ylabel("碰撞率 [%]")
    ax2.set_title("安全—效率权衡", fontsize=10)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "w2plus_comparison.png", dpi=130)
    plt.close(fig)

    print(json.dumps({
        "p0.1": {m: {"collision": round(summary["by_p"]["0.1"][m]["collision_rate"], 4),
                     "makespan": round(summary["by_p"]["0.1"][m]["makespan_mean_slots"], 1),
                     "opt_s": round(summary["by_p"]["0.1"][m]["opt_s_mean"], 3),
                     "dp_evals": round(summary["by_p"]["0.1"][m]["dp_evals_mean"], 1)}
                 for m in METHODS if m in summary["by_p"]["0.1"]},
        "W3_vs_W2+800": {k: round(v["by_task"]["diff"], 4)
                         for k, v in comps.items() if k == "W3_vs_W2+@800"},
    }, ensure_ascii=False, indent=2))
    print(f"输出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
