"""高延迟（p=0.2）鲁棒性研究：优化目标偏差 / 等待预算 / 剩余碰撞诊断。

三个实验（对应实施安排 §三）：
  A. 改变**优化时使用的暂停概率**：A0(p=0.1) / A1(p=0.2) / A2(多 p 加权)
  B. 改变**等待预算**：8 / 12 / 16 / 24 时隙（每机器人）
  C. 剩余碰撞诊断：p=0.2 下仍碰撞的任务，输出轨迹、等待、首次碰撞、预算饱和

所有方案统一在 p ∈ {0.05, 0.1, 0.2} 下用**独立评测种子**做 MC。

输出：results/MPD_HighDelay/<run>/（不覆盖既有 final/）
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

from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget,
    _eval_hard,
    allowed_positions,
    make_mask,
    policy_w0,
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


def run_mc(paths, wait, p, d_safe, T, dt_s, n, seed, traces=False):
    gen = torch.Generator().manual_seed(int(seed))
    pause = torch.full((paths.shape[0],), float(p), dtype=torch.float64)
    sim = simulate_execution(paths, wait, pause, d_safe, T, dt_s, n_samples=n,
                             generator=gen, record_traces=traces)
    coll = sim.collision.to(torch.float64)
    arrived = sim.all_arrived.to(torch.float64)
    out = {
        "collision_rate": float(coll.mean()),
        "safe_completion_rate": float(((1 - coll) * arrived).mean()),
        "makespan_mean_slots": float(sim.makespan_steps.to(torch.float64).mean()),
    }
    if traces:
        out["_sim"] = sim
    return out


def multi_p_risk(risk_model, paths, wait, ps, T, lam_T, base_finish, mode="mean"):
    """多暂停概率的联合软风险（A2 用）。"""
    vals = []
    for p in ps:
        pause = torch.full((paths.shape[0],), float(p), dtype=torch.float64)
        r = risk_model(paths, wait, pause, T, collision_mode="soft", gate_mode="soft",
                       return_diagnostics=False)
        vals.append(r["pair_sum_risk"] + lam_T * (r["finish_cost"] - base_finish))
    stack = torch.stack(vals)
    return stack.max() if mode == "max" else stack.mean()


def optimize_w3_multi_p(w0, mask, budget, risk_model, paths, T, opt_ps,
                        lam_T=0.05, lam_W=0.01, steps=120, lr=0.15, mode="mean"):
    """A2：在多个暂停概率上联合优化（软风险加权/最坏）。"""
    import torch.nn.functional as F

    t0 = time.perf_counter()
    zero = torch.zeros(paths.shape[0], dtype=torch.float64)
    base_finish = _eval_hard(risk_model, paths, w0, torch.full((paths.shape[0],), opt_ps[0],
                                                               dtype=torch.float64), T)[1]
    u = torch.zeros_like(w0, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        w = project(w0 + F.softplus(u) * mask, w0, mask, budget)
        loss = multi_p_risk(risk_model, paths, w, opt_ps, T, lam_T, base_finish, mode) \
            + lam_W * (w - w0).abs().sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w_c = project(w0 + F.softplus(u) * mask, w0, mask, budget)
        w_d = torch.round(w_c - w0) + w0
        w_d = project(w_d, w0, mask, budget).round()
        h0, f0 = _eval_hard(risk_model, paths, w_d, zero, T)
        hard, fin = _eval_hard(risk_model, paths, w_d,
                               torch.full((paths.shape[0],), 0.1, dtype=torch.float64), T)
    ok = h0 <= 1e-9
    return {"wait": w_d, "added": float((w_d - w0).sum()), "hard_p01": hard,
            "finish": fin, "p0_safe": bool(ok), "elapsed": time.perf_counter() - t0,
            "dp_evals": steps + 2, "grad_steps": steps}


def project(w, w0, mask, budget):
    from fhd.optim.wait_policies import project_wait
    return project_wait(w, w0, mask, budget)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(ROOT / "data" / "mpd_delay_plan_bank"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--n-mc", type=int, default=300)
    ap.add_argument("--eval-seeds", default="20261130,20261131,20261132")
    ap.add_argument("--ps", default="0.05,0.1,0.2")
    ap.add_argument("--budgets", default="8,12,16,24")
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

    run_dir = Path(args.out) if args.out else new_run_dir("MPD_HighDelay", "highdelay")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.highdelay", run_dir / "logs" / "highdelay.log")
    log_section(logger, "高延迟 p=0.2 鲁棒性研究")

    bank = json.loads((Path(args.bank) / f"plan_bank_n{n_agents}.json").read_text(encoding="utf-8"))
    trajs = np.load(Path(args.bank) / f"plan_bank_n{n_agents}_trajs.npz")
    tasks = bank["tasks"]
    ps = [float(x) for x in args.ps.split(",")]
    eval_seeds = [int(x) for x in args.eval_seeds.split(",")]
    budgets = [float(x) for x in args.budgets.split(",")]
    logger.info("任务 %d，评测 p=%s，预算 %s，%d 种子 × %d MC",
                len(tasks), ps, budgets, len(eval_seeds), args.n_mc)

    positions = allowed_positions(H, 5)
    rows = []

    def eval_plan(paths, w, name, config, ti):
        for p in ps:
            mets = [run_mc(paths, w, p, d_safe, T, dt_s, args.n_mc, seed=s * 7000 + ti)
                    for s in eval_seeds]
            agg = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
            rows.append({"task_id": tasks[ti]["task_id"], "map_id": tasks[ti]["map_id"],
                         "config": config, "method": name, "p": p, **agg})

    # ---------------- 实验 A：优化概率消融 ----------------
    log_section(logger, "实验 A：改变优化时使用的暂停概率")
    for ti, jt in enumerate(tasks):
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        budget = WaitBudget(time_steps=T, horizon=H)
        mask = make_mask(H, positions, n_agents)
        pause01 = torch.full((n_agents,), 0.1, dtype=torch.float64)
        pause02 = torch.full((n_agents,), 0.2, dtype=torch.float64)

        r_w0 = policy_w0(w0, mask, budget, risk, paths, pause01, T)
        r_a0 = policy_w3_optimize(w0, mask, budget, risk, paths, pause01, T,
                                  lam_R=LAM_R, lam_T=LAM_T, lam_W=LAM_W)
        r_a1 = policy_w3_optimize(w0, mask, budget, risk, paths, pause02, T,
                                  lam_R=LAM_R, lam_T=LAM_T, lam_W=LAM_W)
        a2 = optimize_w3_multi_p(w0, mask, budget, risk, paths, T,
                                 [0.05, 0.1, 0.2], lam_T=LAM_T, lam_W=LAM_W)

        for name, w in (("W0", r_w0.wait), ("A0_p0.1", r_a0.wait),
                        ("A1_p0.2", r_a1.wait), ("A2_multi", a2["wait"])):
            eval_plan(paths, w, name, "A_prob", ti)
        if (ti + 1) % 5 == 0:
            logger.info("  A：%d/%d 任务完成", ti + 1, len(tasks))

    # ---------------- 实验 B：等待预算消融 ----------------
    log_section(logger, "实验 B：等待预算消融")
    for ti, jt in enumerate(tasks):
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        pause01 = torch.full((n_agents,), 0.1, dtype=torch.float64)
        for b in budgets:
            budget = WaitBudget(max_added_total_per_robot=b, time_steps=T, horizon=H)
            mask = make_mask(H, positions, n_agents)
            r = policy_w3_optimize(w0, mask, budget, risk, paths, pause01, T,
                                   lam_R=LAM_R, lam_T=LAM_T, lam_W=LAM_W)
            eval_plan(paths, r.wait, f"W3_b{b:.0f}", "B_budget", ti)
            r2 = policy_w2plus(w0, mask, budget, risk, paths, pause01, T,
                               lam_T=LAM_T, max_evals=300, restarts=2, seed=ti)
            eval_plan(paths, r2.wait, f"W2+_b{b:.0f}", "B_budget", ti)
        if (ti + 1) % 5 == 0:
            logger.info("  B：%d/%d 任务完成", ti + 1, len(tasks))

    # ---------------- 实验 C：剩余碰撞诊断 ----------------
    log_section(logger, "实验 C：p=0.2 剩余碰撞诊断")
    diag = []
    for ti, jt in enumerate(tasks):
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        budget = WaitBudget(time_steps=T, horizon=H)
        mask = make_mask(H, positions, n_agents)
        pause02 = torch.full((n_agents,), 0.2, dtype=torch.float64)
        r = policy_w3_optimize(w0, mask, budget, risk, paths, pause02, T,
                               lam_R=LAM_R, lam_T=LAM_T, lam_W=LAM_W)
        m = run_mc(paths, r.wait, 0.2, d_safe, T, dt_s, args.n_mc, seed=20261200 + ti,
                   traces=True)
        added = (r.wait - w0)
        cap = min(budget.max_added_total_per_robot, budget.max_added_total_allowed())
        saturated = float(added.sum(dim=-1).max()) >= cap - 1e-9
        sim = m.pop("_sim")
        first_pair = sim.first_collision_pair[sim.collision.bool()]
        first_step = sim.first_collision_step[sim.collision.bool()]
        diag.append({
            "task_id": jt["task_id"], "map_id": jt["map_id"],
            "collision_rate_p0.2": m["collision_rate"],
            "added_total": float(added.sum()), "added_max_per_robot": float(added.sum(dim=-1).max()),
            "budget_cap": cap, "budget_saturated": bool(saturated),
            "hard_risk_p0.2": r.hard_risk, "p0_safe": bool(r.p0_safe),
            "fallback": bool(r.fallback),
            "first_collision_step_mean": float(first_step.to(torch.float64).mean())
            if len(first_step) else None,
            "first_collision_pair_mode": int(torch.mode(first_pair.reshape(-1)).values)
            if len(first_pair) else None,
        })
        if m["collision_rate"] > 0.5 and len(diag) <= 6:
            # 保存典型失败任务的轨迹与等待，供画图
            np.savez_compressed(run_dir / f"case_{ti:02d}.npz",
                                paths=paths.numpy(), wait=r.wait.numpy(),
                                w0=w0.numpy(), task_id=np.array([jt["task_id"]]))
    logger.info("  C：完成，%d 个任务碰撞率 > 0.5",
                sum(1 for d in diag if d["collision_rate_p0.2"] > 0.5))

    # ---------------- 汇总 ----------------
    def summarize(config, p):
        sel = [r for r in rows if r["config"] == config and abs(r["p"] - p) < 1e-9]
        out = {}
        for m in sorted({r["method"] for r in sel}):
            s = [r for r in sel if r["method"] == m]
            out[m] = {
                "collision_rate": float(np.mean([r["collision_rate"] for r in s])),
                "safe_completion_rate": float(np.mean([r["safe_completion_rate"] for r in s])),
                "makespan_mean_slots": float(np.mean([r["makespan_mean_slots"] for r in s])),
            }
        return out

    summary = {
        "tasks": len(tasks), "ps": ps,
        "A_prob": {str(p): summarize("A_prob", p) for p in ps},
        "B_budget": {str(p): summarize("B_budget", p) for p in ps},
        "C_diagnosis": diag,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    with (run_dir / "high_delay.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (run_dir / "diagnosis.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diag[0].keys()))
        w.writeheader()
        w.writerows(diag)

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for m in sorted({r["method"] for r in rows if r["config"] == "A_prob"}):
        ys = [summary["A_prob"][str(p)][m]["collision_rate"] for p in ps]
        axes[0].plot(ps, ys, "o-", label=m)
    axes[0].set_xlabel("评测暂停概率 p")
    axes[0].set_ylabel("碰撞率")
    axes[0].set_title("实验 A：优化目标概率的影响", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for m in sorted({r["method"] for r in rows if r["config"] == "B_budget"}):
        ys = [summary["B_budget"][str(p)][m]["collision_rate"] for p in ps]
        axes[1].plot(ps, ys, "s--", label=m)
    axes[1].set_xlabel("评测暂停概率 p")
    axes[1].set_ylabel("碰撞率")
    axes[1].set_title("实验 B：等待预算的影响", fontsize=10)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "high_delay_AB.png", dpi=130)
    plt.close(fig)

    print(json.dumps({
        "A_at_p0.2": {m: round(v["collision_rate"], 4)
                      for m, v in summary["A_prob"]["0.2"].items()},
        "B_at_p0.2": {m: round(v["collision_rate"], 4)
                      for m, v in summary["B_budget"]["0.2"].items()},
    }, ensure_ascii=False, indent=2))
    print(f"输出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
