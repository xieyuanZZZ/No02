"""MPC 动力学执行 vs 理想离散执行（E0–E3），以及 First-Hit 预测误差。

E0 原始等待 + 理想离散执行（D1）
E1 W2+ 等待 + 理想离散执行（D1）
E2 原始等待 + MPC 动力学执行（D2）
E3 W2+ 等待 + MPC 动力学执行（D2）

同一任务、同一等待计划、同一随机扰动序列（种子）配对比较。
D1 与 D2 是不同执行模型，差异不能归因为算法优劣。

输出：results/MPD_MPC_Execution/{02_two_robot,03_four_robot,04_risk_mismatch}/
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
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget,
    _eval_hard,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n2-tasks", type=int, default=8)
    ap.add_argument("--n4-tasks", type=int, default=8)
    ap.add_argument("--n-mc", type=int, default=10)
    ap.add_argument("--ps", default="0,0.1,0.2")
    ap.add_argument("--seed0", type=int, default=20270101)
    ap.add_argument("--budget", type=float, default=12.0)
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "base.yaml")
    set_seed(int(cfg["seed"]))
    dt_s = float(cfg["trajectory"]["dt_s"])
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe

    run_dir = Path(args.out) if args.out else new_run_dir("MPD_MPC_Execution", "mpc_exec")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.mpc_bench", run_dir / "logs" / "bench.log")
    log_section(logger, "MPC 动力学执行基准")

    ps = [float(x) for x in args.ps.split(",")]
    rows = []
    t_start = time.perf_counter()

    for n_agents, n_tasks, bank_json, trajs_npz, tag in (
        (2, args.n2_tasks, "plan_bank.json", "plan_bank_trajs.npz", "N2"),
        (4, args.n4_tasks, "plan_bank_n4.json", "plan_bank_n4_trajs.npz", "N4"),
    ):
        bank = json.loads((ROOT / "data" / "mpd_delay_plan_bank" / bank_json)
                          .read_text(encoding="utf-8"))
        trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / trajs_npz)
        budget = WaitBudget(time_steps=T, horizon=H)
        mask = make_mask(H, allowed_positions(H, 5), n_agents)
        pause01 = torch.full((n_agents,), 0.1, dtype=torch.float64)

        for ti, jt in enumerate(bank["tasks"][:n_tasks]):
            paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
            w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
            r0 = policy_w0(w0, mask, budget, risk, paths, pause01, T)
            r2 = policy_w2plus(w0, mask, budget, risk, paths, pause01, T,
                               lam_T=0.05, max_evals=300, restarts=2, seed=ti)
            plans = {
                "orig": (r0.wait.numpy(), r0.hard_risk),
                "w2plus": (r2.wait.numpy(), r2.hard_risk),
            }
            paths_np = paths.numpy()

            for plan_name, (w_np, dp_hard) in plans.items():
                for p in ps:
                    ideal, mpc = [], []
                    for k in range(args.n_mc):
                        seed = args.seed0 + ti * 100 + k
                        # D1 理想
                        gen = torch.Generator().manual_seed(seed)
                        sim = simulate_execution(paths, torch.tensor(w_np, dtype=torch.float64),
                                                 torch.full((n_agents,), p, dtype=torch.float64),
                                                 d_safe, T, dt_s, n_samples=1, generator=gen)
                        ideal.append({
                            "collision": bool(sim.collision.any()),
                            "safe": bool((~sim.collision & sim.all_arrived).all()),
                            "completion": float(sim.makespan_steps[0]) * dt_s,
                        })
                        # D2 MPC
                        m = simulate_mpc(paths_np, w_np, p,
                                         MPCSimConfig(d_safe=d_safe, horizon_steps=T,
                                                      pause_fixed_s=0.4, pause_mode="fixed"),
                                         seed=seed, record=False)
                        mpc.append({
                            "collision": bool(m.collision),
                            "safe": bool((not m.collision) and m.all_arrived),
                            "completion": float(m.completion_time),
                            "brake": float(m.extra_brake_time),
                            "rmse": float(m.tracking_rmse),
                            "max_v": float(m.max_speed),
                            "max_a": float(m.max_accel),
                            "viol": int(m.constraint_violations),
                            "fails": int(m.mpc_failures),
                            "pauses": int(m.pause_count),
                        })
                    rows.append({
                        "group": tag, "task_id": jt["task_id"], "n_agents": n_agents,
                        "plan": plan_name, "p": p, "n_mc": args.n_mc,
                        "dp_hard_risk": dp_hard,
                        "ideal_collision": float(np.mean([x["collision"] for x in ideal])),
                        "ideal_safe": float(np.mean([x["safe"] for x in ideal])),
                        "ideal_completion": float(np.mean([x["completion"] for x in ideal])),
                        "mpc_collision": float(np.mean([x["collision"] for x in mpc])),
                        "mpc_safe": float(np.mean([x["safe"] for x in mpc])),
                        "mpc_completion": float(np.mean([x["completion"] for x in mpc])),
                        "extra_brake_s": float(np.mean([x["brake"] for x in mpc])),
                        "tracking_rmse": float(np.mean([x["rmse"] for x in mpc])),
                        "max_speed": float(np.max([x["max_v"] for x in mpc])),
                        "max_accel": float(np.max([x["max_a"] for x in mpc])),
                        "constraint_violations": int(np.sum([x["viol"] for x in mpc])),
                        "mpc_failures": int(np.sum([x["fails"] for x in mpc])),
                        "pauses": float(np.mean([x["pauses"] for x in mpc])),
                    })
            logger.info("%s %s 完成（%d 任务 × %d p × %d MC）", tag, jt["task_id"],
                        len(ps), len(plans), args.n_mc)

    total = time.perf_counter() - t_start
    with (run_dir / "mpc_execution.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 汇总
    def agg(sel):
        return {
            "ideal_collision": float(np.mean([r["ideal_collision"] for r in sel])),
            "ideal_safe": float(np.mean([r["ideal_safe"] for r in sel])),
            "ideal_completion": float(np.mean([r["ideal_completion"] for r in sel])),
            "mpc_collision": float(np.mean([r["mpc_collision"] for r in sel])),
            "mpc_safe": float(np.mean([r["mpc_safe"] for r in sel])),
            "mpc_completion": float(np.mean([r["mpc_completion"] for r in sel])),
            "extra_brake_s": float(np.mean([r["extra_brake_s"] for r in sel])),
            "tracking_rmse": float(np.mean([r["tracking_rmse"] for r in sel])),
            "mpc_failures": int(np.sum([r["mpc_failures"] for r in sel])),
            "constraint_violations": int(np.sum([r["constraint_violations"] for r in sel])),
        }

    summary = {"n_mc": args.n_mc, "total_time_s": total, "by_group": {}}
    for tag in ("N2", "N4"):
        summary["by_group"][tag] = {}
        for plan in ("orig", "w2plus"):
            summary["by_group"][tag][plan] = {
                str(p): agg([r for r in rows if r["group"] == tag and r["plan"] == plan
                             and abs(r["p"] - p) < 1e-9]) for p in ps}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for tag, ls in (("N2", "-"), ("N4", "--")):
        for plan, c in (("orig", "#4c72b0"), ("w2plus", "#dd8452")):
            y1 = [summary["by_group"][tag][plan][str(p)]["ideal_collision"] for p in ps]
            y2 = [summary["by_group"][tag][plan][str(p)]["mpc_collision"] for p in ps]
            axes[0].plot(ps, y1, ls, marker="o", c=c, label=f"{tag} {plan} 理想")
            axes[1].plot(ps, y2, ls, marker="s", c=c, label=f"{tag} {plan} MPC")
    axes[0].set_title("理想离散执行（D1）碰撞率", fontsize=10)
    axes[1].set_title("MPC 动力学执行（D2）碰撞率", fontsize=10)
    for ax in axes:
        ax.set_xlabel("暂停概率 p")
        ax.set_ylabel("碰撞率")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "mpc_vs_ideal.png", dpi=130)
    plt.close(fig)

    print(json.dumps({g: {pl: {p: round(v["mpc_collision"], 3)
                               for p, v in summary["by_group"][g][pl].items()}
                          for pl in ("orig", "w2plus")} for g in ("N2", "N4")},
                     ensure_ascii=False, indent=1))
    print(f"输出：{run_dir}  总耗时 {total/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
