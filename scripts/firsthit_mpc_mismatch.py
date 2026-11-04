"""First-Hit 风险预测 vs 理想 MC vs MPC MC：误差量化。

对应实施安排 §八。三个对象：
  1. 原始 First-Hit DP 的两两硬风险 R_ij
  2. 理想离散执行（D1）的 MC 两两碰撞频率
  3. MPC 动力学执行（D2）的 MC 两两碰撞频率

注意：`Σ_{i<j} R_ij` **不是**全队碰撞概率，不能直接算 Brier Score；
全队风险单独报告上界（min(1, ΣR)）与 MC 估计。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


def pair_rates(flags, pairs):
    """flags: [K,P] bool（按 pairs 顺序）-> 每对碰撞频率。"""
    return {pairs[k]: float(np.mean(flags[:, k])) for k in range(len(pairs))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n2-tasks", type=int, default=6)
    ap.add_argument("--n4-tasks", type=int, default=6)
    ap.add_argument("--n-mc", type=int, default=20)
    ap.add_argument("--ps", default="0.05,0.1,0.2")
    ap.add_argument("--seed0", type=int, default=20270201)
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "base.yaml")
    set_seed(int(cfg["seed"]))
    dt_s = float(cfg["trajectory"]["dt_s"])
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe

    run_dir = Path(args.out) if args.out else new_run_dir("MPD_MPC_Execution", "mismatch")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.mismatch", run_dir / "logs" / "mismatch.log")
    log_section(logger, "First-Hit 预测误差分析")

    ps = [float(x) for x in args.ps.split(",")]
    rows = []
    for n_agents, n_tasks, bank_json, trajs_npz, tag in (
        (2, args.n2_tasks, "plan_bank.json", "plan_bank_trajs.npz", "N2"),
        (4, args.n4_tasks, "plan_bank_n4.json", "plan_bank_n4_trajs.npz", "N4"),
    ):
        bank = json.loads((ROOT / "data" / "mpd_delay_plan_bank" / bank_json)
                          .read_text(encoding="utf-8"))
        trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / trajs_npz)
        budget = WaitBudget(time_steps=T, horizon=H)
        mask = make_mask(H, allowed_positions(H, 5), n_agents)
        for ti, jt in enumerate(bank["tasks"][:n_tasks]):
            paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
            w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
            paths_np = paths.numpy()
            for plan_name in ("orig", "w2plus"):
                if plan_name == "orig":
                    r = policy_w0(w0, mask, budget, risk, paths,
                                  torch.full((n_agents,), 0.1, dtype=torch.float64), T)
                else:
                    r = policy_w2plus(w0, mask, budget, risk, paths,
                                      torch.full((n_agents,), 0.1, dtype=torch.float64), T,
                                      lam_T=0.05, max_evals=300, restarts=2, seed=ti)
                w_np = r.wait.numpy()
                pairs = [(a, b) for a in range(n_agents) for b in range(a + 1, n_agents)]
                for p in ps:
                    # DP 两两风险
                    with torch.no_grad():
                        dp = risk(paths, r.wait,
                                  torch.full((n_agents,), p, dtype=torch.float64), T,
                                  collision_mode="hard", gate_mode="hard",
                                  return_diagnostics=False)
                    dp_pairs = {pair: float(dp["pair_risks"][k])
                                for k, pair in enumerate(pairs)}
                    ideal_flags, mpc_flags = [], []
                    for k in range(args.n_mc):
                        seed = args.seed0 + ti * 1000 + k
                        gen = torch.Generator().manual_seed(seed)
                        sim = simulate_execution(paths, torch.tensor(w_np, dtype=torch.float64),
                                                 torch.full((n_agents,), p, dtype=torch.float64),
                                                 d_safe, T, dt_s, n_samples=1, generator=gen)
                        pc = (sim.pair_min_distance.numpy()[0] < d_safe).astype(bool)
                        ideal_flags.append(pc)
                        m = simulate_mpc(paths_np, w_np, p,
                                         MPCSimConfig(d_safe=d_safe, horizon_steps=T),
                                         seed=seed, record=False)
                        mpc_flags.append(np.asarray(m.pair_collision, dtype=bool))
                    ideal_flags = np.asarray(ideal_flags)
                    mpc_flags = np.asarray(mpc_flags)
                    ri = pair_rates(ideal_flags, pairs)
                    rm = pair_rates(mpc_flags, pairs)
                    for (i, j), v in dp_pairs.items():
                        rows.append({
                            "group": tag, "task_id": jt["task_id"], "plan": plan_name,
                            "p": p, "pair": f"{i}-{j}",
                            "dp_risk": v, "ideal_rate": ri[(i, j)], "mpc_rate": rm[(i, j)],
                            "n_mc": args.n_mc,
                        })
            logger.info("%s %s 完成", tag, jt["task_id"])

    with (run_dir / "risk_mismatch.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dp = np.array([r["dp_risk"] for r in rows])
    ri = np.array([r["ideal_rate"] for r in rows])
    rm = np.array([r["mpc_rate"] for r in rows])

    def stats(a, b):
        return {
            "mae": float(np.mean(np.abs(a - b))),
            "bias": float(np.mean(a - b)),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "spearman": float(_spearman(a, b)),
            "n": int(len(a)),
        }

    summary = {
        "n_mc": args.n_mc,
        "pairs_total": len(rows),
        "dp_vs_ideal": stats(dp, ri),
        "dp_vs_mpc": stats(dp, rm),
        "ideal_vs_mpc": stats(ri, rm),
        "by_p": {},
    }
    for p in ps:
        sel = [r for r in rows if abs(r["p"] - p) < 1e-9]
        summary["by_p"][str(p)] = {
            "dp_vs_ideal": stats(np.array([r["dp_risk"] for r in sel]),
                                 np.array([r["ideal_rate"] for r in sel])),
            "dp_vs_mpc": stats(np.array([r["dp_risk"] for r in sel]),
                               np.array([r["mpc_rate"] for r in sel])),
            "ideal_vs_mpc": stats(np.array([r["ideal_rate"] for r in sel]),
                                  np.array([r["mpc_rate"] for r in sel])),
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    # 图：预测 vs 实测
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, y, name in ((axes[0], ri, "理想离散 MC"), (axes[1], rm, "MPC 动力学 MC")):
        ax.scatter(dp, y, s=18, alpha=0.7)
        lim = [0, max(1.0, float(max(dp.max(), y.max())) * 1.05)]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("First-Hit DP 两两硬风险")
        ax.set_ylabel(f"{name} 两两碰撞频率")
        ax.set_title(f"DP vs {name}（MAE={np.mean(np.abs(dp-y)):.3f}）", fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "risk_calibration.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("dp_", "ideal_"))},
                     ensure_ascii=False, indent=1))
    print(f"输出：{run_dir}")
    return 0


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    return float(spearmanr(a, b).correlation)


if __name__ == "__main__":
    raise SystemExit(main())
