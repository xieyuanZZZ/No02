"""M1：段通行时间修正的风险预测（对比 M0 原始 First-Hit）。

做法：
  1. 在校准任务上跑 MPC（p=0 与 p=0.1），采集**每台机器人每段**的实际通行时间；
  2. 得到 durations[i, j]（单位：时隙），作为 DP 的进度时间尺度；
  3. M0 = 原始 DP（每段 1 时隙）；M1 = 带 durations 的 DP；
  4. 在**独立测试任务**上，用 MPC MC 的两两碰撞频率评价 M0 与 M1。

注意：校准只用**时间信息**（不接触碰撞结果）；测试任务与校准任务不重叠。
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
from fhd.execution.mpc_simulator import MPCSimConfig, make_pause_events, simulate_mpc  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget, allowed_positions, make_mask, policy_w0, policy_w2plus,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

_setup_fonts()


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def stats(pred, obs):
    pred, obs = np.asarray(pred), np.asarray(obs)
    return {"mae": float(np.mean(np.abs(pred - obs))),
            "bias": float(np.mean(pred - obs)),
            "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
            "spearman": spearman(pred, obs), "n": int(len(pred))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n2-tasks", type=int, default=8)
    ap.add_argument("--n4-tasks", type=int, default=8)
    ap.add_argument("--n-mc", type=int, default=10)
    ap.add_argument("--ps", default="0.1,0.2")
    ap.add_argument("--seed0", type=int, default=20280101)
    ap.add_argument("--m2-draws", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "base.yaml")
    set_seed(int(cfg["seed"]))
    dt_s = float(cfg["trajectory"]["dt_s"])
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe
    ps = [float(x) for x in args.ps.split(",")]

    run_dir = Path(args.out) if args.out else new_run_dir("Online_FirstHit", "m1")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.m1", run_dir / "logs" / "m1.log")
    log_section(logger, "M1 段通行时间修正")

    rows = []
    for n_agents, n_tasks, bank_json, trajs_npz, tag in (
        (2, args.n2_tasks, "plan_bank.json", "plan_bank_trajs.npz", "N2"),
        (4, args.n4_tasks, "plan_bank_n4.json", "plan_bank_n4_trajs.npz", "N4"),
    ):
        bank = json.loads((ROOT / "data" / "mpd_delay_plan_bank" / bank_json)
                          .read_text(encoding="utf-8"))
        trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / trajs_npz)
        tasks = bank["tasks"][:n_tasks]
        half = len(tasks) // 2
        calib, test = tasks[:half], tasks[half:]
        budget = WaitBudget(time_steps=T, horizon=H)
        mask = make_mask(H, allowed_positions(H, 5), n_agents)

        # ---------- 1) 校准：采段通行时间 ----------
        dur_table = {}
        for jt in calib:
            paths = np.asarray(trajs[jt["task_id"]], dtype=np.float64)
            w = np.asarray(jt["baseline_wait"], dtype=np.float64)
            per_robot = []
            for i in range(n_agents):
                dts = []
                for seed in (1, 2):
                    r = simulate_mpc(paths, w, 0.0, MPCSimConfig(d_safe=d_safe,
                                                                 horizon_steps=T),
                                     seed=seed, record=False)
                    segs = r.segment_times[i]
                    dts.append([s["dt"] for s in segs])
                arr = np.array([[d if d is not None else np.nan for d in row] for row in dts],
                               dtype=np.float64)
                med = np.nanmedian(arr, axis=0)
                # 缺失段用全局中位数补
                g = np.nanmedian(med) if np.isfinite(np.nanmedian(med)) else dt_s
                med = np.where(np.isfinite(med), med, g)
                per_robot.append(med / dt_s)          # 换成时隙单位
            dur_table[jt["task_id"]] = np.stack(per_robot)   # [N, H-1]
        logger.info("[%s] 校准任务 %d 个，段时长中位 %.3f 时隙（名义 1.0）", tag, len(calib),
                    float(np.median(np.concatenate([v.reshape(-1) for v in dur_table.values()]))))

        # ---------- 2) 测试：M0 vs M1 vs MPC MC ----------
        for ti, jt in enumerate(test):
            paths_t = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
            w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
            paths_np = np.asarray(trajs[jt["task_id"]], dtype=np.float64)
            # 段时长必须在**有暂停**的运行上标定：p=0 时实测就是 1.000 时隙，
            # 用它做 M1 会退化为 M0（这本身是一个结论）。
            dur_by_p, samples_by_p = {}, {}
            for p_ in ps:
                acc = []
                for seed in (11, 12, 13, 14, 15, 16):
                    r_ = simulate_mpc(paths_np, w0.numpy(), p_,
                                      MPCSimConfig(d_safe=d_safe, horizon_steps=T),
                                      seed=seed, record=False)
                    acc.append(np.array([[s["dt"] if s["dt"] is not None else np.nan
                                          for s in r_.segment_times[i]]
                                         for i in range(n_agents)], dtype=np.float64))
                stack = np.stack(acc)                       # [R, N, H-1]
                gmed = np.nanmedian(stack) if np.isfinite(np.nanmedian(stack)) else dt_s
                stack = np.where(np.isfinite(stack), stack, gmed) / dt_s
                samples_by_p[p_] = stack
                dur_by_p[p_] = torch.tensor(np.nanmean(stack, axis=0), dtype=torch.float64)

            for plan_name in ("orig", "w2plus"):
                if plan_name == "orig":
                    rr = policy_w0(w0, mask, budget, risk, paths_t,
                                   torch.full((n_agents,), 0.1, dtype=torch.float64), T)
                else:
                    rr = policy_w2plus(w0, mask, budget, risk, paths_t,
                                       torch.full((n_agents,), 0.1, dtype=torch.float64), T,
                                       lam_T=0.05, max_evals=250, restarts=2, seed=ti)
                for p in ps:
                    pause = torch.full((n_agents,), p, dtype=torch.float64)
                    dur_t = dur_by_p[p]
                    with torch.no_grad():
                        dp0 = risk(paths_t, rr.wait, pause, T, collision_mode="hard",
                                   gate_mode="hard", return_diagnostics=False)
                        dp1 = risk(paths_t, rr.wait, pause, T, collision_mode="hard",
                                   gate_mode="hard", return_diagnostics=False,
                                   segment_durations=dur_t)
                    m0 = np.asarray(dp0["pair_risks"])
                    m1 = np.asarray(dp1["pair_risks"])
                    # M2：半马尔可夫 —— 每个段时长从观测分布里抽样，再跑首次碰撞 DP
                    rng = np.random.default_rng(args.seed0 + ti * 31 + int(p * 100))
                    samp = samples_by_p[p]
                    acc = np.zeros_like(m0)
                    K = args.m2_draws
                    for _k in range(K):
                        idx = rng.integers(0, samp.shape[0], size=n_agents)
                        dur_k = np.take_along_axis(samp, idx[:, None, None], axis=0)[:, 0, :]
                        with torch.no_grad():
                            dk = risk(paths_t, rr.wait, pause, T, collision_mode="hard",
                                      gate_mode="hard", return_diagnostics=False,
                                      segment_durations=torch.tensor(dur_k, dtype=torch.float64))
                        acc += np.asarray(dk["pair_risks"])
                    m2 = acc / K
                    hits = []
                    for k in range(args.n_mc):
                        # 每次 MC 用**不同**种子的事件序列（之前复用同一串是 bug）
                        sim = simulate_mpc(paths_np, rr.wait.numpy(), p,
                                           MPCSimConfig(d_safe=d_safe, horizon_steps=T),
                                           seed=args.seed0 + ti * 100 + k, record=False)
                        hits.append(np.asarray(sim.pair_collision, dtype=bool))
                    obs = np.asarray(hits).mean(axis=0)
                    pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
                    for idx, (i, j) in enumerate(pairs):
                        rows.append({"group": tag, "task_id": jt["task_id"], "plan": plan_name,
                                     "p": p, "pair": f"{i}-{j}",
                                     "m0": float(m0[idx]), "m1": float(m1[idx]),
                                     "m2": float(m2[idx]),
                                     "obs": float(obs[idx]), "n_mc": args.n_mc})
            logger.info("[%s] 测试 %s 完成", tag, jt["task_id"])

    with (run_dir / "m1_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    m0 = np.array([r["m0"] for r in rows])
    m1 = np.array([r["m1"] for r in rows])
    m2 = np.array([r["m2"] for r in rows])
    ob = np.array([r["obs"] for r in rows])
    summary = {"n_mc": args.n_mc, "pairs": len(rows), "m2_draws": args.m2_draws,
               "M0_vs_MPC": stats(m0, ob), "M1_vs_MPC": stats(m1, ob),
               "M2_vs_MPC": stats(m2, ob),
               "by_group": {}, "by_p": {}}
    for g in ("N2", "N4"):
        sel = [r for r in rows if r["group"] == g]
        summary["by_group"][g] = {
            "M0": stats([r["m0"] for r in sel], [r["obs"] for r in sel]),
            "M1": stats([r["m1"] for r in sel], [r["obs"] for r in sel]),
            "M2": stats([r["m2"] for r in sel], [r["obs"] for r in sel])}
    for p in ps:
        sel = [r for r in rows if abs(r["p"] - p) < 1e-9]
        summary["by_p"][str(p)] = {
            "M0": stats([r["m0"] for r in sel], [r["obs"] for r in sel]),
            "M1": stats([r["m1"] for r in sel], [r["obs"] for r in sel]),
            "M2": stats([r["m2"] for r in sel], [r["obs"] for r in sel])}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, pred, name in ((axes[0], m0, "M0 原始 First-Hit"),
                           (axes[1], m1, "M1 段时长修正")):
        ax.scatter(pred, ob, s=16, alpha=0.7)
        lim = [0, max(0.6, float(max(pred.max(), ob.max())) * 1.05)]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(name)
        ax.set_ylabel("MPC MC 两两碰撞频率")
        ax.set_title(f"MAE={np.mean(np.abs(pred-ob)):.3f}  ρ={spearman(pred,ob):.3f}",
                     fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "m1_vs_m0.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "n"}
                      for k, v in summary.items()
                      if k in ("M0_vs_MPC", "M1_vs_MPC", "M2_vs_MPC")},
                     ensure_ascii=False, indent=1))
    print(f"输出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
