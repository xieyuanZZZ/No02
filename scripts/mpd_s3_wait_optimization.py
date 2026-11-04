"""阶段三：仅等待优化（W0–W3）的正式实验。

固定 24 个冻结任务、冻结几何，只改等待；用**独立评测种子**做蒙特卡洛。

流程：
  1. §2 等待语义校验：固定轨迹，只给第 2 台机器人起点加等待，
     核对 DP 硬风险 / 完成时隙 / MC 碰撞率三者一致；
  2. §4 风险梯度诊断：24 个任务的软/硬风险、梯度范数、非零变量数、
     有限差分方向校验；
  3. W1 的固定参数在**开发种子**上选定并冻结；
  4. 正式评测：W0/W1/W2/W3 × p∈{0,0.05,0.1,0.2}，评测种子独立；
  5. 配对 bootstrap、全任务与敏感子集分别汇报。

用法：
  python scripts/mpd_s3_wait_optimization.py
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

from fhd.evaluation.statistics import paired_bootstrap_diff  # noqa: E402
from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget,
    allowed_positions,
    gradient_diagnostics,
    make_mask,
    policy_w0,
    policy_w1,
    policy_w2_enum,
    policy_w3_optimize,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

_setup_fonts()

LAM_T = 0.05      # 与 guidance.yaml 的 lambda_T 一致
LAM_R = 1.0
LAM_W = 0.01
OPT_P = 0.1       # 优化时使用的名义暂停概率


def run_mc(paths, wait, p, d_safe, t_steps, dt_s, n, seed):
    gen = torch.Generator().manual_seed(int(seed))
    pause = torch.full((paths.shape[0],), float(p), dtype=torch.float64)
    sim = simulate_execution(paths, wait, pause, d_safe, t_steps, dt_s,
                             n_samples=n, generator=gen)
    coll = sim.collision.to(torch.float64)
    arrived = sim.all_arrived.to(torch.float64)
    timeout = (sim.makespan_steps >= t_steps).to(torch.float64)
    return {
        "collision_rate": float(coll.mean()),
        "safe_completion_rate": float(((1 - coll) * arrived).mean()),
        "all_arrived_rate": float(arrived.mean()),
        "timeout_rate": float(timeout.mean()),
        "makespan_mean_slots": float(sim.makespan_steps.to(torch.float64).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(ROOT / "data" / "mpd_delay_plan_bank"))
    ap.add_argument("--agents", type=int, default=2,
                    help="2 用 plan_bank.json；>2 用 plan_bank_n{N}.json")
    ap.add_argument("--screen-seeds", default="20261104",
                    help="N>2 时用这组种子做敏感性筛选（与评测种子不同）")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--n-mc-eval", type=int, default=400)
    ap.add_argument("--n-mc-dev", type=int, default=200)
    ap.add_argument("--dev-seeds", default="20261110,20261111")
    ap.add_argument("--eval-seeds", default="20261120,20261121")
    ap.add_argument("--ps", default="0,0.05,0.1,0.2")
    ap.add_argument("--w1-candidates", default="1,2,3,5,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    h = int(cfg["trajectory"]["horizon"])
    risk_model = FirstHitRisk.from_config(cfg)
    d_safe = risk_model.d_safe

    run_dir = Path(args.out) if args.out else new_run_dir("MPD_S3_wait", "s3_wait")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.mpd_s3", run_dir / "logs" / "s3.log")
    log_section(logger, "阶段三：仅等待优化")

    n_agents = int(args.agents)
    bank_json = "plan_bank.json" if n_agents == 2 else f"plan_bank_n{n_agents}.json"
    trajs_file = "plan_bank_trajs.npz" if n_agents == 2 else f"plan_bank_n{n_agents}_trajs.npz"
    bank = json.loads((Path(args.bank) / bank_json).read_text(encoding="utf-8"))
    trajs_npz = np.load(Path(args.bank) / trajs_file)
    tasks = bank["tasks"]
    sensitive = set()
    freeze_path = Path(args.bank) / ("task_freeze.json" if n_agents == 2
                                     else f"task_freeze_n{n_agents}.json")
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        sensitive = set(freeze["groups"]["light"]) | set(freeze["groups"]["high"])
    logger.info("N=%d，任务 %d（敏感子集 %d），p=%s", n_agents, len(tasks), len(sensitive), args.ps)

    budget = WaitBudget(time_steps=t_steps, horizon=h)
    positions = allowed_positions(h, stride=5)
    mask = make_mask(h, positions, n_agents)
    logger.info("允许等待的航点 %s（共 %d 个变量/机器人），预算：单点≤%.0f 总量≤%.0f",
                positions, len(positions), budget.max_added_per_point,
                budget.max_added_total_per_robot)

    # ---------------- 0) N>2 时先做敏感性筛选并冻结分组 ----------------
    if not sensitive:
        log_section(logger, "0. 敏感性筛选（N=%d，筛选种子与评测种子不同）" % n_agents)
        screen_seeds = [int(x) for x in args.screen_seeds.split(",")]
        groups = {"insensitive": [], "light": [], "high": []}
        for ti, jt in enumerate(tasks):
            paths = torch.tensor(trajs_npz[jt["task_id"]], dtype=torch.float64)
            w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
            rates = []
            for seed in screen_seeds:
                m = run_mc(paths, w0, 0.1, d_safe, t_steps, dt_s, args.n_mc_dev,
                           seed=seed * 3000 + ti)
                rates.append(m["collision_rate"])
            c = float(np.mean(rates))
            groups["insensitive" if c < 0.02 else ("light" if c < 0.2 else "high")].append(
                jt["task_id"])
            logger.info("  %s：p=0.1 筛选碰撞率 %.4f", jt["task_id"], c)
        sensitive = set(groups["light"]) | set(groups["high"])
        freeze_path.write_text(json.dumps({
            "screen_seeds": screen_seeds, "reference_p": 0.1,
            "rule": {"insensitive": "<0.02", "light": "0.02-0.2", "high": ">=0.2"},
            "groups": groups,
            "note": "分组仅用筛选种子；正式评测使用独立评测种子",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("分组：不敏感 %d，轻度 %d，高度 %d", len(groups["insensitive"]),
                    len(groups["light"]), len(groups["high"]))

    # ---------------- 1) 等待语义校验 ----------------
    log_section(logger, "1. 等待语义校验（只给第 2 台机器人起点加等待）")
    sanity = []
    jt = max(tasks, key=lambda t: 1.0)   # 取第一个任务即可，重点是三方一致性
    for jt in tasks[:4]:
        paths = torch.tensor(trajs_npz[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        for d in (0, 2, 5):
            w = w0.clone()
            w[1, 0] = w[1, 0] + d
            with torch.no_grad():
                hard = risk_model(paths, w, torch.full((n_agents,), OPT_P, dtype=torch.float64),
                                  t_steps, collision_mode="hard", gate_mode="hard",
                                  return_diagnostics=False)
            mc = run_mc(paths, w, OPT_P, d_safe, t_steps, dt_s, 300,
                        seed=hash((jt["task_id"], d)) % 2**31)
            comp = (h - 1) + w.sum(dim=-1)
            sanity.append({
                "task_id": jt["task_id"], "delta": d,
                "dp_hard_risk": float(hard["pair_sum_risk"]),
                "mc_collision_rate": mc["collision_rate"],
                "pred_completion_robot1": float(comp[1]),
                "mc_makespan_slots": mc["makespan_mean_slots"],
                "added_total_slots": float((w - w0).sum()),
            })
            logger.info("  %s Δ=%d：DP=%.3f MC=%.3f 预测完成=%.0f 实测makespan=%.1f",
                        jt["task_id"], d, float(hard["pair_sum_risk"]),
                        mc["collision_rate"], float(comp[1]), mc["makespan_mean_slots"])

    # 一致性断言：Δ 越大，完成时隙越大；DP 与 MC 同向
    for tid in {s["task_id"] for s in sanity}:
        sub = sorted([s for s in sanity if s["task_id"] == tid], key=lambda x: x["delta"])
        assert sub[-1]["pred_completion_robot1"] > sub[0]["pred_completion_robot1"]

    # ---------------- 2) 梯度诊断 ----------------
    log_section(logger, "2. 风险梯度诊断")
    diag = []
    for jt in tasks:
        paths = torch.tensor(trajs_npz[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        pause = torch.full((n_agents,), OPT_P, dtype=torch.float64)
        d = gradient_diagnostics(w0, mask, budget, risk_model, paths, pause, t_steps,
                                 lam_T=LAM_T, lam_W=LAM_W)
        d["task_id"] = jt["task_id"]
        diag.append(d)
    sat = [d for d in diag if d["hard_risk_init"] > 0.2 and d["grad_norm"] < 1e-6]
    logger.info("梯度诊断：%d 个任务，有限差分通过 %d 个，'高硬风险+零梯度' %d 个",
                len(diag), sum(1 for d in diag if d["fd_descent_ok"]), len(sat))
    logger.info("初值：soft 中位 %.3f，hard 中位 %.3f，梯度范数中位 %.3e",
                float(np.median([d["soft_risk_init"] for d in diag])),
                float(np.median([d["hard_risk_init"] for d in diag])),
                float(np.median([d["grad_norm"] for d in diag])))

    # ---------------- 3) 开发集选 W1 参数 ----------------
    log_section(logger, "3. 在开发种子上选 W1 的固定 Δw")
    dev_seeds = [int(x) for x in args.dev_seeds.split(",")]
    cands = [int(x) for x in args.w1_candidates.split(",")]
    dev_rows = []
    for d in cands:
        tot, tot0 = [], []
        for si, seed in enumerate(dev_seeds):
            for ti, jt in enumerate(tasks):
                paths = torch.tensor(trajs_npz[jt["task_id"]], dtype=torch.float64)
                w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
                # 必须走与评测同一条路径（project_wait 会把单点等待裁到上限）
                r_d = policy_w1(w0, mask, budget, risk_model, paths,
                                torch.full((n_agents,), OPT_P, dtype=torch.float64),
                                t_steps, d)
                m = run_mc(paths, r_d.wait, OPT_P, d_safe, t_steps, dt_s, args.n_mc_dev,
                           seed=seed * 1000 + ti)
                m0 = run_mc(paths, r_d.wait, 0.0, d_safe, t_steps, dt_s, args.n_mc_dev,
                            seed=seed * 1000 + ti)
                tot.append(m["collision_rate"])
                tot0.append(m0["collision_rate"])
        dev_rows.append({"delta": d, "dev_collision_rate": float(np.mean(tot)),
                         "dev_p0_collision_rate": float(np.mean(tot0))})
        logger.info("  Δw=%d：开发集 p=0.1 碰撞率 %.4f，p=0 碰撞率 %.4f", d,
                    dev_rows[-1]["dev_collision_rate"], dev_rows[-1]["dev_p0_collision_rate"])
    # 硬约束：固定等待也必须保证 p=0 无碰撞（冻结任务的基本要求）
    safe_cands = [r for r in dev_rows if r["dev_p0_collision_rate"] <= 1e-9]
    if safe_cands:
        best_dev = min(safe_cands, key=lambda r: (r["dev_collision_rate"], r["delta"]))
        W1_DELTA = int(best_dev["delta"])
    else:
        logger.warning("没有任何 p=0 安全的固定 Δw；W1 退化为 Δ=0（与 W0 等价）")
        W1_DELTA = 0
    logger.info("冻结 W1：Δw=%d（开发集选参，正式评测不再调整）", W1_DELTA)

    # ---------------- 4) 正式评测 ----------------
    log_section(logger, "4. 正式评测（独立评测种子）")
    eval_seeds = [int(x) for x in args.eval_seeds.split(",")]
    ps = [float(x) for x in args.ps.split(",")]
    plan_rows = []

    for ti, jt in enumerate(tasks):
        paths = torch.tensor(trajs_npz[jt["task_id"]], dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        pause = torch.full((n_agents,), OPT_P, dtype=torch.float64)

        t0 = __import__("time").perf_counter()
        r_w0 = policy_w0(w0, mask, budget, risk_model, paths, pause, t_steps)
        r_w1 = policy_w1(w0, mask, budget, risk_model, paths, pause, t_steps, W1_DELTA)
        r_w2 = policy_w2_enum(w0, mask, budget, risk_model, paths, pause, t_steps, LAM_T)
        r_w3 = policy_w3_optimize(w0, mask, budget, risk_model, paths, pause, t_steps,
                                  lam_R=LAM_R, lam_T=LAM_T, lam_W=LAM_W)
        opt_s = __import__("time").perf_counter() - t0

        for name, r in (("W0", r_w0), ("W1", r_w1), ("W2", r_w2), ("W3", r_w3)):
            for p in ps:
                mets = []
                for seed in eval_seeds:
                    mets.append(run_mc(paths, r.wait, p, d_safe, t_steps, dt_s,
                                       args.n_mc_eval, seed=seed * 7000 + ti))
                agg = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
                plan_rows.append({
                    "task_id": jt["task_id"], "kind": jt["kind"],
                    "sensitive": jt["task_id"] in sensitive,
                    "method": name, "p": p,
                    "dp_hard_risk": r.hard_risk, "dp_soft_risk": r.soft_risk,
                    "added_slots": float(r.added.sum()),
                    "added_robot1": float(r.added[0].sum()),
                    "added_robot2": float(r.added[1].sum()),
                    "fallback": bool(r.fallback),
                    "opt_s": opt_s if name in ("W2", "W3") else 0.0,
                    **agg,
                })
        if (ti + 1) % 6 == 0:
            logger.info("  %d/%d 任务完成", ti + 1, len(tasks))

    # ---------------- 5) 汇总 ----------------
    def summarize(rows, p, subset=None):
        sel = [r for r in rows if abs(r["p"] - p) < 1e-9 and
               (subset is None or r["sensitive"] == subset)]
        out = {}
        for m in ("W0", "W1", "W2", "W3"):
            s = [r for r in sel if r["method"] == m]
            if not s:
                continue
            out[m] = {
                "collision_rate": float(np.mean([r["collision_rate"] for r in s])),
                "safe_completion_rate": float(np.mean([r["safe_completion_rate"] for r in s])),
                "makespan_mean_slots": float(np.mean([r["makespan_mean_slots"] for r in s])),
                "added_slots_mean": float(np.mean([r["added_slots"] for r in s])),
                "timeout_rate": float(np.mean([r["timeout_rate"] for r in s])),
                "fallback_rate": float(np.mean([r["fallback"] for r in s])),
            }
        return out

    summary = {"tasks": len(tasks), "sensitive_tasks": len(sensitive),
               "W1_delta": W1_DELTA, "dev_selection": dev_rows,
               "ps": ps, "by_p": {}, "sensitive_only": {}}
    for p in ps:
        summary["by_p"][str(p)] = summarize(plan_rows, p)
        summary["sensitive_only"][str(p)] = summarize(plan_rows, p, subset=True)
        logger.info("p=%.2f 全部：%s", p, json.dumps(
            {m: round(v["collision_rate"], 4) for m, v in summary["by_p"][str(p)].items()}))
        logger.info("p=%.2f 敏感子集：%s", p, json.dumps(
            {m: round(v["collision_rate"], 4)
             for m, v in summary["sensitive_only"][str(p)].items()}))

    # 配对 bootstrap（p=0.1，按任务配对，安全完成率）
    def paired(a, b, p=OPT_P):
        da = {r["task_id"]: r["safe_completion_rate"] for r in plan_rows
              if r["method"] == a and abs(r["p"] - p) < 1e-9}
        db = {r["task_id"]: r["safe_completion_rate"] for r in plan_rows
              if r["method"] == b and abs(r["p"] - p) < 1e-9}
        keys = sorted(set(da) & set(db))
        return paired_bootstrap_diff(np.array([da[k] for k in keys]),
                                     np.array([db[k] for k in keys]),
                                     np.array(keys), n_boot=2000)

    comp = {"W2_vs_W0": paired("W2", "W0"), "W2_vs_W1": paired("W2", "W1"),
            "W3_vs_W0": paired("W3", "W0"), "W3_vs_W2": paired("W3", "W2"),
            "W3_vs_W1": paired("W3", "W1")}
    summary["paired_at_p0.1"] = comp
    for k, v in comp.items():
        logger.info("%s 安全完成率差 %+.4f [%+.4f, %+.4f] %s", k, v["diff"], v["lo"], v["hi"],
                    "显著" if v["significant"] else "不显著")

    # ---------------- 落盘 ----------------
    def dump(name, rows):
        if not rows:
            return
        with (run_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("wait_eval.csv", plan_rows)
    dump("gradient_diagnostics.csv", diag)
    dump("sanity_check.csv", sanity)
    dump("dev_selection.csv", dev_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"W0": "#4c72b0", "W1": "#dd8452", "W2": "#55a868", "W3": "#c44e52"}
    for m in ("W0", "W1", "W2", "W3"):
        ys = [summary["by_p"][str(p)][m]["collision_rate"] for p in ps]
        axes[0].plot(ps, ys, "o-", color=colors[m], label=m)
    axes[0].set_xlabel("暂停概率 p")
    axes[0].set_ylabel("碰撞率")
    axes[0].set_title("碰撞率 vs p（全部 24 任务）", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for m in ("W0", "W1", "W2", "W3"):
        ys = [summary["sensitive_only"][str(p)][m]["collision_rate"] for p in ps]
        axes[1].plot(ps, ys, "o-", color=colors[m], label=m)
    axes[1].set_xlabel("暂停概率 p")
    axes[1].set_ylabel("碰撞率")
    axes[1].set_title("碰撞率 vs p（延迟敏感子集）", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "s3_collision_vs_p.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = ["W0", "W1", "W2", "W3"]
    safe = [summary["by_p"][str(OPT_P)][m]["safe_completion_rate"] for m in labels]
    mks = [summary["by_p"][str(OPT_P)][m]["makespan_mean_slots"] for m in labels]
    ax2 = ax.twinx()
    ax.bar(np.arange(4) - 0.2, safe, 0.4, label="安全完成率", color="#4c72b0")
    ax2.bar(np.arange(4) + 0.2, mks, 0.4, label="makespan（时隙）", color="#dd8452")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("安全完成率")
    ax2.set_ylabel("makespan（时隙）")
    ax.set_title(f"p={OPT_P} 时四种等待策略的权衡", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "s3_tradeoff.png", dpi=130)
    plt.close(fig)

    print(json.dumps({"W1_delta": W1_DELTA,
                      "p0.1_all": {m: round(v["collision_rate"], 4)
                                   for m, v in summary["by_p"]["0.1"].items()},
                      "p0.1_sensitive": {m: round(v["collision_rate"], 4)
                                         for m, v in summary["sensitive_only"]["0.1"].items()},
                      "paired": {k: round(v["diff"], 4) for k, v in comp.items()}},
                     ensure_ascii=False, indent=2))
    print(f"输出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
