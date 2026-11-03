"""阶段二：确认随机执行暂停确实造成碰撞，并冻结任务集。

流程（对应实施安排 §3）：
  1. 原轨迹、基准发车间隔保持不变，只加逐步随机暂停，p ∈ {0, 0.05, 0.1, 0.2}；
  2. 每任务多次蒙特卡洛执行（统一种子集），记录首次碰撞、完成时间、超时；
  3. 用 First-Hit Risk DP 预测同一配置的碰撞概率，与 MC 对照；
  4. 用**筛选种子**把任务分成"延迟不敏感/轻度/高度敏感"，冻结任务 ID；
     正式评测将使用另一组评测种子（本脚本不产出评测结论）。

输出：data/mpd_delay_plan_bank/delay/
  delay_mc.csv        逐 (任务, p) 的 MC 结果
  delay_dp.csv        DP 预测
  dp_vs_mc.csv        对照
  task_freeze.json    冻结的任务分组
  figures/*.png       碰撞率、完成时间曲线、典型失败案例时间轴
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
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

_setup_fonts()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(ROOT / "data" / "mpd_delay_plan_bank"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--n-mc", type=int, default=400, help="每个 (任务,p) 的蒙特卡洛次数")
    ap.add_argument("--ps", default="0,0.05,0.1,0.2")
    ap.add_argument("--screen-seed", type=int, default=20261104)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    risk_model = FirstHitRisk.from_config(cfg)
    d_safe = risk_model.d_safe

    run_dir = Path(args.out) if args.out else new_run_dir("MPD_S2_delay", "s2_delay")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.mpd_s2", run_dir / "logs" / "s2.log")
    log_section(logger, "阶段二：随机暂停下的碰撞验证")

    bank = json.loads((Path(args.bank) / "plan_bank.json").read_text(encoding="utf-8"))
    trajs = np.load(Path(args.bank) / "plan_bank_trajs.npz")
    tasks = bank["tasks"]
    ps = [float(x) for x in args.ps.split(",")]
    logger.info("任务 %d 个，p=%s，每格 %d 次 MC，d_safe=%.3f",
                len(tasks), ps, args.n_mc, d_safe)

    mc_rows, dp_rows, cmp_rows = [], [], []
    traces_keep = []
    for ti, jt in enumerate(tasks):
        paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
        n = paths.shape[0]
        wait0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        for p in ps:
            pause_p = torch.full((n,), p, dtype=torch.float64)
            gen = torch.Generator().manual_seed(args.screen_seed + 1000 * ti + int(p * 1000))
            sim = simulate_execution(paths, wait0, pause_p, d_safe, t_steps, dt_s,
                                     n_samples=args.n_mc, generator=gen,
                                     record_traces=(p > 0))
            coll = sim.collision.to(torch.float64)
            arrived = sim.all_arrived.to(torch.float64)
            timeout = (sim.makespan_steps >= t_steps).to(torch.float64)
            safe = (1 - coll) * arrived
            mc_rows.append({
                "task_id": jt["task_id"], "kind": jt["kind"], "p": p, "n_mc": args.n_mc,
                "collision_rate": float(coll.mean()),
                "safe_completion_rate": float(safe.mean()),
                "all_arrived_rate": float(arrived.mean()),
                "timeout_rate": float(timeout.mean()),
                "makespan_mean_slots": float(sim.makespan_steps.to(torch.float64).mean()),
                "first_collision_step_mean": float(
                    sim.first_collision_step.to(torch.float64)[coll.bool()].mean())
                if bool(coll.any()) else float("nan"),
                "pair_min_distance_mean": float(sim.pair_min_distance.mean()),
            })
            # DP 预测（同一 wait / pause_p）
            with torch.no_grad():
                risk = risk_model(paths, wait0, pause_p, t_steps,
                                  collision_mode="hard", gate_mode="hard",
                                  return_diagnostics=False)
            dp_rows.append({"task_id": jt["task_id"], "p": p,
                            "hard_risk": float(risk["pair_sum_risk"])})
            cmp_rows.append({"task_id": jt["task_id"], "p": p,
                             "mc_collision_rate": float(coll.mean()),
                             "dp_hard_risk": float(risk["pair_sum_risk"])})
            if p > 0 and bool(coll.any()) and len(traces_keep) < 6:
                idx = int(torch.nonzero(coll.bool())[0])
                tr = sim.traces
                traces_keep.append({
                    "task_id": jt["task_id"], "p": p, "sample": idx,
                    "collision": True,
                    "first_step": int(sim.first_collision_step[idx]),
                    "paths": paths.numpy().tolist(),
                    "pair_min_distance": float(sim.pair_min_distance[idx].min()),
                })

    # ---------------- 汇总与分组 ----------------
    def agg(rows, key):
        d: dict[str, list] = {}
        for r in rows:
            d.setdefault(r["p"], []).append(r[key])
        return {k: float(np.mean(v)) for k, v in sorted(d.items())}

    logger.info("总体碰撞率 vs p：%s", agg(mc_rows, "collision_rate"))
    logger.info("总体安全完成率 vs p：%s", agg(mc_rows, "safe_completion_rate"))

    # 用 p=0.1 的碰撞率做分组（筛选种子，不用于正式评测）
    ref_p = 0.1
    task_stats = {}
    for r in mc_rows:
        if abs(r["p"] - ref_p) < 1e-9:
            task_stats[r["task_id"]] = r
    groups = {"insensitive": [], "light": [], "high": []}
    for tid, r in task_stats.items():
        c = r["collision_rate"]
        if c < 0.02:
            groups["insensitive"].append(tid)
        elif c < 0.2:
            groups["light"].append(tid)
        else:
            groups["high"].append(tid)
    logger.info("分组（p=%.2f）：不敏感 %d，轻度 %d，高度 %d", ref_p,
                len(groups["insensitive"]), len(groups["light"]), len(groups["high"]))

    p0_max = max((r["collision_rate"] for r in mc_rows if r["p"] == 0.0), default=0.0)
    logger.info("p=0 时最大碰撞率 = %.4f（必须为 0）", p0_max)

    # DP vs MC 相关性
    a = np.array([r["dp_hard_risk"] for r in cmp_rows])
    b = np.array([r["mc_collision_rate"] for r in cmp_rows])
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
    mae = float(np.mean(np.abs(a - b)))
    logger.info("DP vs MC：相关系数 %.3f，平均绝对误差 %.3f", corr, mae)

    # ---------------- 落盘 ----------------
    def dump(name, rows):
        if not rows:
            return
        with (run_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("delay_mc.csv", mc_rows)
    dump("delay_dp.csv", dp_rows)
    dump("dp_vs_mc.csv", cmp_rows)
    (run_dir / "task_freeze.json").write_text(json.dumps({
        "screen_seed": args.screen_seed, "reference_p": ref_p,
        "rule": {"insensitive": "<0.02", "light": "0.02-0.2", "high": ">=0.2"},
        "groups": groups,
        "all_tasks": [jt["task_id"] for jt in tasks],
        "note": "分组仅用筛选种子；正式评测必须使用独立评测种子",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "typical_cases.json").write_text(json.dumps(traces_keep, ensure_ascii=False),
                                                encoding="utf-8")

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    xs = sorted(set(r["p"] for r in mc_rows))
    axes[0].plot(xs, [agg(mc_rows, "collision_rate")[p] for p in xs], "o-", label="全部任务")
    for gname, label in (("insensitive", "延迟不敏感"), ("light", "轻度敏感"), ("high", "高度敏感")):
        ids = set(groups[gname])
        if not ids:
            continue
        sub = [r for r in mc_rows if r["task_id"] in ids]
        d = {}
        for r in sub:
            d.setdefault(r["p"], []).append(r["collision_rate"])
        axes[0].plot(sorted(d), [float(np.mean(d[p])) for p in sorted(d)], "s--", label=label)
    axes[0].set_xlabel("暂停概率 p")
    axes[0].set_ylabel("蒙特卡洛碰撞率")
    axes[0].set_title("随机暂停导致的碰撞率", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, [agg(mc_rows, "safe_completion_rate")[p] for p in xs], "o-",
                 label="安全完成率")
    axes[1].plot(xs, [agg(mc_rows, "all_arrived_rate")[p] for p in xs], "s--", label="全部到达率")
    axes[1].set_xlabel("暂停概率 p")
    axes[1].set_ylabel("比例")
    axes[1].set_title("安全完成率与到达率", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "s2_delay_curves.png", dpi=130)
    plt.close(fig)

    if traces_keep:
        fig, axes = plt.subplots(1, min(3, len(traces_keep)), figsize=(5.2 * min(3, len(traces_keep)), 4.6))
        axes = np.atleast_1d(axes)
        for ax, tc in zip(axes, traces_keep[:3]):
            paths = np.asarray(tc["paths"])
            for i in range(paths.shape[0]):
                ax.plot(paths[i, :, 0], paths[i, :, 1], "-o", ms=2, lw=1.2,
                        label=f"机器人 {i + 1}")
            ax.set_title(f"{tc['task_id']}  p={tc['p']}\n首次碰撞时隙 {tc['first_step']}",
                         fontsize=9)
            ax.set_aspect("equal")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / "s2_typical_collision.png", dpi=130)
        plt.close(fig)

    summary = {
        "task_id_count": len(tasks),
        "p0_max_collision_rate": p0_max,
        "collision_rate_vs_p": agg(mc_rows, "collision_rate"),
        "safe_completion_vs_p": agg(mc_rows, "safe_completion_rate"),
        "makespan_vs_p": agg(mc_rows, "makespan_mean_slots"),
        "dp_mc_corr": corr,
        "dp_mc_mae": mae,
        "groups": {k: len(v) for k, v in groups.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"输出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
