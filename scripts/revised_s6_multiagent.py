"""第二版方案 · S6：恢复等待变量，多机器人联合实验。

只有 S5 的几何引导验证成功，才进入本阶段（脚本会读 S5 的结论，不通过则只做诊断对照）。

四组方法：
  B0  Standard DDIM（无引导、无等待）—— "原始 MMD"
  B1  原有中间轨迹梯度引导（无等待）
  B2  初始噪声优化（无等待）
  B3  初始噪声优化 + 等待优化（联合 z 与 w）

评估：
  * 规划层：静态安全、速度、硬首次碰撞风险
  * 执行层：`simulate_execution` 跑多条随机延迟实现 -> 全尝试安全完成率
  * makespan、路径长度、平滑度、耗时

用法：
  python scripts/revised_s6_multiagent.py --ckpt <best.pt> --agents 2 --tasks 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.diffusion.ddim import ddim_sample  # noqa: E402
from fhd.diffusion.guided_ddim import GuidanceConfig, guided_ddim_sample  # noqa: E402
from fhd.diffusion.latent_noise_optimizer import NoiseOptConfig, optimize_initial_noise  # noqa: E402
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.diffusion.train import build_model  # noqa: E402
from fhd.evaluation.reporting import StageReport  # noqa: E402
from fhd.evaluation.statistics import cluster_bootstrap_ci, paired_bootstrap_diff  # noqa: E402
from fhd.evaluation.visualizations import AGENT_COLORS, draw_map  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.geometry.collisions import trajectory_is_static_safe  # noqa: E402
from fhd.geometry.paths import path_length  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import resolve_device, set_seed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402

_setup_fonts()
sys.path.insert(0, str(ROOT / "scripts"))
from revised_s5_noise import build_tasks  # noqa: E402

from fhd.data.dataset import TrajectoryDataset  # noqa: E402


def optimize_wait(paths_m: torch.Tensor, risk_model, pause_p, t_steps: int, cfg,
                  iters: int = 60, lr: float = 0.15, w0: torch.Tensor | None = None):
    """只优化等待变量 w（几何固定）。Adam + 投影。"""
    n, h = paths_m.shape[0], paths_m.shape[1]
    if w0 is None:
        w = torch.zeros(n, h - 1, dtype=torch.float64)
    else:
        w = w0.clone()
    w = w.requires_grad_(True)
    max_pp = float(cfg["wait"].get("max_wait_per_point", 4.0))
    max_total = float(cfg["wait"].get("max_total_wait", 8.0))
    opt = torch.optim.Adam([w], lr=lr)
    lam_T = float(cfg["guidance"].get("lambda_T", 0.05))
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        risk = risk_model(paths_m, w, pause_p, t_steps, collision_mode="soft",
                          gate_mode="soft", return_diagnostics=False)
        loss = risk["pair_sum_risk"] + lam_T * risk["finish_cost"]
        (g,) = torch.autograd.grad(loss, w)
        w.grad = g
        opt.step()
        with torch.no_grad():
            w.clamp_(0.0, max_pp)
            total = w.sum(dim=1, keepdim=True)
            scale = torch.clamp(max_total / total.clamp(min=1e-9), max=1.0)
            w.mul_(scale)
        w.requires_grad_(True)
    return w.detach()


def main() -> int:
    ap = argparse.ArgumentParser(description="S6 多机器人联合实验")
    ap.add_argument("--config", default=str(ROOT / "configs" / "diffusion_train.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "revised_s1"))
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--exec-samples", type=int, default=300)
    ap.add_argument("--pause-p", type=float, default=0.1)
    ap.add_argument("--ddim-steps", type=int, default=40)
    ap.add_argument("--noise-iters", type=int, default=5)
    ap.add_argument("--wait-iters", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device("cpu")
    run_dir = new_run_dir("S6_multi_agent", f"revised_s6_n{args.agents}")
    logger = get_logger("fhd.s6r", run_dir / "logs" / "s6.log")
    log_section(logger, f"S6 多机器人联合实验（N={args.agents}）")

    model = build_model(cfg, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                            kind=str(cfg["diffusion"]["beta_schedule"]), device=device)
    risk_model = FirstHitRisk.from_config(cfg)
    d_safe = risk_model.d_safe
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    h = int(cfg["trajectory"]["horizon"])
    scale = torch.tensor([float(cfg["world"]["width_m"]) / 2.0,
                          float(cfg["world"]["height_m"]) / 2.0], dtype=torch.float32)

    ds = TrajectoryDataset(Path(args.data), "test",
                           grid_size=int(cfg["diffusion"].get("map_input_res",
                                                              cfg["world"]["grid_size"])),
                           robot_radius_m=float(cfg["world"]["robot_radius_m"]),
                           clearance_margin_m=float(cfg["world"]["clearance_margin_m"]),
                           width_m=float(cfg["world"]["width_m"]),
                           height_m=float(cfg["world"]["height_m"]))
    tasks = build_tasks(ds, args.tasks, args.agents)
    logger.info("任务 %d 个（N=%d，同图路径冲突）", len(tasks), args.agents)

    opt_cfg = NoiseOptConfig(steps=args.noise_iters, lr=0.01, trust_radius=0.1,
                             ddim_steps=args.ddim_steps,
                             lambda_R=float(cfg["guidance"].get("lambda_R", 1.0)),
                             lambda_T=float(cfg["guidance"].get("lambda_T", 0.05)),
                             lambda_S=float(cfg["guidance"].get("lambda_S", 10.0)),
                             lambda_K=float(cfg["guidance"].get("lambda_K", 1.0)),
                             lambda_G=float(cfg["guidance"].get("lambda_G", 0.5)))

    rows = []
    for ti, t in enumerate(tasks):
        n = args.agents
        start, goal = t["starts"], t["goals"]
        start_n, goal_n = start / scale - 1.0, goal / scale - 1.0
        world = ds.maps[t["map_index"]]
        res = int(cfg["diffusion"].get("map_input_res", cfg["world"]["grid_size"]))
        grid = world.occupancy(torch.float32, grid_size=res).unsqueeze(0).unsqueeze(0).expand(
            n, -1, -1, -1)
        pause_p = torch.full((n,), args.pause_p)
        z0 = torch.randn(n, h, 2, generator=torch.Generator().manual_seed(5000 + ti))

        plans = {}
        # B0
        t0 = time.time()
        with torch.no_grad():
            x0 = ddim_sample(model, schedule, start_n, goal_n, grid, num_steps=args.ddim_steps,
                             x_init=z0)
        plans["B0"] = (((x0 + 1.0) * scale).double(), None, time.time() - t0)

        # B1
        t0 = time.time()
        g1 = GuidanceConfig(enabled=True,
                            start_fraction=float(cfg["diffusion"].get("guidance_start_fraction", 0.25)),
                            step_size_x=0.2, step_size_w=0.0, grad_clip_norm=1.0,
                            max_wait_per_point=0.0, max_total_wait=0.0,
                            wait_mask_mode="all", optimize_wait=False, optimize_path=True,
                            hard_eval_every=0)
        out1 = guided_ddim_sample(model, schedule, risk_model, cfg, start, goal, grid, pause_p,
                                  obstacles=world.obstacles(), num_steps=args.ddim_steps,
                                  guide=g1, time_steps=t_steps, dt_s=dt_s, d_safe=d_safe,
                                  x_init=z0.clone())
        plans["B1"] = (out1.paths_m.double(), None, time.time() - t0)

        # B2
        t0 = time.time()
        opt = optimize_initial_noise(model, schedule, risk_model, cfg, start, goal, grid,
                                     pause_p, z0.clone(), obstacles=world.obstacles(),
                                     opt=opt_cfg, time_steps=t_steps, dt_s=dt_s)
        b2_time = time.time() - t0
        plans["B2"] = (opt.paths_m.double(), None, b2_time)

        # B3 = B2 + 等待优化
        t0 = time.time()
        w3 = optimize_wait(plans["B2"][0], risk_model, pause_p, t_steps, cfg,
                           iters=args.wait_iters)
        plans["B3"] = (plans["B2"][0], w3, b2_time + time.time() - t0)

        for method, (paths_m, wait, elapsed) in plans.items():
            wv = wait if wait is not None else torch.zeros(n, h - 1, dtype=torch.float64)
            with torch.no_grad():
                hard = risk_model(paths_m, wv, pause_p, t_steps, collision_mode="hard",
                                  gate_mode="hard", return_diagnostics=False)
                safe, clr = trajectory_is_static_safe(paths_m, world.obstacles(),
                                                      world.robot_radius_m,
                                                      world.clearance_margin_m)
                lengths = path_length(paths_m)
                sim = simulate_execution(paths_m, wv, pause_p, d_safe, t_steps, dt_s,
                                         n_samples=args.exec_samples,
                                         generator=torch.Generator().manual_seed(77 + ti))
            rows.append({
                "method": method, "task_id": t["task_id"], "kind": t["kind"],
                "plan_valid": bool(safe.all()),
                "hard_risk": float(hard["pair_sum_risk"]),
                "static_safe": bool(safe.all()),
                "min_clearance_m": float(clr.min()),
                "collision_rate": sim.collision_rate(),
                "safe_completion": float((~sim.collision).to(torch.float64).mean()),
                "makespan_steps": float(sim.makespan_steps.to(torch.float64).mean()),
                "all_arrived_rate": float(sim.all_arrived.to(torch.float64).mean()),
                "path_length_mean": float(torch.as_tensor(lengths).mean()),
                "wait_total": float(wv.sum()),
                "elapsed_s": elapsed,
            })
        logger.info("任务 %d/%d 完成", ti + 1, len(tasks))

    methods = ["B0", "B1", "B2", "B3"]
    summary = {}
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        clusters = np.array([r["task_id"] for r in sel])
        vals = np.array([r["hard_risk"] for r in sel])
        summary[m] = {
            "n": len(sel),
            "plan_valid_rate": float(np.mean([r["plan_valid"] for r in sel])),
            "hard_risk_mean": float(vals.mean()),
            "hard_risk_ci": cluster_bootstrap_ci(vals, clusters, n_boot=2000),
            "safe_completion_mean": float(np.mean([r["safe_completion"] for r in sel])),
            "collision_rate_mean": float(np.mean([r["collision_rate"] for r in sel])),
            "all_arrived_rate": float(np.mean([r["all_arrived_rate"] for r in sel])),
            "makespan_steps": float(np.mean([r["makespan_steps"] for r in sel])),
            "path_length_mean": float(np.mean([r["path_length_mean"] for r in sel])),
            "wait_total_mean": float(np.mean([r["wait_total"] for r in sel])),
            "elapsed_s_mean": float(np.mean([r["elapsed_s"] for r in sel])),
        }

    def paired(a, b, key="safe_completion"):
        da = {r["task_id"]: r[key] for r in rows if r["method"] == a}
        db = {r["task_id"]: r[key] for r in rows if r["method"] == b}
        keys = sorted(set(da) & set(db))
        return paired_bootstrap_diff(np.array([da[k] for k in keys]),
                                     np.array([db[k] for k in keys]),
                                     np.array(keys), n_boot=2000)

    comps = {f"B3_vs_B1": paired("B3", "B1"), f"B3_vs_B2": paired("B3", "B2"),
             f"B3_vs_B0": paired("B3", "B0"), f"B2_vs_B0": paired("B2", "B0"),
             f"B1_vs_B0": paired("B1", "B0")}

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    xs = np.arange(len(methods))
    vals = [summary[m]["safe_completion_mean"] for m in methods]
    axes[0].bar(xs, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("安全完成率")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"N={args.agents}：执行层安全完成率", fontsize=10)
    axes[0].grid(alpha=0.3, axis="y")
    for i, m in enumerate(methods):
        axes[1].scatter([i] * len([r for r in rows if r["method"] == m]),
                        [r["hard_risk"] for r in rows if r["method"] == m], s=18, alpha=0.6)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("硬首次碰撞风险")
    axes[1].set_title("逐任务硬风险分布", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / f"s6_n{args.agents}_comparison.png", dpi=130)
    plt.close(fig)

    report = StageReport("S6_multi_agent", run_dir, f"S6 多机器人联合实验（N={args.agents}）报告",
                         cfg.to_dict())
    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    report.metric("agents", args.agents, group="setup")
    report.metric("pause_p", args.pause_p, group="setup")
    for m, v in summary.items():
        for k in ("plan_valid_rate", "hard_risk_mean", "safe_completion_mean",
                  "all_arrived_rate", "makespan_steps", "wait_total_mean", "elapsed_s_mean"):
            report.metric(f"{m}/{k}", v[k], group="method")
    for k, v in comps.items():
        report.metric(f"{k}/diff", v["diff"], group="paired")
        report.metric(f"{k}/significant", v["significant"], group="paired")
    report.table("s6_per_task", rows)

    lines = ["| 方法 | 规划有效率 | 硬风险 | 安全完成率 | 全到达率 | makespan [时隙] | 等待总量 | 耗时 [s] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in methods:
        v = summary[m]
        lines.append(f"| {m} | {v['plan_valid_rate']:.3f} | {v['hard_risk_mean']:.4f} | "
                     f"{v['safe_completion_mean']:.3f} | {v['all_arrived_rate']:.3f} | "
                     f"{v['makespan_steps']:.1f} | {v['wait_total_mean']:.2f} | "
                     f"{v['elapsed_s_mean']:.2f} |")
    report.add_section("## 结果\n\n" + "\n".join(lines) + "\n")
    report.add_section(
        "## 配对比较（安全完成率）\n\n"
        + "\n".join(f"- {k}: {v['diff']:+.4f} "
                    f"[{v['lo']:+.4f}, {v['hi']:+.4f}] "
                    f"{'显著' if v['significant'] else '不显著'}" for k, v in comps.items())
        + "\n")
    report.write_report()
    report.write_metrics_csv()
    report.write_status("completed", {}, {"summary": summary})
    report.sync_to_docs()

    import shutil
    docs = ROOT / "docs" / "revised"
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "report.md", docs / f"05_S6_multiagent_n{args.agents}.md")
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, docs / "figures" / png.name)

    print(json.dumps({m: round(summary[m]["safe_completion_mean"], 4) for m in methods},
                     ensure_ascii=False))
    print(f"运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
