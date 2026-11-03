"""第二版方案 · S5：初始噪声优化。

两件事，严格按门槛顺序：

  A. 噪声敏感性：固定模型/地图/起终点，换 20 个初始噪声，看轨迹是否真的不同。
     若几乎相同（轨迹 RMS < 1 cm），**停止**噪声优化，先诊断模型的噪声利用能力。

  B. 三方法对照（+ 随机扰动对照）：
     B0 = 普通 DDIM；B1 = 原有逐步引导；B2 = 初始噪声优化；R = 同信赖域随机扰动。
     20 个两机器人任务，同一 checkpoint / 地图 / 起终点 / z0 / DDIM 步数 / 风险计算。

用法：
  python scripts/revised_s5_noise.py --config configs/diffusion_train.yaml --ckpt <best.pt> \
      --data data/revised_s1 --tasks 20 --iters 5
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

from fhd.data.dataset import TrajectoryDataset  # noqa: E402
from fhd.diffusion.ddim import ddim_sample  # noqa: E402
from fhd.diffusion.guided_ddim import GuidanceConfig, guided_ddim_sample  # noqa: E402
from fhd.diffusion.latent_noise_optimizer import (  # noqa: E402
    NoiseOptConfig,
    optimize_initial_noise,
    random_perturbation_control,
)
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.diffusion.train import build_model  # noqa: E402
from fhd.evaluation.reporting import StageReport  # noqa: E402
from fhd.evaluation.statistics import cluster_bootstrap_ci, paired_bootstrap_diff  # noqa: E402
from fhd.evaluation.visualizations import AGENT_COLORS, draw_map  # noqa: E402
from fhd.geometry.collisions import trajectory_is_static_safe  # noqa: E402
from fhd.geometry.paths import path_length  # noqa: E402
from fhd.maps.simple_scenes import KIND_LABELS  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import resolve_device, set_seed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_tasks(ds: TrajectoryDataset, n_tasks: int, n_agents: int, seed: int = 11) -> list[dict]:
    """从 S1 测试集构造 N 机器人任务：同一张图、路径互有冲突。"""
    import json as _json

    tasks_file = ds.root / f"tasks_{ds.split}.jsonl"
    meta = {}
    if tasks_file.exists():
        for line in tasks_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = _json.loads(line)
                meta[t["task_id"]] = t
    by_map = {}
    for idx in range(len(ds)):
        mi = int(ds.data["map_idx"][idx])
        tid = str(ds.data.get("task_id", ["?"] * len(ds))[idx]) \
            if "task_id" in ds.data else f"t{idx}"
        by_map.setdefault(mi, []).append((idx, tid))
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n_tasks and tries < n_tasks * 60:
        tries += 1
        mi = int(rng.choice(list(by_map)))
        entries = by_map[mi]
        if len(entries) < n_agents:
            continue
        pick = rng.choice(len(entries), size=n_agents, replace=False)
        idxs = [entries[int(p)][0] for p in pick]
        # 至少一对路径有冲突（最小点距 < d_safe），否则没有研究价值
        paths = [ds.data["paths"][i] for i in idxs]
        min_d = min(float(torch.linalg.vector_norm(paths[a] - paths[b], dim=-1).min())
                    for a in range(n_agents) for b in range(a + 1, n_agents))
        if min_d >= 0.35:
            continue
        out.append({
            "task_id": f"s5_{len(out):03d}",
            "map_index": mi, "kind": ds.maps[mi].kind,
            "task_indices": idxs,
            "starts": torch.stack([ds.data["starts"][i] for i in idxs]).float(),
            "goals": torch.stack([ds.data["goals"][i] for i in idxs]).float(),
            "ref_paths": torch.stack(paths),
            "path_conflict_min_dist": min_d,
        })
    return out


def evaluate_paths(paths_m, risk_model, pause_p, t_steps, world, d_safe) -> dict:
    with torch.no_grad():
        hard = risk_model(paths_m, torch.zeros(paths_m.shape[0], paths_m.shape[1] - 1,
                                              dtype=paths_m.dtype), pause_p, t_steps,
                          collision_mode="hard", gate_mode="hard", return_diagnostics=False)
        safe, clr = trajectory_is_static_safe(paths_m, world.obstacles(), world.robot_radius_m,
                                              world.clearance_margin_m)
        lengths = path_length(paths_m)
        accel = paths_m[:, 2:] - 2 * paths_m[:, 1:-1] + paths_m[:, :-2]
        # 最小机器人间距（同步执行下）
        n = paths_m.shape[0]
        pair_min = min(float(torch.linalg.vector_norm(paths_m[a] - paths_m[b], dim=-1).min())
                       for a in range(n) for b in range(a + 1, n))
    return {"hard_risk": float(hard["pair_sum_risk"]),
            "static_safe": bool(safe.all()), "min_clearance_m": float(clr.min()),
            "pair_min_dist": pair_min,
            "path_length_mean": float(torch.as_tensor(lengths).mean()),
            "smoothness": float(accel.pow(2).sum(-1).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description="S5 初始噪声优化")
    ap.add_argument("--config", default=str(ROOT / "configs" / "diffusion_train.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "revised_s1"))
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--trust-radius", type=float, default=0.1)
    ap.add_argument("--ddim-steps", type=int, default=40)
    ap.add_argument("--noise-samples", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device("cpu")
    run_dir = new_run_dir("S5_noise_optimization", "revised_s5")
    logger = get_logger("fhd.s5r", run_dir / "logs" / "s5.log")
    log_section(logger, "S5 初始噪声优化（先敏感性，再对照）")

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
    dt_s, t_steps = float(cfg["trajectory"]["dt_s"]), int(cfg["execution"]["horizon_steps"])
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
    logger.info("构造 %d 个 %d 机器人任务（同图、路径冲突 < d_safe）", len(tasks), args.agents)
    if not tasks:
        raise SystemExit("未能构造出冲突任务")

    # ---------------- A. 噪声敏感性 ----------------
    log_section(logger, "A. 初始噪声敏感性")
    sens_rows = []
    for t in tasks[:5]:
        n = args.agents
        start = t["starts"]
        goal = t["goals"]
        start_n, goal_n = start / scale - 1.0, goal / scale - 1.0
        grid = ds.maps[t["map_index"]].occupancy(torch.float32,
                                                 grid_size=int(cfg["diffusion"].get(
                                                     "map_input_res",
                                                     cfg["world"]["grid_size"]))
                                                 ).unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1)
        world = ds.maps[t["map_index"]]
        outs = []
        for k in range(args.noise_samples):
            z = torch.randn(n, h, 2, generator=torch.Generator().manual_seed(1000 + k))
            with torch.no_grad():
                x0 = ddim_sample(model, schedule, start_n, goal_n, grid,
                                 num_steps=args.ddim_steps, x_init=z)
            paths = (x0 + 1.0) * scale
            o = evaluate_paths(paths, risk_model, torch.full((n,), 0.1), t_steps, world, d_safe)
            o["path"] = paths
            outs.append(o)
        # 两两 RMS
        pair_rms = [float((outs[a]["path"] - outs[b]["path"]).pow(2).mean().sqrt())
                    for a in range(len(outs)) for b in range(a + 1, len(outs))]
        risky = [o["hard_risk"] for o in outs]
        sens_rows.append({
            "task_id": t["task_id"], "kind": t["kind"],
            "pairwise_rms_mean_m": float(np.mean(pair_rms)),
            "pairwise_rms_max_m": float(np.max(pair_rms)),
            "risk_mean": float(np.mean(risky)), "risk_std": float(np.std(risky)),
            "risk_min": float(np.min(risky)), "risk_max": float(np.max(risky)),
            "path_length_std_m": float(np.std([o["path_length_mean"] for o in outs])),
            "n_noise": args.noise_samples,
        })
        logger.info("任务 %s：不同噪声的轨迹两两 RMS 平均 %.3f m（最大 %.3f），硬风险 %.3f±%.3f",
                    t["task_id"], sens_rows[-1]["pairwise_rms_mean_m"],
                    sens_rows[-1]["pairwise_rms_max_m"], sens_rows[-1]["risk_mean"],
                    sens_rows[-1]["risk_std"])
    sens_ok = float(np.mean([r["pairwise_rms_mean_m"] for r in sens_rows])) > 0.01

    # 可视化：第一个任务的 20 条轨迹
    t0 = tasks[0]
    n = args.agents
    grid0 = ds.maps[t0["map_index"]].occupancy(
        torch.float32, grid_size=int(cfg["diffusion"].get("map_input_res",
                                                          cfg["world"]["grid_size"]))
    ).unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    draw_map(ax, ds.maps[t0["map_index"]])
    for k in range(min(args.noise_samples, 20)):
        z = torch.randn(n, h, 2, generator=torch.Generator().manual_seed(1000 + k))
        with torch.no_grad():
            x0 = ddim_sample(model, schedule, t0["starts"] / scale - 1.0, t0["goals"] / scale - 1.0,
                             grid0, num_steps=args.ddim_steps, x_init=z)
        paths = ((x0 + 1.0) * scale).numpy()
        for i in range(n):
            ax.plot(paths[i, :, 0], paths[i, :, 1], color=AGENT_COLORS[i % len(AGENT_COLORS)],
                    lw=1.0, alpha=0.35)
    ax.set_title(f"同一任务、20 个不同初始噪声的轨迹（RMS {sens_rows[0]['pairwise_rms_mean_m']:.3f} m）",
                 fontsize=10)
    fig.tight_layout()
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "plots" / "s5_noise_sensitivity.png", dpi=130)
    plt.close(fig)

    if not sens_ok:
        logger.warning("噪声敏感性不足（< 1 cm），按方案暂停噪声优化")
        report = StageReport("S5_noise_optimization", run_dir,
                             "S5 初始噪声敏感性报告（未进入优化）", cfg.to_dict())
        report.table("s5_noise_sensitivity", sens_rows)
        report.add_section(
            "## 结论\n\n"
            f"不同初始噪声产生的轨迹两两 RMS 平均 "
            f"{np.mean([r['pairwise_rms_mean_m'] for r in sens_rows]):.4f} m，低于 0.01 m 门槛。\n\n"
            "**暂停初始噪声优化**，先诊断模型为何不利用初始噪声。\n")
        report.write_report()
        report.write_metrics_csv()
        report.write_status("blocked", {}, {"reason": "noise_sensitivity"})
        report.sync_to_docs()
        print("噪声敏感性不足，已停止")
        return 0

    # ---------------- B. B0 / B1 / B2 / R 对照 ----------------
    log_section(logger, "B. B0 / B1 / B2 / 随机扰动 对照")
    opt_cfg = NoiseOptConfig(steps=args.iters, lr=args.lr, trust_radius=args.trust_radius,
                             ddim_steps=args.ddim_steps,
                             lambda_R=float(cfg["guidance"]["lambda_R"]),
                             lambda_T=float(cfg["guidance"]["lambda_T"]),
                             lambda_S=float(cfg["guidance"]["lambda_S"]),
                             lambda_K=float(cfg["guidance"]["lambda_K"]),
                             lambda_G=float(cfg["guidance"]["lambda_G"]))
    rows = []
    t_start = time.time()
    for ti, t in enumerate(tasks):
        n = args.agents
        start, goal = t["starts"], t["goals"]
        start_n, goal_n = start / scale - 1.0, goal / scale - 1.0
        world = ds.maps[t["map_index"]]
        res = int(cfg["diffusion"].get("map_input_res", cfg["world"]["grid_size"]))
        grid = world.occupancy(torch.float32, grid_size=res).unsqueeze(0).unsqueeze(0).expand(
            n, -1, -1, -1)
        pause_p = torch.full((n,), float(cfg["execution"]["pause_probability"]))
        z0 = torch.randn(n, h, 2, generator=torch.Generator().manual_seed(5000 + ti))

        # B0
        t0 = time.time()
        with torch.no_grad():
            x0_b0 = ddim_sample(model, schedule, start_n, goal_n, grid,
                                num_steps=args.ddim_steps, x_init=z0)
        b0 = evaluate_paths((x0_b0 + 1.0) * scale, risk_model, pause_p, t_steps, world, d_safe)
        b0.update({"method": "B0", "task_id": t["task_id"], "kind": t["kind"],
                   "noise_rms": 0.0, "elapsed_s": time.time() - t0})

        # B1（原方法：逐步引导，关闭等待）
        t0 = time.time()
        g = GuidanceConfig(enabled=True, start_fraction=float(cfg["diffusion"]["guidance_start_fraction"]),
                           step_size_x=0.2, step_size_w=0.0, grad_clip_norm=1.0,
                           max_wait_per_point=0.0, max_total_wait=0.0,
                           wait_mask_mode="all", optimize_wait=False, optimize_path=True,
                           hard_eval_every=0, updates_per_step=1)
        out_b1 = guided_ddim_sample(model, schedule, risk_model, cfg, start, goal, grid, pause_p,
                                    obstacles=world.obstacles(), num_steps=args.ddim_steps,
                                    guide=g, time_steps=t_steps, dt_s=dt_s, d_safe=d_safe,
                                    x_init=z0.clone())
        b1 = evaluate_paths(out_b1.paths_m, risk_model, pause_p, t_steps, world, d_safe)
        b1.update({"method": "B1", "task_id": t["task_id"], "kind": t["kind"],
                   "noise_rms": 0.0, "elapsed_s": time.time() - t0})

        # B2（初始噪声优化）
        t0 = time.time()
        opt = optimize_initial_noise(model, schedule, risk_model, cfg, start, goal, grid,
                                     pause_p, z0.clone(), obstacles=world.obstacles(),
                                     opt=opt_cfg, time_steps=t_steps, dt_s=dt_s)
        b2 = evaluate_paths(opt.paths_m, risk_model, pause_p, t_steps, world, d_safe)
        b2.update({"method": "B2", "task_id": t["task_id"], "kind": t["kind"],
                   "noise_rms": float((opt.z - z0).pow(2).mean().sqrt()),
                   "elapsed_s": time.time() - t0})

        # R（同信赖域随机扰动）
        t0 = time.time()
        zr = random_perturbation_control(z0, args.trust_radius,
                                         torch.Generator().manual_seed(9000 + ti))
        with torch.no_grad():
            x0_r = ddim_sample(model, schedule, start_n, goal_n, grid,
                               num_steps=args.ddim_steps, x_init=zr)
        rr = evaluate_paths((x0_r + 1.0) * scale, risk_model, pause_p, t_steps, world, d_safe)
        rr.update({"method": "R", "task_id": t["task_id"], "kind": t["kind"],
                   "noise_rms": float((zr - z0).pow(2).mean().sqrt()),
                   "elapsed_s": time.time() - t0})

        for r in (b0, b1, b2, rr):
            r.pop("path", None)
            rows.append(r)
        logger.info("任务 %d/%d：B0 %.4f | B1 %.4f | B2 %.4f | R %.4f",
                    ti + 1, len(tasks), b0["hard_risk"], b1["hard_risk"], b2["hard_risk"],
                    rr["hard_risk"])

    # ---------------- 汇总 ----------------
    by_method = {}
    for m in ("B0", "B1", "B2", "R"):
        sel = [r for r in rows if r["method"] == m]
        vals = np.array([r["hard_risk"] for r in sel])
        clusters = np.array([r["task_id"] for r in sel])
        by_method[m] = {
            "n": len(sel),
            "hard_risk_mean": float(vals.mean()),
            "hard_risk_ci": cluster_bootstrap_ci(vals, clusters, n_boot=2000),
            "static_safe_rate": float(np.mean([r["static_safe"] for r in sel])),
            "pair_min_dist_mean": float(np.mean([r["pair_min_dist"] for r in sel])),
            "path_length_mean": float(np.mean([r["path_length_mean"] for r in sel])),
            "smoothness_mean": float(np.mean([r["smoothness"] for r in sel])),
            "noise_rms_mean": float(np.mean([r["noise_rms"] for r in sel])),
            "elapsed_s_mean": float(np.mean([r["elapsed_s"] for r in sel])),
        }

    def paired(a, b):
        da = {r["task_id"]: r["hard_risk"] for r in rows if r["method"] == a}
        db = {r["task_id"]: r["hard_risk"] for r in rows if r["method"] == b}
        keys = sorted(set(da) & set(db))
        return paired_bootstrap_diff(np.array([da[k] for k in keys]),
                                     np.array([db[k] for k in keys]),
                                     np.array(keys), n_boot=2000)

    comparisons = {"B2_vs_B0": paired("B2", "B0"), "B2_vs_B1": paired("B2", "B1"),
                   "B2_vs_R": paired("B2", "R")}
    for k, v in comparisons.items():
        logger.info("%s 硬风险差 %+.4f [%+.4f, %+.4f] %s", k, v["diff"], v["lo"], v["hi"],
                    "显著" if v["significant"] else "不显著")

    # ---------------- 图与报告 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    methods = ["B0", "B1", "B2", "R"]
    xs = np.arange(len(methods))
    vals = [by_method[m]["hard_risk_mean"] for m in methods]
    los = [by_method[m]["hard_risk_ci"]["lo"] for m in methods]
    his = [by_method[m]["hard_risk_ci"]["hi"] for m in methods]
    axes[0].bar(xs, vals, color=["#4c72b0", "#dd8452", "#55a868", "#8c8c8c"])
    axes[0].errorbar(xs, vals, yerr=[np.array(vals) - los, np.array(his) - np.array(vals)],
                     fmt="none", ecolor="k", capsize=4)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("硬首次碰撞风险（均值）")
    axes[0].set_title("四种方法的硬风险（任务级 95% CI）", fontsize=10)
    axes[0].grid(alpha=0.3, axis="y")

    for m, c in zip(methods, ["#4c72b0", "#dd8452", "#55a868", "#8c8c8c"]):
        sel = [r for r in rows if r["method"] == m]
        axes[1].scatter([r["noise_rms"] for r in sel], [r["hard_risk"] for r in sel],
                        label=m, color=c, s=30, alpha=0.8)
    axes[1].set_xlabel("初始噪声偏移 RMS")
    axes[1].set_ylabel("硬风险")
    axes[1].set_title("噪声偏移与收益", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "s5_method_comparison.png", dpi=130)
    plt.close(fig)

    report = StageReport("S5_noise_optimization", run_dir,
                         "S5 初始噪声优化报告", cfg.to_dict())
    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    report.metric("ckpt", args.ckpt, group="setup")
    report.metric("noise_sensitivity_mean_rms_m",
                  float(np.mean([r["pairwise_rms_mean_m"] for r in sens_rows])), group="sensitivity")
    for m, v in by_method.items():
        report.metric(f"{m}/hard_risk", v["hard_risk_mean"], group="method")
        report.metric(f"{m}/static_safe_rate", v["static_safe_rate"], group="method")
        report.metric(f"{m}/noise_rms", v["noise_rms_mean"], group="method")
        report.metric(f"{m}/elapsed_s", v["elapsed_s_mean"], group="method")
    for k, v in comparisons.items():
        report.metric(f"{k}/diff", v["diff"], group="paired")
        report.metric(f"{k}/lo", v["lo"], group="paired")
        report.metric(f"{k}/hi", v["hi"], group="paired")
        report.metric(f"{k}/significant", v["significant"], group="paired")
    report.table("s5_noise_sensitivity", sens_rows)
    report.table("s5_methods", rows)

    lines = ["| 方法 | 硬风险均值 | 95% CI | 静态安全率 | 最小机器人间距 | 噪声偏移 | 耗时 [s] |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for m in methods:
        v = by_method[m]
        lines.append(f"| {m} | {v['hard_risk_mean']:.4f} | "
                     f"[{v['hard_risk_ci']['lo']:.4f}, {v['hard_risk_ci']['hi']:.4f}] | "
                     f"{v['static_safe_rate']:.2f} | {v['pair_min_dist_mean']:.3f} | "
                     f"{v['noise_rms_mean']:.3f} | {v['elapsed_s_mean']:.1f} |")
    comp_lines = ["| 对照 | 硬风险差 | 95% CI | 显著 |", "|---|---:|---|---|"]
    for k, v in comparisons.items():
        comp_lines.append(f"| {k} | {v['diff']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | "
                          f"{'是' if v['significant'] else '否'} |")

    report.add_section(
        "## A. 噪声敏感性\n\n"
        f"- 每个任务 {args.noise_samples} 个初始噪声，轨迹两两 RMS 平均 "
        f"**{np.mean([r['pairwise_rms_mean_m'] for r in sens_rows]):.3f} m**，"
        f"最大 {np.max([r['pairwise_rms_max_m'] for r in sens_rows]):.3f} m\n"
        f"- 硬风险跨噪声的标准差平均 "
        f"{np.mean([r['risk_std'] for r in sens_rows]):.4f}\n"
        "- 结论：模型确实利用初始噪声产生不同轨迹，可以进入噪声优化\n"
    )
    report.add_section("## B. 方法对照\n\n" + "\n".join(lines) + "\n\n" + "\n".join(comp_lines) + "\n")
    report.add_section(
        "## 判定\n\n"
        f"- B2 相对 B0：{comparisons['B2_vs_B0']['diff']:+.4f}"
        f"（{'显著' if comparisons['B2_vs_B0']['significant'] else '不显著'}）\n"
        f"- B2 相对 B1：{comparisons['B2_vs_B1']['diff']:+.4f}"
        f"（{'显著' if comparisons['B2_vs_B1']['significant'] else '不显著'}）\n"
        f"- B2 相对随机扰动 R：{comparisons['B2_vs_R']['diff']:+.4f}"
        f"（{'显著' if comparisons['B2_vs_R']['significant'] else '不显著'}）\n\n"
        "只有同时优于 B1 与 R，才能认为「初始噪声优化」这个机制本身有效；"
        "否则收益可能只是「换了条轨迹」或「随机搜索」。\n"
    )
    report.write_report()
    report.write_metrics_csv()
    report.write_status("completed", {}, {"comparisons": comparisons})
    report.sync_to_docs()

    import shutil
    docs = ROOT / "docs" / "revised"
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "report.md", docs / "04_S5_noise.md")
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, docs / "figures" / png.name)

    print(json.dumps({"hard_risk": {m: round(by_method[m]["hard_risk_mean"], 4) for m in methods},
                      "comparisons": {k: round(v["diff"], 4) for k, v in comparisons.items()}},
                     ensure_ascii=False))
    print(f"运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
