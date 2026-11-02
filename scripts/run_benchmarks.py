"""S5：完整二维离散执行对照实验（文档 §8、§9 S5、§10）。

流程：
  1. 构造任务集：4 类地图 × {2,4,6} 台 × 每格若干任务，起点/终点随机且 A* 可行；
  2. 每个任务用 7 种方法各规划一次（共享噪声、共享 DDIM 步数、共享引导预算）；
  3. 每个规划在 4 个暂停强度下各做 N 次独立随机执行；
  4. 统计“全尝试安全完成率”、规划成功率、执行碰撞率、完成时间、等待、耗时，
     并给出以任务为簇的 bootstrap 95% 置信区间；
  5. 输出曲线、消融图、失败归因与中文报告。

用法：
  python scripts/run_benchmarks.py --config configs/benchmark.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.data.astar import build_grid, plan_path  # noqa: E402
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.diffusion.train import build_model  # noqa: E402
from fhd.evaluation.baselines import METHODS, plan_with_method  # noqa: E402
from fhd.evaluation.metrics import (  # noqa: E402
    attribute_failure,
    execution_metrics,
    plan_metrics,
)
from fhd.evaluation.reporting import StageReport, update_project_progress  # noqa: E402
from fhd.evaluation.statistics import cluster_bootstrap_ci, paired_bootstrap_diff, wilson_interval  # noqa: E402
from fhd.evaluation.visualizations import AGENT_COLORS  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.maps.generate import generate_map  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

TASK_SEED_OFFSET = 500_000
SHARED_NOISE_SEED = 20261103


def make_task(kind: str, n_agents: int, cfg, task_index: int, rng: np.random.Generator) -> dict | None:
    """采样一个 A* 可行的 N 机器人任务（只用于提供起终点与可行性判定）。"""
    seed = TASK_SEED_OFFSET + task_index
    try:
        world = generate_map(kind, cfg, seed)
    except RuntimeError:
        return None
    resolution = int(cfg["dataset"]["astar_resolution"])
    h = int(cfg["trajectory"]["horizon"])
    grid = build_grid(world, resolution)
    starts, goals, refs = [], [], []
    for _ in range(n_agents):
        ok = False
        for _try in range(12):
            try:
                pair = world.sample_free_points(rng, 2, min_sep=float(cfg["dataset"]["min_start_goal_dist_m"]),
                                                border_m=0.6)
            except RuntimeError:
                return None
            s, g = pair[0], pair[1]
            p = plan_path(world, s, g, resolution, h, rng=rng, noise_scale=0.0, grid=grid)
            if p is None:
                continue
            diag_len = float(torch.linalg.vector_norm(p[-1] - p[0]))
            if diag_len < float(cfg["dataset"]["min_start_goal_dist_m"]):
                continue
            starts.append(torch.as_tensor(s, dtype=torch.float64))
            goals.append(torch.as_tensor(g, dtype=torch.float64))
            refs.append(p)
            ok = True
            break
        if not ok:
            return None
    # 机器人之间初始位置须互不碰撞
    starts_t = torch.stack(starts)
    dmat = torch.cdist(starts_t, starts_t) + torch.eye(n_agents, dtype=torch.float64) * 1e9
    if float(dmat.min()) < 0.35:
        return None
    return {
        "kind": kind, "n_agents": n_agents, "seed": seed,
        "world": world, "starts": starts_t, "goals": torch.stack(goals),
        "ref_paths": torch.stack(refs),
    }


def calibrate_risk_weights(
    cfg, model, schedule, risk_model, device, calib_tasks, methods=("A3", "B3"),
    lambdas=(0.05, 0.5, 2.0, 10.0, 25.0), ddim_steps: int = 20, logger=None,
) -> tuple[dict, list[dict]]:
    """在**独立验证任务**上为 A3 与 B3 分别选择风险权重（文档 §8.1）。

    两者使用完全相同的候选网格与搜索预算，避免“只给某一方调参”的偏袒。
    选择准则：在规划有效的任务上使平均硬风险最小；若并列，取等待更少的。
    """
    import copy as _copy

    rows: list[dict] = []
    best: dict[str, float] = {}
    for method in methods:
        candidates = []
        for lam in lambdas:
            cfg_i = _copy.deepcopy(cfg)
            cfg_i["guidance"]["lambda_R"] = float(lam)
            risks, waits, valids = [], [], []
            for task in calib_tasks:
                n = task["n_agents"]
                start_m = task["starts"].to(torch.float32)
                goal_m = task["goals"].to(torch.float32)
                grid = task["world"].occupancy(torch.float32).unsqueeze(0).unsqueeze(0).expand(
                    n, -1, -1, -1)
                pause_p = torch.full((n,), float(cfg_i["execution"]["pause_probability"]),
                                     dtype=torch.float32)
                gen = torch.Generator(device="cpu").manual_seed(424242 + int(lam * 100))
                x_init = torch.randn(n, int(cfg_i["trajectory"]["horizon"]), 2, generator=gen)
                try:
                    res = plan_with_method(
                        method, model, schedule, risk_model, cfg_i, start_m, goal_m, grid,
                        pause_p, task["world"].obstacles(), num_steps=ddim_steps,
                        x_init=x_init, d_safe=risk_model.d_safe,
                        time_steps=int(cfg_i["execution"]["horizon_steps"]),
                        dt_s=float(cfg_i["trajectory"]["dt_s"]))
                except Exception:  # noqa: BLE001
                    continue
                plan = plan_metrics(res.paths_m, res.wait, task["starts"].to(torch.float32),
                                    task["goals"].to(torch.float32), task["world"],
                                    float(cfg_i["trajectory"]["max_speed_mps"]),
                                    float(cfg_i["trajectory"]["dt_s"]))
                risks.append(res.final["hard_risk_sum"])
                waits.append(res.final["wait_total"])
                valids.append(1.0 if plan.valid else 0.0)
            if not risks:
                continue
            row = {
                "method": method, "lambda_R": float(lam),
                "mean_hard_risk": float(np.mean(risks)),
                "mean_wait": float(np.mean(waits)),
                "plan_valid_rate": float(np.mean(valids)),
            }
            rows.append(row)
            candidates.append(row)
            if logger:
                logger.info("[校准] %s λ_R=%.1f：硬风险 %.4f，等待 %.2f，有效 %.2f",
                            method, lam, row["mean_hard_risk"], row["mean_wait"],
                            row["plan_valid_rate"])
        if candidates:
            candidates.sort(key=lambda r: (r["mean_hard_risk"], r["mean_wait"]))
            best[method] = candidates[0]["lambda_R"]
    return best, rows


def main() -> int:
    ap = argparse.ArgumentParser(description="S5 对照实验")
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tasks-per-cell", type=int, default=None)
    ap.add_argument("--methods", default=None, help="逗号分隔，如 B0,A3")
    ap.add_argument("--robot-counts", default=None, help="逗号分隔，如 2,4")
    ap.add_argument("--pause-probs", default=None, help="逗号分隔，如 0,0.1")
    ap.add_argument("--calibrate", action="store_true",
                    help="在独立验证任务上为 A3/B3 分别校准风险权重（文档 §8.1）")
    ap.add_argument("--calib-tasks", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = torch.device("cpu")           # 引导阶段风险 DP 在 CPU 上更合适（小张量、96 步串行）
    run_dir = new_run_dir("S5_evaluation", "s5_benchmark")
    logger = get_logger("fhd.s5", run_dir / "logs" / "s5.log")
    log_section(logger, "S5 二维离散执行对照实验")

    b = cfg["benchmark"]
    tasks_per_cell = int(args.tasks_per_cell or b["tasks_per_cell"])
    robot_counts = [int(x) for x in (args.robot_counts.split(",") if args.robot_counts
                                     else b["robot_counts"])]
    pause_probs = [float(x) for x in (args.pause_probs.split(",") if args.pause_probs
                                      else b["pause_probabilities"])]
    methods = args.methods.split(",") if args.methods else list(b["methods"])
    repeats = int(b["execution_repeats"])
    ddim_steps = int(b["ddim_steps"])
    n_boot = int(b["n_bootstrap"])

    # ------------------------------------------------------------------
    # 模型与任务集
    # ------------------------------------------------------------------
    model = build_model(cfg, device)
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        cands = sorted((ROOT / "results" / "S3_diffusion_training").glob("*/checkpoints/best.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        ckpt_path = cands[0] if cands else None
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit("未找到 Diffusion checkpoint")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("模型 %s（step=%s），设备=%s，方法=%s", ckpt_path, ckpt.get("step"), device, methods)

    schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                            kind=str(cfg["diffusion"]["beta_schedule"]), device=device)
    risk_model = FirstHitRisk.from_config(cfg)
    d_safe = risk_model.d_safe
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    h = int(cfg["trajectory"]["horizon"])
    w_m = float(cfg["world"]["width_m"])
    h_m = float(cfg["world"]["height_m"])
    scale = torch.tensor([w_m / 2.0, h_m / 2.0], dtype=torch.float32)
    p_assumed = float(cfg["execution"]["pause_probability"])

    log_section(logger, "1. 构造任务集")
    rng = np.random.default_rng(int(cfg["seed"]))
    tasks = []
    idx = 0
    for kind in b["map_kinds"]:
        for n_agents in robot_counts:
            made = 0
            attempt = 0
            while made < tasks_per_cell and attempt < tasks_per_cell * 8:
                attempt += 1
                idx += 1
                task = make_task(kind, n_agents, cfg, idx, rng)
                if task is None:
                    continue
                tasks.append(task)
                made += 1
            logger.info("地图 %-14s N=%d：构造 %d 个任务", kind, n_agents, made)
    logger.info("任务总数 %d", len(tasks))

    # ------------------------------------------------------------------
    # 风险权重校准（A3 与 B3 独立选参，预算相同）
    # ------------------------------------------------------------------
    lambda_r = {m: float(cfg["guidance"]["lambda_R"]) for m in methods}
    calib_rows: list[dict] = []
    if args.calibrate and "A3" in methods and "B3" in methods:
        log_section(logger, "1b. 风险权重校准（独立验证任务）")
        calib_tasks = []
        calib_rng = np.random.default_rng(int(cfg["seed"]) + 777)
        cidx = 900_000
        for kind in ("open", "dual_corridor", "bottleneck"):
            made = 0
            attempt = 0
            while made < args.calib_tasks and attempt < args.calib_tasks * 8:
                attempt += 1
                cidx += 1
                t = make_task(kind, 2, cfg, cidx, calib_rng)
                if t is None:
                    continue
                calib_tasks.append(t)
                made += 1
        logger.info("校准任务数 %d", len(calib_tasks))
        lambda_r, calib_rows = calibrate_risk_weights(
            cfg, model, schedule, risk_model, torch.device("cpu"), calib_tasks,
            methods=("A3", "B3"), lambdas=(0.05, 0.5, 2.0, 10.0, 25.0), ddim_steps=ddim_steps,
            logger=logger,
        )
        logger.info("校准结果：%s", lambda_r)
        for m, lam in list(lambda_r.items()):
            if m == "B3":
                lambda_r["B1"] = lam
                lambda_r["B2"] = lam

    # ------------------------------------------------------------------
    # 规划
    # ------------------------------------------------------------------
    log_section(logger, "2. 逐方法规划")
    plan_records = []
    plan_arrays: dict[str, dict] = {}
    for ti, task in enumerate(tasks):
        n = task["n_agents"]
        world = task["world"]
        start_m = task["starts"].to(torch.float32)
        goal_m = task["goals"].to(torch.float32)
        grid = world.occupancy(torch.float32).unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1)
        obstacles = world.obstacles()
        pause_p = torch.full((n,), p_assumed, dtype=torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(SHARED_NOISE_SEED + ti)
        x_init = torch.randn(n, h, 2, generator=gen, device=device)

        for method in methods:
            t0 = time.time()
            cfg_m = cfg
            cfg_m["guidance"]["lambda_R"] = float(lambda_r.get(method, cfg["guidance"]["lambda_R"]))
            try:
                res = plan_with_method(method, model, schedule, risk_model, cfg_m, start_m, goal_m,
                                       grid, pause_p, obstacles, num_steps=ddim_steps,
                                       generator=None, x_init=x_init.clone(), d_safe=d_safe,
                                       time_steps=t_steps, dt_s=dt_s)
                plan = plan_metrics(res.paths_m, res.wait, task["starts"].to(torch.float32),
                                    task["goals"].to(torch.float32), world,
                                    float(cfg["trajectory"]["max_speed_mps"]), dt_s)
                status = "ok" if plan.valid else "invalid"
            except Exception as exc:  # noqa: BLE001
                logger.warning("任务 %d 方法 %s 规划异常: %s", ti, method, exc)
                plan_records.append({"task": ti, "kind": task["kind"], "n_agents": n,
                                     "method": method, "status": "error", "error": str(exc),
                                     "plan_time_s": time.time() - t0, "valid": False,
                                     "static_safe": False, "speed_ok": False,
                                     "endpoint_err_m": float("nan"), "min_clearance_m": float("nan"),
                                     "max_segment_m": float("nan"), "path_length_mean_m": float("nan"),
                                     "smoothness": float("nan"), "wait_total": 0.0, "wait_points": 0,
                                     "hard_risk": float("nan"), "soft_risk": float("nan")})
                continue

            with torch.no_grad():
                hard = risk_model(res.paths_m, res.wait, pause_p, t_steps, collision_mode="hard",
                                  gate_mode="hard", return_diagnostics=False)
                soft = risk_model(res.paths_m, res.wait, pause_p, t_steps, collision_mode="soft",
                                  gate_mode="soft", return_diagnostics=False)
            record = {
                "task": ti, "kind": task["kind"], "n_agents": n, "method": method,
                "status": status, "valid": plan.valid,
                "static_safe": plan.static_safe, "speed_ok": plan.speed_ok,
                "endpoint_err_m": plan.endpoint_err_m,
                "min_clearance_m": plan.min_clearance_m,
                "max_segment_m": plan.max_segment_m,
                "path_length_mean_m": plan.path_length_mean_m,
                "smoothness": plan.smoothness,
                "wait_total": plan.wait_total, "wait_points": plan.wait_points,
                "hard_risk": float(hard["pair_sum_risk"]),
                "soft_risk": float(soft["pair_sum_risk"]),
                "plan_time_s": time.time() - t0,
            }
            plan_records.append(record)
            plan_arrays[f"{ti}_{method}"] = {
                "paths": res.paths_m.detach().numpy(),
                "wait": res.wait.detach().numpy(),
                "starts": task["starts"].numpy(), "goals": task["goals"].numpy(),
            }
        if (ti + 1) % 6 == 0:
            logger.info("已规划 %d/%d 个任务（%d 条规划记录）", ti + 1, len(tasks), len(plan_records))

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    log_section(logger, "3. 随机执行复现")
    exec_records = []
    for rec in plan_records:
        key = f"{rec['task']}_{rec['method']}"
        if key not in plan_arrays:
            continue
        arr = plan_arrays[key]
        paths = torch.as_tensor(arr["paths"])
        wait = torch.as_tensor(arr["wait"])
        n = paths.shape[0]
        for p_exec in pause_probs:
            pause_p = torch.full((n,), p_exec, dtype=paths.dtype)
            batch = simulate_execution(
                paths, wait, pause_p, d_safe, t_steps, dt_s, n_samples=repeats,
                chunk_size=repeats, generator=torch.Generator().manual_seed(int(p_exec * 1000) + rec["task"]),
            )
            em = execution_metrics(batch, dt_s)
            safe_completion = bool(rec["valid"]) and em.safe_completion_rate
            exec_records.append({
                "task": rec["task"], "method": rec["method"], "kind": rec["kind"],
                "n_agents": rec["n_agents"], "pause_p": p_exec, "valid_plan": rec["valid"],
                "collision_rate": em.collision_rate,
                "safe_completion_rate": em.safe_completion_rate if rec["valid"] else 0.0,
                "execution_success": em.safe_completion_rate if rec["valid"] else 0.0,
                "completion_rate": em.completion_rate,
                "makespan_mean_s": em.makespan_mean_s,
                "makespan_p50_s": em.makespan_p50_s,
                "makespan_p90_s": em.makespan_p90_s,
                "unfinished_agent_rate": em.unfinished_agent_rate,
                "hard_risk": rec["hard_risk"],
                "failure": attribute_failure_from(rec, em) if rec["valid"] else rec["status"],
            })

    # ------------------------------------------------------------------
    # 聚合与统计
    # ------------------------------------------------------------------
    log_section(logger, "4. 聚合与置信区间")
    plan_by_method = defaultdict(list)
    for rec in plan_records:
        plan_by_method[rec["method"]].append(rec)

    aggregate = {}
    for method in methods:
        rows = plan_by_method[method]
        if not rows:
            continue
        valid = np.array([1.0 if r["valid"] else 0.0 for r in rows])
        times = np.array([r["plan_time_s"] for r in rows], dtype=float)
        waits = np.array([r["wait_total"] for r in rows], dtype=float)
        risks = np.array([r["hard_risk"] for r in rows], dtype=float)
        clusters = np.array([r["task"] for r in rows])
        aggregate[method] = {
            "n_plans": len(rows),
            "plan_valid_rate": float(valid.mean()),
            "plan_valid_ci": cluster_bootstrap_ci(valid, clusters, n_boot=n_boot),
            "plan_time_mean_s": float(np.nanmean(times)),
            "wait_total_mean": float(np.nanmean(waits)),
            "hard_risk_mean": float(np.nanmean(risks)),
        }
        logger.info("%-3s 规划成功率 %.3f | 平均耗时 %.2fs | 平均等待 %.2f 时隙 | 平均硬风险 %.4f",
                    method, aggregate[method]["plan_valid_rate"],
                    aggregate[method]["plan_time_mean_s"],
                    aggregate[method]["wait_total_mean"],
                    aggregate[method]["hard_risk_mean"])

    exec_agg = {}
    for method in methods:
        for p_exec in pause_probs:
            sel = [r for r in exec_records if r["method"] == method and r["pause_p"] == p_exec]
            if not sel:
                continue
            vals = np.array([r["safe_completion_rate"] for r in sel], dtype=float)
            clusters = np.array([r["task"] for r in sel])
            # 每个 (task, repeat) 粒度的成功标记
            expanded_vals, expanded_clusters = [], []
            for r in sel:
                k = int(round(r["safe_completion_rate"] * repeats))
                expanded_vals.extend([1.0] * k + [0.0] * (repeats - k))
                expanded_clusters.extend([r["task"]] * repeats)
            ci = cluster_bootstrap_ci(np.array(expanded_vals), np.array(expanded_clusters),
                                      n_boot=n_boot)
            total_k = sum(int(round(r["safe_completion_rate"] * repeats)) for r in sel)
            total_n = len(sel) * repeats
            p_hat, lo, hi = wilson_interval(total_k, total_n)
            exec_agg[(method, p_exec)] = {
                "n": total_n, "successes": total_k,
                "safe_completion_rate": p_hat, "wilson_lo": lo, "wilson_hi": hi,
                "collision_rate": float(np.mean([r["collision_rate"] for r in sel])),
                "makespan_mean_s": float(np.mean([r["makespan_mean_s"] for r in sel])),
                "makespan_p90_s": float(np.mean([r["makespan_p90_s"] for r in sel])),
                "bootstrap": ci,
            }
            logger.info("%-3s p=%.2f：安全完成率 %.3f [%.3f, %.3f]，执行碰撞率 %.3f，makespan %.2fs",
                        method, p_exec, p_hat, lo, hi,
                        exec_agg[(method, p_exec)]["collision_rate"],
                        exec_agg[(method, p_exec)]["makespan_mean_s"])

    # 配对比较：A3 vs B3（核心创新性检验 G2）与 A3 vs B2（G3）
    paired = {}
    for base_method in ("B3", "B2", "B0"):
        a, c, cl = [], [], []
        by_key = defaultdict(dict)
        for r in exec_records:
            by_key[(r["task"], r["pause_p"])][r["method"]] = r["safe_completion_rate"]
        for (task, p_exec), d in by_key.items():
            if "A3" in d and base_method in d:
                a.append(d["A3"])
                c.append(d[base_method])
                cl.append(task)
        if a:
            paired[f"A3_vs_{base_method}"] = paired_bootstrap_diff(
                np.array(a), np.array(c), np.array(cl), n_boot=n_boot)
            logger.info("A3 - %s 安全完成率差 = %+.4f [%+.4f, %+.4f] %s",
                        base_method, paired[f"A3_vs_{base_method}"]["diff"],
                        paired[f"A3_vs_{base_method}"]["lo"], paired[f"A3_vs_{base_method}"]["hi"],
                        "显著" if paired[f"A3_vs_{base_method}"]["significant"] else "不显著")

    # ------------------------------------------------------------------
    # 图与表
    # ------------------------------------------------------------------
    report = StageReport("S5_evaluation", run_dir, "S5 完整对照实验报告", cfg.to_dict())
    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    report.table("plans", plan_records)
    report.table("executions", exec_records)
    for method, agg in aggregate.items():
        report.metrics_from_dict({f"{method}/{k}": v for k, v in agg.items()
                                  if not isinstance(v, dict)}, group="plan")
    for (method, p_exec), agg in exec_agg.items():
        for key, val in agg.items():
            if isinstance(val, dict):
                continue
            report.metric(f"{method}/p{p_exec}/{key}", val, group=f"execution/p{p_exec}")

    # 图 1：各方法安全完成率（按暂停概率）
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for mi, method in enumerate(methods):
        xs = [p for p in pause_probs if (method, p) in exec_agg]
        ys = [exec_agg[(method, p)]["safe_completion_rate"] for p in xs]
        los = [exec_agg[(method, p)]["wilson_lo"] for p in xs]
        his = [exec_agg[(method, p)]["wilson_hi"] for p in xs]
        axes[0].errorbar(xs, ys, yerr=[np.array(ys) - los, np.array(his) - np.array(ys)],
                         fmt="o-", capsize=3, label=f"{method} {METHODS[method].label}",
                         color=AGENT_COLORS[mi % len(AGENT_COLORS)])
        ys2 = [exec_agg[(method, p)]["collision_rate"] for p in xs]
        axes[1].plot(xs, ys2, "o-", label=method, color=AGENT_COLORS[mi % len(AGENT_COLORS)])
    axes[0].set_xlabel("执行暂停概率 p")
    axes[0].set_ylabel("全尝试安全完成率")
    axes[0].set_title("安全完成率随暂停概率变化（95% Wilson 区间）")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("执行暂停概率 p")
    axes[1].set_ylabel("执行碰撞率")
    axes[1].set_title("执行碰撞率")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    report.save_figure(fig, "safe_completion_vs_pause")

    # 图 2：规划时间与完成时间
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(len(methods))
    axes[0].bar(xs, [aggregate[m]["plan_time_mean_s"] for m in methods], color="#4c72b0")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("平均规划时间 [s]")
    axes[0].set_title("规划开销")
    axes[0].grid(alpha=0.3, axis="y")
    p_ref = pause_probs[len(pause_probs) // 2]
    axes[1].bar(xs, [exec_agg[(m, p_ref)]["makespan_mean_s"] for m in methods], color="#dd8452")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("平均完成时间 [s]")
    axes[1].set_title(f"完成时间（p={p_ref}）")
    axes[1].grid(alpha=0.3, axis="y")
    report.save_figure(fig, "plan_time_and_makespan")

    # 图 3：消融（A1/A2/A3 + B 系列）
    fig, ax = plt.subplots(figsize=(8, 5))
    for mi, method in enumerate(methods):
        sel = [exec_agg[(method, p)] for p in pause_probs if (method, p) in exec_agg]
        if not sel:
            continue
        ax.scatter([s["makespan_mean_s"] for s in sel],
                   [s["safe_completion_rate"] for s in sel],
                   s=60, label=f"{method}", color=AGENT_COLORS[mi % len(AGENT_COLORS)])
    ax.set_xlabel("平均完成时间 [s]")
    ax.set_ylabel("安全完成率")
    ax.set_title("安全—效率散点（每个点对应一个暂停强度）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    report.save_figure(fig, "safety_efficiency_scatter")

    # 图 4：失败归因
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cats = ["static_collision", "speed_violation", "endpoint_error",
            "inter_robot_collision", "unfinished", "ok"]
    bottom = np.zeros(len(methods))
    for ci, cat in enumerate(cats):
        vals = []
        for m in methods:
            rows = [r for r in exec_records if r["method"] == m]
            if not rows:
                vals.append(0)
                continue
            vals.append(sum(1 for r in rows if r["failure"] == cat) / len(rows))
        ax.bar(methods, vals, bottom=bottom, label=cat)
        bottom += np.array(vals)
    ax.set_ylabel("比例")
    ax.set_title("失败归因（按执行记录）")
    ax.legend(fontsize=8, loc="lower right")
    report.save_figure(fig, "failure_attribution")

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    best_p = max(pause_probs)
    a3 = exec_agg.get(("A3", best_p), {})
    b3 = exec_agg.get(("B3", best_p), {})
    g2 = paired.get("A3_vs_B3", {})
    report.add_section(
        "## 实验设计\n\n"
        f"- 任务：{len(tasks)} 个（4 类地图 × {robot_counts} 台），每格 {tasks_per_cell} 个；\n"
        f"- 方法：{', '.join(methods)}（共享同一 Diffusion 权重、同一噪声、{ddim_steps} 步 DDIM、"
        "相同引导预算）；\n"
        f"- 执行：每份规划在 p∈{pause_probs} 下各做 {repeats} 次独立随机执行；\n"
        "- 成功口径：规划有效 + 执行中全部到达 + 执行全程无碰撞（失败规划计 0）；\n"
        f"- 置信区间：以任务为簇的 bootstrap（{n_boot} 次重采样）。\n"
    )
    lines = ["| 方法 | 规划成功率 | 平均耗时 [s] | 平均等待 | 平均硬风险 |",
             "|---|---|---|---|---|"]
    for m in methods:
        a = aggregate.get(m, {})
        lines.append(f"| {m} {METHODS[m].label} | {a.get('plan_valid_rate', 0):.3f} | "
                     f"{a.get('plan_time_mean_s', 0):.2f} | {a.get('wait_total_mean', 0):.2f} | "
                     f"{a.get('hard_risk_mean', 0):.4f} |")
    lines += ["", f"## 执行结果（p={best_p}）", "",
              "| 方法 | 安全完成率 | 95% 区间 | 碰撞率 | makespan [s] |", "|---|---|---|---|---|"]
    for m in methods:
        e = exec_agg.get((m, best_p))
        if e:
            lines.append(f"| {m} | {e['safe_completion_rate']:.3f} | "
                         f"[{e['wilson_lo']:.3f}, {e['wilson_hi']:.3f}] | "
                         f"{e['collision_rate']:.3f} | {e['makespan_mean_s']:.2f} |")
    report.add_section("## 结果\n\n" + "\n".join(lines) + "\n")
    report.add_section(
        "## 关键检验（G2：首次碰撞吸收是否不可替代）\n\n"
        f"- A3 vs B3（逐时刻期望风险）：安全完成率差 {g2.get('diff', float('nan')):+.4f}，"
        f"95% 区间 [{g2.get('lo', float('nan')):+.4f}, {g2.get('hi', float('nan')):+.4f}]，"
        f"{'显著' if g2.get('significant') else '不显著'}；\n"
        f"- A3 安全完成率 {a3.get('safe_completion_rate', float('nan')):.3f}，"
        f"B3 {b3.get('safe_completion_rate', float('nan')):.3f}（p={best_p}）。\n"
    )
    report.add_section(
        "## 结论与局限\n\n"
        "- 本报告为**预实验规模**（配置 `benchmark.scale`），不可当作完整规模的最终结论；\n"
        "- 若 A3 与 B3 差异不显著，则“首次碰撞吸收”的独立贡献尚未被证明（G2 未达成）；\n"
        "- 所有失败均按 §10.2 口径计入分母，未做后验剔除。\n"
    )
    (run_dir / "aggregate.json").write_text(json.dumps({
        "plan": aggregate,
        "execution": {f"{m}|p{p}": v for (m, p), v in exec_agg.items()},
        "paired": paired,
        "n_tasks": len(tasks),
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (run_dir / "cases.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=float) for r in plan_records),
        encoding="utf-8")
    report.write_report()
    report.write_metrics_csv()
    report.write_status("completed", {}, {"n_tasks": len(tasks), "methods": methods})
    report.sync_to_docs()
    update_project_progress([
        {"stage": "S5", "summary": f"{len(tasks)} 个任务 × {len(methods)} 方法，"
                                   f"A3 安全完成率 {a3.get('safe_completion_rate', float('nan')):.3f}。"},
    ])
    print(json.dumps({"n_tasks": len(tasks), "aggregate": aggregate}, ensure_ascii=False,
                     indent=2, default=float))
    print(f"运行目录：{run_dir}")
    return 0


def attribute_failure_from(rec: dict, em) -> str:
    """按 §10.2 的失败归因口径分类。"""
    if rec.get("status") == "error":
        return "numeric_error"
    if not rec.get("static_safe", True):
        return "static_collision"
    if not rec.get("speed_ok", True):
        return "speed_violation"
    if rec.get("endpoint_err_m", 0.0) > 1e-3:
        return "endpoint_error"
    if em.unfinished_agent_rate > 0:
        return "unfinished"
    if em.collision_rate > 0:
        return "inter_robot_collision"
    return "ok"


if __name__ == "__main__":
    raise SystemExit(main())
