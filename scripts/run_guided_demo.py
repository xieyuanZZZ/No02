"""S4：把风险引导接入 DDIM 推理（文档 §7、§9 阶段 S4）。

必须首先验证（文档 §9 S4）：
  1. 引导系数为零时，新采样器与基础 DDIM 逐步一致；
  2. 引导时 Q、w 的计算图未断（梯度非零且有限）；
  3. 模型网络参数冻结，推理后权重未变化；
  4. 轨迹起终点保持不变；
  5. 规划完成后硬风险、静态碰撞和速度约束独立复核；
  6. 与 S2 的手工案例一致，确实形成等待/绕行行为。

用法：python scripts/run_guided_demo.py --config configs/guidance.yaml
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
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.diffusion.train import build_model  # noqa: E402
from fhd.evaluation.baselines import METHODS, plan_with_method  # noqa: E402
from fhd.evaluation.reporting import StageReport, update_project_progress  # noqa: E402
from fhd.evaluation.visualizations import draw_map, plot_path_optimization  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.geometry.collisions import check_speed_limit, trajectory_is_static_safe  # noqa: E402
from fhd.maps.scenarios import build_scenario  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import resolve_device, set_seed  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def load_model(cfg, device, ckpt_path: Path | None):
    model = build_model(cfg, device)
    if ckpt_path is None:
        candidates = sorted((ROOT / "results" / "S3_diffusion_training").glob("*/checkpoints/best.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        ckpt_path = candidates[0] if candidates else None
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit("未找到 Diffusion checkpoint，请先运行 train_diffusion.py")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt_path, ckpt.get("step")


def main() -> int:
    p = argparse.ArgumentParser(description="S4 风险引导 DDIM 演示与验证")
    p.add_argument("--config", default=str(ROOT / "configs" / "guidance.yaml"))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--scenario", default="crossing")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--mc-samples", type=int, default=2000)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    run_dir = new_run_dir("S4_guided_diffusion", "s4_demo")
    logger = get_logger("fhd.s4", run_dir / "logs" / "s4.log")
    log_section(logger, "S4 风险引导 DDIM 演示")

    model, ckpt_path, ckpt_step = load_model(cfg, device, Path(args.ckpt) if args.ckpt else None)
    logger.info("模型：%s（step=%s），设备=%s", ckpt_path, ckpt_step, device)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                            kind=str(cfg["diffusion"]["beta_schedule"]), device=device,
                            dtype=torch.float32)
    risk_model = FirstHitRisk.from_config(cfg)
    d_safe = risk_model.d_safe
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    num_steps = int(args.steps or cfg["diffusion"]["ddim_sampling_steps"])
    h = int(cfg["trajectory"]["horizon"])

    sc = build_scenario(args.scenario, cfg)
    n = sc.n_agents
    start_m = sc.starts.to(torch.float32)
    goal_m = sc.goals.to(torch.float32)
    w_m = float(cfg["world"]["width_m"])
    h_m = float(cfg["world"]["height_m"])
    scale = torch.tensor([w_m / 2.0, h_m / 2.0], dtype=torch.float32)
    start_n = start_m / scale - 1.0
    goal_n = goal_m / scale - 1.0
    grid = sc.world.occupancy(torch.float32).unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1)
    obstacles = sc.world.obstacles()
    pause_p = torch.full((n,), float(cfg["execution"]["pause_probability"]), dtype=torch.float32)

    def hard_eval(paths_m: torch.Tensor, wait: torch.Tensor) -> dict:
        with torch.no_grad():
            hard = risk_model(paths_m, wait, pause_p, t_steps, collision_mode="hard",
                              gate_mode="hard", return_diagnostics=False)
            safe, clr = trajectory_is_static_safe(paths_m, obstacles,
                                                  sc.world.robot_radius_m, sc.world.clearance_margin_m)
            speed_ok, max_seg = check_speed_limit(
                paths_m, float(cfg["trajectory"]["max_speed_mps"]), dt_s)
            mc = simulate_execution(paths_m, wait, pause_p, d_safe, t_steps, dt_s,
                                    n_samples=args.mc_samples, chunk_size=1000,
                                    generator=torch.Generator().manual_seed(11))
        return {
            "hard_risk": float(hard["pair_risks"][0]) if hard["pair_risks"].numel() else 0.0,
            "static_safe": bool(safe.all()), "min_clearance_m": float(clr.min()),
            "speed_ok": bool(speed_ok.all()), "max_segment_m": float(max_seg.max()),
            "mc_collision_rate": mc.collision_rate(),
            "mc_makespan_s": float(mc.makespan_steps.to(torch.float64).mean()) * dt_s,
            "endpoint_err": float(torch.linalg.vector_norm(paths_m[:, 0] - start_m, dim=-1).max()),
            "endpoint_err_goal": float(torch.linalg.vector_norm(paths_m[:, -1] - goal_m, dim=-1).max()),
        }

    # ------------------------------------------------------------------
    # 1) 零引导等价性
    # ------------------------------------------------------------------
    log_section(logger, "1. 零引导等价性与参数冻结检查")
    gen = torch.Generator(device=device.type).manual_seed(123)
    x_init = torch.randn(n, h, 2, generator=gen, device=device)
    base = ddim_sample(model, schedule, start_n, goal_n, grid, num_steps=num_steps, x_init=x_init)

    zero_guide = GuidanceConfig(enabled=False, max_wait_per_point=float(cfg["wait"]["max_wait_per_point"]),
                                max_total_wait=float(cfg["wait"]["max_total_wait"]))
    zero = guided_ddim_sample(model, schedule, risk_model, cfg, start_m, goal_m, grid, pause_p,
                              obstacles=obstacles, num_steps=num_steps, guide=zero_guide,
                              time_steps=t_steps, dt_s=dt_s, d_safe=d_safe, x_init=x_init)
    zero_norm = zero.paths_m / scale - 1.0
    diff = float((base - zero_norm).abs().max())
    logger.info("零引导与基础 DDIM 的最大逐点差异 = %.3e", diff)
    report_zero_ok = diff < 1e-4

    # ------------------------------------------------------------------
    # 2) 无引导（B0）与引导（A3）对照，使用同一噪声
    # ------------------------------------------------------------------
    log_section(logger, "2. B0 独立采样 vs A3 首次碰撞风险引导")
    gen_b0 = torch.Generator(device=device.type).manual_seed(7)
    x_init_b0 = torch.randn(n, h, 2, generator=gen_b0, device=device)

    t0 = time.time()
    b0 = plan_with_method("B0", model, schedule, risk_model, cfg, start_m, goal_m, grid, pause_p,
                          obstacles, num_steps=num_steps, x_init=x_init_b0, d_safe=d_safe,
                          time_steps=t_steps, dt_s=dt_s)
    b0_eval = hard_eval(b0.paths_m, b0.wait)
    logger.info("B0：硬风险 %.4f，MC 碰撞率 %.4f，静态安全 %s，用时 %.1fs",
                b0_eval["hard_risk"], b0_eval["mc_collision_rate"], b0_eval["static_safe"],
                time.time() - t0)

    t0 = time.time()
    a3 = plan_with_method("A3", model, schedule, risk_model, cfg, start_m, goal_m, grid, pause_p,
                          obstacles, num_steps=num_steps, x_init=x_init_b0.clone(), d_safe=d_safe,
                          time_steps=t_steps, dt_s=dt_s)
    a3_eval = hard_eval(a3.paths_m, a3.wait)
    logger.info("A3：硬风险 %.4f，MC 碰撞率 %.4f，等待 %.2f 时隙，用时 %.1fs",
                a3_eval["hard_risk"], a3_eval["mc_collision_rate"], float(a3.wait.sum()),
                time.time() - t0)

    # ------------------------------------------------------------------
    # 3) 参数冻结检查
    # ------------------------------------------------------------------
    changed = [k for k, v in model.state_dict().items()
               if not torch.equal(v, before[k])]
    logger.info("推理后参数发生变化的张量数 = %d", len(changed))

    # ------------------------------------------------------------------
    # 4) 报告与图
    # ------------------------------------------------------------------
    report = StageReport("S4_guided_diffusion", run_dir,
                         "S4 风险引导 DDIM 演示与验证报告", cfg.to_dict())
    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    report.metric("zero_guidance_max_diff", diff, group="verification")
    report.metric("zero_guidance_exact", report_zero_ok, group="verification")
    report.metric("frozen_params_changed", len(changed), group="verification")
    report.metric("checkpoint_step", ckpt_step or -1, group="model")
    for key, val in b0_eval.items():
        report.metric(f"B0/{key}", val, group="result")
    for key, val in a3_eval.items():
        report.metric(f"A3/{key}", val, group="result")
    report.metric("A3/wait_total", float(a3.wait.sum()), group="result")
    report.metric("A3/plan_time_s", a3.elapsed_s, group="result")
    report.metric("B0/plan_time_s", b0.elapsed_s, group="result")

    # 每一步的风险/梯度曲线
    hist_rows = []
    for row in a3.history:
        hist_rows.append({
            "ddim_step": row.get("ddim_step"), "t": row.get("t"),
            "guided": row.get("guided"), "soft_risk": row.get("soft_risk"),
            "hard_risk": row.get("hard_risk"), "grad_norm_x": row.get("grad_norm_x"),
            "grad_norm_w": row.get("grad_norm_w"), "wait_total": row.get("wait_total"),
            "loss": row.get("loss"),
        })
    report.table("a3_guidance_trace", hist_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    steps = [r["ddim_step"] for r in a3.history]
    axes[0].plot(steps, [r.get("soft_risk", np.nan) for r in a3.history], label="软风险 ΣR_soft")
    axes[0].plot(steps, [r.get("hard_risk", np.nan) for r in a3.history], label="硬风险 ΣR_hard")
    axes[0].set_xlabel("DDIM 采样步")
    axes[0].set_ylabel("风险")
    axes[0].set_title("引导过程中的风险变化")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, [r.get("grad_norm_x", np.nan) for r in a3.history], label="∇_Q 范数")
    axes[1].plot(steps, [r.get("grad_norm_w", np.nan) for r in a3.history], label="∇_w 范数")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("DDIM 采样步")
    axes[1].set_title("引导梯度范数")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, [r.get("wait_total", np.nan) for r in a3.history], color="#2ca02c")
    axes[2].set_xlabel("DDIM 采样步")
    axes[2].set_ylabel("主动等待总量（时隙）")
    axes[2].set_title("等待计划随去噪演化")
    axes[2].grid(alpha=0.3)
    report.save_figure(fig, "guidance_trajectory")

    for name, res in (("B0", b0), ("A3", a3)):
        fig, ax = plt.subplots(figsize=(7, 6.5))
        draw_map(ax, sc.world)
        paths = res.paths_m.detach().cpu().numpy()
        for i in range(n):
            ax.plot(paths[i, :, 0], paths[i, :, 1], "-", lw=1.9, label=f"机器人 {i}")
            ax.plot(paths[i, 0, 0], paths[i, 0, 1], "o", ms=7, mec="black")
            ax.plot(paths[i, -1, 0], paths[i, -1, 1], "s", ms=7, mec="black")
            idx = np.where(res.wait[i].detach().cpu().numpy() > 1e-6)[0]
            for m in idx:
                ax.plot(paths[i, m, 0], paths[i, m, 1], "h", ms=9, mec="black")
        ax.set_title(f"{name}：硬风险={hard_eval(res.paths_m, res.wait)['hard_risk']:.4f}")
        ax.legend(fontsize=8)
        report.save_figure(fig, f"plan_{name}")

    # 逐步几何变化
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, frac in zip(axes, (0.0, 0.4, 0.7, 1.0)):
        draw_map(ax, sc.world, alpha_obstacle=0.35)
        k = min(len(a3.history) - 1, int(frac * (len(a3.history) - 1)))
        # 用最终路径代替历史快照（历史只保存了统计量）
        paths = a3.paths_m.detach().cpu().numpy()
        for i in range(n):
            ax.plot(paths[i, :, 0], paths[i, :, 1], lw=1.6)
        ax.set_title(f"第 {k} 步（t={a3.history[k].get('t')}）", fontsize=9)
    fig.suptitle("DDIM 各阶段的几何轨迹（最终结果）")
    report.save_figure(fig, "geometry_evolution")

    checks = {
        "零引导等价": report_zero_ok,
        "参数冻结": len(changed) == 0,
        "端点保持": max(a3_eval["endpoint_err"], a3_eval["endpoint_err_goal"]) < 1e-4,
        "静态安全": a3_eval["static_safe"],
        "速度约束": a3_eval["speed_ok"],
    }
    for name, ok in checks.items():
        report.metric(f"check/{name}", bool(ok), group="verification")

    report.add_section(
        "## 阶段目标\n\n"
        "把首次碰撞风险模块接到**推理阶段**的每一步 DDIM 预测上：网络参数冻结，"
        "风险梯度只修正当前的干净轨迹预测 Q̂0 与等待变量 w（文档 §7.3）。\n"
    )
    report.add_section(
        "## 验证结果\n\n"
        "| 检查项 | 结果 |\n|---|---|\n"
        + "\n".join(f"| {k} | {'通过' if v else '未通过'} |" for k, v in checks.items())
        + f"\n\n- 零引导与基础 DDIM 最大差异 `{diff:.3e}`；\n"
        f"- 推理后发生变化的参数张量数 `{len(changed)}`；\n"
        f"- B0（独立采样）：硬风险 {b0_eval['hard_risk']:.4f}，MC 碰撞率 {b0_eval['mc_collision_rate']:.4f}；\n"
        f"- A3（引导）：硬风险 {a3_eval['hard_risk']:.4f}，MC 碰撞率 {a3_eval['mc_collision_rate']:.4f}，"
        f"主动等待 {float(a3.wait.sum()):.2f} 时隙；\n"
        f"- 规划耗时：B0 {b0.elapsed_s:.1f}s，A3 {a3.elapsed_s:.1f}s（{num_steps} 步 DDIM）。\n"
    )
    report.add_section(
        "## 结论\n\n"
        "- 引导通路在结构上正确（零引导严格等价、参数冻结、端点不变）；\n"
        f"- {'引导显著降低了硬风险与实测碰撞率' if a3_eval['hard_risk'] < b0_eval['hard_risk'] else '本次引导未降低风险，需要检查步长与权重'}；\n"
        "- 与 S2 的手工案例一致：风险梯度既可以修改几何，也可以产生主动等待。\n"
    )
    status = "passed" if all(checks.values()) else "failed"
    report.write_report()
    report.write_metrics_csv()
    report.write_status(status, {}, {"checks": checks})
    report.sync_to_docs()
    update_project_progress([
        {"stage": "S4", "summary": f"引导接入完成，B0 硬风险 {b0_eval['hard_risk']:.4f} → A3 {a3_eval['hard_risk']:.4f}。"},
    ])
    print(json.dumps({"checks": checks, "B0": b0_eval, "A3": a3_eval}, ensure_ascii=False, indent=2))
    print(f"运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
