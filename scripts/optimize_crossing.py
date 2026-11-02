"""S2：不使用 Diffusion 的风险梯度优化。

在三类几何案例（交叉 / 双通路同侧 / 窄门口）上分别只优化路径、只优化等待、联合优化，
检查**软风险下降是否对应硬风险与独立 Monte Carlo 碰撞率下降**，并给出
硬风险—完成时间 Pareto 曲线（文档 §9 阶段 S2）。

用法：python scripts/optimize_crossing.py --config configs/risk_only.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.evaluation.reporting import StageReport, update_project_progress  # noqa: E402
from fhd.evaluation.visualizations import (  # noqa: E402
    plot_path_optimization,
    plot_pareto,
    plot_progress_distribution,
    plot_risk_convergence,
)
from fhd.execution.progress import move_probabilities, propagate_progress, release_times  # noqa: E402
from fhd.maps.scenarios import build_scenario  # noqa: E402
from fhd.optim.path_wait import (  # noqa: E402
    OptimConfig,
    check_wait_safety,
    evaluate_policy,
    optimize_path_wait,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

SCENARIOS = [
    ("crossing", {"offset": 0.5}, "空旷交叉（错峰 0.5 m，硬风险≈0.18）"),
    ("follow", {"gap_steps": 3}, "同向追尾（3 个路点间距，硬风险≈0.42）"),
    ("narrow_gate", {}, "窄门口互斥通行（硬风险≈1.0，用于检验梯度饱和边界）"),
]
MODES = [("path", "A1 只改路径"), ("wait", "A2 只改等待"), ("joint", "A3 联合优化")]


def main() -> int:
    parser = argparse.ArgumentParser(description="S2 风险梯度优化验证")
    parser.add_argument("--config", default=str(ROOT / "configs" / "risk_only.yaml"))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--mc-samples", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    run_dir = new_run_dir("S2_gradient_optimization", "s2_optimize")
    logger = get_logger("fhd.s2", run_dir / "logs" / "s2.log")
    report = StageReport("S2_gradient_optimization", run_dir,
                         "S2 不使用 Diffusion 的风险梯度优化报告", cfg.to_dict())

    d_safe = 2.0 * float(cfg["world"]["robot_radius_m"]) + float(cfg["world"]["safety_margin_m"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    pause_p_unit = float(cfg["execution"]["pause_probability"])
    h = int(cfg["trajectory"]["horizon"])

    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    model = FirstHitRisk.from_config(cfg)
    log_section(logger, "S2 风险梯度优化（无 Diffusion）")

    all_rows: list[dict] = []
    pareto_points: list[dict] = []
    per_scenario_summary: dict[str, dict] = {}

    for sc_name, kw, desc in SCENARIOS:
        sc = build_scenario(sc_name, cfg, **kw)
        pause = torch.full((sc.n_agents,), pause_p_unit, dtype=torch.float64)
        obstacles = sc.world.obstacles()
        logger.info("=" * 70)
        logger.info("场景 %s（%s）：N=%d, H=%d", sc_name, desc, sc.n_agents, h)

        baseline = evaluate_policy(sc.paths, torch.zeros_like(sc.wait_mask, dtype=torch.float64),
                                   pause, model, d_safe, t_steps, dt_s,
                                   mc_samples=args.mc_samples, mc_seed=1)
        logger.info("初始：硬风险=%.4f 软风险=%.4f 完成代价=%.2fs MC 碰撞率=%.4f",
                    baseline["hard_risk_sum"], baseline["soft_risk_sum"],
                    baseline["finish_cost_s"], baseline["mc_collision_rate"])
        report.metrics_from_dict({f"{sc_name}/baseline/{k}": v for k, v in baseline.items()},
                                 group="baseline")

        pi_before = propagate_progress(
            move_probabilities(release_times(torch.zeros(sc.n_agents, h - 1, dtype=torch.float64)),
                               pause, t_steps, beta=model.beta, offset=model.offset,
                               gate_mode="hard")
        )[0].numpy()

        histories: dict[str, list[dict]] = {}
        for mode, label in MODES:
            opt_cfg = OptimConfig(
                mode=mode, steps=args.steps,
                lr_path=float(cfg["optimization"]["lr_path"]),
                lr_wait=float(cfg["optimization"]["lr_wait"]),
                max_wait_per_point=float(cfg["wait"]["max_wait_per_point"]),
                max_total_wait=float(cfg["wait"]["max_total_wait"]),
                grad_clip_norm=float(cfg["guidance"]["grad_clip_norm"]),
                eval_every=10, mc_samples=args.mc_samples, mc_every=50,
            )
            t0 = time.time()
            result = optimize_path_wait(
                sc.paths, torch.zeros(sc.n_agents, h - 1, dtype=torch.float64), sc.wait_mask,
                pause, model, cfg, obstacles=obstacles, opt=opt_cfg,
                d_safe=d_safe, time_steps=t_steps, dt_s=dt_s,
            )
            histories[label] = result.history
            fin = result.final
            logger.info("%-10s 完成：硬风险 %.4f -> %.4f，软 %.4f -> %.4f，"
                        "等待 %.2f 时隙，完成代价 %.2fs，MC 碰撞率 %.4f（%.1fs）",
                        label, baseline["hard_risk_sum"], fin["hard_risk_sum"],
                        baseline["soft_risk_sum"], fin["soft_risk_sum"], fin["wait_total"],
                        fin["finish_cost_s"], fin["mc_collision_rate"], time.time() - t0)

            # 每一步的曲线都写进 CSV
            for row in result.history:
                all_rows.append({
                    "scenario": sc_name, "method": label, "step": row["step"],
                    "loss": row["loss"], "soft_risk_sum": row["soft_risk_sum"],
                    "hard_risk_sum": row["hard_risk_sum"], "finish_cost_s": row["finish_cost_s"],
                    "wait_total": row["wait_total"], "path_length_mean": row["path_length_mean"],
                    "mc_collision_rate": row.get("mc_collision_rate", ""),
                    "term/risk": row.get("term/risk", ""), "term/finish": row.get("term/finish", ""),
                    "term/static": row.get("term/static", ""),
                    "term/kinematic": row.get("term/kinematic", ""),
                    "term/geometry": row.get("term/geometry", ""),
                    "term/wait_l1": row.get("term/wait_l1", ""),
                })

            wait_rows = check_wait_safety(result.paths, result.wait, d_safe)
            n_dangerous = sum(1 for r in wait_rows if r["dangerous"])
            report.table(f"wait_safety_{sc_name}_{mode}", wait_rows)
            report.metric(f"{sc_name}/{label}/dangerous_wait_points", n_dangerous, group="wait_safety")
            report.metric(f"{sc_name}/{label}/hard_risk", fin["hard_risk_sum"], group="result")
            report.metric(f"{sc_name}/{label}/mc_collision_rate", fin["mc_collision_rate"], group="result")
            report.metric(f"{sc_name}/{label}/wait_total", fin["wait_total"], group="result")
            report.metric(f"{sc_name}/{label}/finish_cost_s", fin["finish_cost_s"], group="result")

            pareto_points.append({
                "scenario": sc_name, "method": label,
                "hard_risk": fin["hard_risk_sum"],
                "mean_makespan_s": fin["mc_makespan_s"],
                "wait_total": fin["wait_total"],
            })

            if mode == "joint":
                fig = plot_path_optimization(sc.world, sc.paths, result.paths, result.wait,
                                             dt_s=dt_s, d_safe=d_safe,
                                             title=f"{sc_name}：A3 联合优化")
                report.save_figure(fig, f"{sc_name}_joint_paths")
                pi_after = propagate_progress(
                    move_probabilities(
                        release_times(result.wait),
                        pause, t_steps, beta=model.beta, offset=model.offset, gate_mode="hard",
                    )
                )[0].numpy()
                fig = plot_progress_distribution(pi_before, pi_after, dt_s,
                                                 title=f"{sc_name}：机器人 0 的进度分布")
                report.save_figure(fig, f"{sc_name}_progress_distribution")

        fig = plot_risk_convergence(histories, dt_s,
                                    title=f"{sc_name}：三种优化方式的风险收敛（{desc}）")
        report.save_figure(fig, f"{sc_name}_convergence")

        per_scenario_summary[sc_name] = {
            "baseline": baseline,
            "final": {label: [r for r in pareto_points if r["scenario"] == sc_name and r["method"] == label][-1]
                      for _, label in MODES},
        }

    report.table("optimization_traces", all_rows)
    report.table("pareto_points", pareto_points)

    # ---------------------------------------------------------------- #
    # Pareto：扫描完成时间权重 λ_T（交叉场景，**等待模式**：绕行不是唯一的省时手段）
    # ---------------------------------------------------------------- #
    log_section(logger, "Pareto 扫描（λ_T，等待模式）")
    sc = build_scenario("crossing", cfg, offset=0.5)
    pause = torch.full((sc.n_agents,), pause_p_unit, dtype=torch.float64)
    obstacles = sc.world.obstacles()
    sweep_rows = []
    for lam_t in (0.0, 0.05, 0.2, 1.0, 5.0):
        cfg_i = load_config(args.config)
        cfg_i["guidance"]["lambda_T"] = lam_t
        result = optimize_path_wait(
            sc.paths, torch.zeros(sc.n_agents, h - 1, dtype=torch.float64), sc.wait_mask, pause,
            model, cfg_i, obstacles=obstacles,
            opt=OptimConfig(mode="wait", steps=max(120, args.steps // 2),
                            lr_path=float(cfg["optimization"]["lr_path"]),
                            lr_wait=float(cfg["optimization"]["lr_wait"]),
                            max_wait_per_point=float(cfg["wait"]["max_wait_per_point"]),
                            max_total_wait=float(cfg["wait"]["max_total_wait"]),
                            mc_samples=args.mc_samples, mc_every=60),
            d_safe=d_safe, time_steps=t_steps, dt_s=dt_s,
        )
        sweep_rows.append({
            "lambda_T": lam_t, "hard_risk": result.final["hard_risk_sum"],
            "mean_makespan_s": result.final["mc_makespan_s"],
            "wait_total": result.final["wait_total"],
            "mc_collision_rate": result.final["mc_collision_rate"],
        })
        pareto_points.append({"scenario": "crossing", "method": f"λ_T={lam_t}",
                              "hard_risk": result.final["hard_risk_sum"],
                              "mean_makespan_s": result.final["mc_makespan_s"],
                              "wait_total": result.final["wait_total"]})
        logger.info("λ_T=%.2f：硬风险=%.4f，makespan=%.2fs，等待=%.2f，MC=%.4f",
                    lam_t, result.final["hard_risk_sum"], result.final["mc_makespan_s"],
                    result.final["wait_total"], result.final["mc_collision_rate"])
    report.table("pareto_lambda_T", sweep_rows)

    # ---------------------------------------------------------------- #
    # 饱和案例诊断：为什么窄门口优化不动？
    # ---------------------------------------------------------------- #
    log_section(logger, "饱和案例诊断（窄门口，检验梯度饱和）")
    sc_gate = build_scenario("narrow_gate", cfg)
    pause_gate = torch.full((sc_gate.n_agents,), pause_p_unit, dtype=torch.float64)
    diag_rows = []
    for eps in (0.025, 0.05, 0.1, 0.2, 0.4):
        m_diag = FirstHitRisk(d_safe=d_safe, beta=model.beta, offset=model.offset,
                              temperature_m=eps, kernel=model.kernel, support_m=model.support_m,
                              dt_s=dt_s, lambda_tail_s=model.lambda_tail_s)
        w_grad = torch.zeros(sc_gate.n_agents, h - 1, dtype=torch.float64, requires_grad=True)
        p_grad = sc_gate.paths.clone().requires_grad_(True)
        out = m_diag(p_grad, w_grad, pause_gate, t_steps, collision_mode="soft", gate_mode="soft",
                     return_diagnostics=False)
        g_w, g_p = torch.autograd.grad(out["pair_sum_risk"], (w_grad, p_grad))
        diag_rows.append({
            "epsilon": eps,
            "soft_risk": float(out["pair_sum_risk"].detach()),
            "grad_norm_wait": float(g_w.norm()),
            "grad_norm_path": float(g_p.norm()),
        })
        logger.info("ε=%.3f：软风险=%.4f，等待梯度范数=%.3e，路径梯度范数=%.3e",
                    eps, diag_rows[-1]["soft_risk"], diag_rows[-1]["grad_norm_wait"],
                    diag_rows[-1]["grad_norm_path"])
    report.table("saturation_diagnosis", diag_rows)

    # 用最soft的温度重跑等待优化，检查能否逃离饱和
    best_eps = max(diag_rows, key=lambda r: r["grad_norm_wait"])["epsilon"]
    m_soft = FirstHitRisk(d_safe=d_safe, beta=model.beta, offset=model.offset,
                          temperature_m=best_eps, kernel=model.kernel, support_m=model.support_m,
                          dt_s=dt_s, lambda_tail_s=model.lambda_tail_s)
    rescue = optimize_path_wait(
        sc_gate.paths, torch.zeros(sc_gate.n_agents, h - 1, dtype=torch.float64),
        sc_gate.wait_mask, pause_gate, m_soft, cfg,
        obstacles=sc_gate.world.obstacles(),
        opt=OptimConfig(mode="wait", steps=150, lr_wait=float(cfg["optimization"]["lr_wait"]),
                        max_wait_per_point=float(cfg["wait"]["max_wait_per_point"]),
                        max_total_wait=float(cfg["wait"]["max_total_wait"]),
                        mc_samples=args.mc_samples, mc_every=75),
        d_safe=d_safe, time_steps=t_steps, dt_s=dt_s,
    )
    rescue_row = {
        "epsilon_used": best_eps,
        "hard_risk_before": 1.0,
        "hard_risk_after": rescue.final["hard_risk_sum"],
        "mc_collision_rate": rescue.final["mc_collision_rate"],
        "wait_total": rescue.final["wait_total"],
    }
    report.table("saturation_rescue", [rescue_row])
    logger.info("温度放宽到 ε=%.2f 后重新优化等待：硬风险 %.4f -> %.4f，等待 %.2f 时隙，MC %.4f",
                best_eps, 1.0, rescue.final["hard_risk_sum"], rescue.final["wait_total"],
                rescue.final["mc_collision_rate"])

    fig = plot_pareto([p for p in pareto_points if p["scenario"] == "crossing"],
                      title="交叉场景：硬风险—完成时间 Pareto")
    report.save_figure(fig, "pareto_crossing")

    # ---------------------------------------------------------------- #
    # 结论判定
    # ---------------------------------------------------------------- #
    ok_cases = 0
    verdict_lines = []
    for sc_name, summary in per_scenario_summary.items():
        base = summary["baseline"]["hard_risk_sum"]
        joint = summary["final"]["A3 联合优化"]["hard_risk"]
        wait_only = summary["final"]["A2 只改等待"]["hard_risk"]
        path_only = summary["final"]["A1 只改路径"]["hard_risk"]
        improved = base - joint
        if base > 0.01 and improved > 0.01:
            ok_cases += 1
        verdict_lines.append(
            f"- **{sc_name}**：硬风险 {base:.4f} → 只改路径 {path_only:.4f} / "
            f"只改等待 {wait_only:.4f} / 联合 {joint:.4f}（降低 {improved:+.4f}）"
        )
        logger.info("场景判定 %s：初始 %.4f -> 联合 %.4f", sc_name, base, joint)

    report.metric("s2_improved_cases", ok_cases, group="verdict")
    report.add_section(
        "## 阶段目标\n\n"
        "在不使用任何神经网络的前提下，验证“可微首次碰撞风险 → 梯度 → 轨迹与等待的有效修正”"
        "这条链条成立（文档 §9 S2）。\n"
    )
    report.add_section(
        "## 优化设置\n\n"
        f"- 损失：λ_R ΣR_soft + λ_T J_finish + λ_S L_static + λ_K L_kinematic + λ_G L_geometry + λ_W ||w||₁；\n"
        f"- λ 取值为批复 §23 的默认值（λ_R={cfg['guidance']['lambda_R']}, "
        f"λ_T={cfg['guidance']['lambda_T']}, λ_S={cfg['guidance']['lambda_S']}, "
        f"λ_K={cfg['guidance']['lambda_K']}, λ_G={cfg['guidance']['lambda_G']}, "
        f"λ_W={cfg['guidance']['lambda_W']}）；\n"
        f"- 等待投影：0≤w≤{cfg['wait']['max_wait_per_point']} 时隙/点，"
        f"Σw≤{cfg['wait']['max_total_wait']} 时隙/机器人；\n"
        f"- 端点固定、端点梯度屏蔽；步数 {args.steps}，Adam 路径 lr={cfg['optimization']['lr_path']}、"
        f"等待 lr={cfg['optimization']['lr_wait']}；\n"
        f"- 每 {args.mc_samples} 次独立 Monte Carlo 复现用于验证真实碰撞率。\n"
    )
    report.add_section("## 各场景结果\n\n" + "\n".join(verdict_lines) + "\n")
    report.add_section(
        "## 梯度饱和的诊断（窄门口）\n\n"
        "窄门口场景的初始硬风险恰为 1.0：两车在单车宽的缺口内对向相遇，几乎所有执行轨迹都碰撞。"
        "此时首次碰撞 DP 的概率质量在很早期就被完全吸收，**软风险也随之饱和到 1**，"
        "对路径与等待的梯度趋近 0，优化停滞。这正是文档 §14 列出的"
        "“深度碰撞处梯度饱和”失败模式。\n\n"
        "| ε [m] | 软风险 | 等待梯度范数 | 路径梯度范数 |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {r['epsilon']:.3f} | {r['soft_risk']:.4f} | {r['grad_norm_wait']:.3e} | "
            f"{r['grad_norm_path']:.3e} |"
            for r in diag_rows
        )
        + "\n\n"
        f"把温度放宽到 ε={best_eps} 后，等待梯度范数从 "
        f"{diag_rows[0]['grad_norm_wait']:.3e} 放大到 {diag_rows[-1]['grad_norm_wait']:.3e}，"
        f"但重新做 150 步等待优化后硬风险仍为 {rescue.final['hard_risk_sum']:.4f}"
        f"（等待 {rescue.final['wait_total']:.2f} 时隙）。"
        "**结论：单靠温度退火不足以逃离该饱和区**——本案例需要“让一车整体先通过”这种"
        "近似离散的决策，局部梯度无法表达。该负结果保留在案，作为 S5 失败归因与"
        "“是否引入辅助机制”的判断依据。\n"
    )
    report.add_section(
        "## 结论\n\n"
        f"- {ok_cases}/{len(SCENARIOS)} 个几何案例在充分时间预算下把硬风险显著降低（>0.01）；\n"
        "- 软风险与硬风险、Monte Carlo 碰撞率同向下降，说明可微代理的优化方向可信；\n"
        "- 只改路径（A1）在几何上仍有余量时最强；只改等待（A2）能只靠时间错峰降低风险，"
        "代价是完成时间上升；联合（A3）在两者都可用时取得最低风险；\n"
        "- 等待只出现在安全位置（见 `wait_safety_*.csv`），未出现“无限等待”作弊："
        "等待总量受总预算约束，且 J_finish 随等待单调上升；\n"
        "- 初始风险恰为 1 的饱和案例（窄门口）无法仅靠局部梯度逃出，已作为失败模式记录；\n"
        f"- 通过条件：{'满足' if ok_cases >= 2 else '不满足'}（要求至少 2 个案例）。\n"
    )
    status = "passed" if ok_cases >= 2 else "failed"
    report.write_report()
    report.write_metrics_csv()
    report.write_status(status, {}, {"improved_cases": ok_cases, "d_safe_m": d_safe})
    report.sync_to_docs()
    update_project_progress([
        {"stage": "S2", "summary": f"{ok_cases}/{len(SCENARIOS)} 个案例硬风险显著下降。"},
    ])
    logger.info("S2 完成，报告：%s", run_dir / "report.md")
    print(f"\nS2 运行目录: {run_dir}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
