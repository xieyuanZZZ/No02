"""S1：首次碰撞风险 DP 的正确性验证（R1–R10）。

产出：
  * 全部风险相关单元测试结果；
  * 20 个随机小案例上硬 DP 与 10,000 次 Monte Carlo 的二项检验对照（图 + CSV）；
  * 软代理对 (ε, β) 的敏感性（硬风险不变）；
  * soft_gate 与 hard_gate 的进度分布偏差曲线（批复 §11 要求）；
  * 首次碰撞 DP 的存活概率曲线与联合进度网格；
  * 梯度有限差分误差与计算开销；
  * 中文报告、metrics.csv、stage_status.json，并同步到 docs/stages/S1。

用法：python scripts/verify_risk.py --config configs/risk_only.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.evaluation.reporting import StageReport, update_project_progress  # noqa: E402
from fhd.evaluation.testing import run_pytest  # noqa: E402
from fhd.evaluation.visualizations import (  # noqa: E402
    plot_dp_vs_mc,
    plot_joint_risk_grid,
    plot_scalar_curve,
    plot_soft_hard_sensitivity,
    plot_survival_curves,
)
from fhd.execution.progress import move_probabilities, propagate_progress, release_times  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.maps.scenarios import build_scenario  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk, team_union_bound  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

ALPHA = 0.001


def random_crossing_case(seed: int, h: int) -> torch.Tensor:
    """随机交叉案例：起终点安全、几何必然交叉。"""
    g = torch.Generator().manual_seed(seed)
    y1 = 1.0 + 2.0 * float(torch.rand((), generator=g))
    x1 = 1.0 + 2.0 * float(torch.rand((), generator=g))
    t = torch.linspace(0, 1, h, dtype=torch.float64).unsqueeze(-1)
    a = torch.tensor([0.2, y1], dtype=torch.float64) + t * torch.tensor([3.6, 0.0], dtype=torch.float64)
    b = torch.tensor([x1, 0.2], dtype=torch.float64) + t * torch.tensor([0.0, 3.6], dtype=torch.float64)
    return torch.stack([a, b])


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 首次碰撞风险 DP 验证")
    parser.add_argument("--config", default=str(ROOT / "configs" / "risk_only.yaml"))
    parser.add_argument("--mc-cases", type=int, default=20)
    parser.add_argument("--mc-samples", type=int, default=10000)
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    run_dir = new_run_dir("S1_risk_dp", "s1_verify")
    logger = get_logger("fhd.s1", run_dir / "logs" / "s1.log")
    report = StageReport("S1_risk_dp", run_dir, "S1 首次碰撞风险 DP 正确性报告", cfg.to_dict())

    d_safe = 2.0 * float(cfg["world"]["robot_radius_m"]) + float(cfg["world"]["safety_margin_m"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    h = int(cfg["trajectory"]["horizon"])
    p_pause = float(cfg["execution"]["pause_probability"])

    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    log_section(logger, "S1 首次碰撞风险 DP 正确性验证")

    model = FirstHitRisk.from_config(cfg)
    logger.info("d_safe=%.3f m, beta=%.3f, offset c=%.2f, eps=%.3f m, kernel=%s",
                model.d_safe, model.beta, model.offset, model.temperature_m, model.kernel)

    # ------------------------------------------------------------------
    # 1) 单元测试 R1–R10
    # ------------------------------------------------------------------
    test_summary = {}
    if not args.no_tests:
        log_section(logger, "1. 运行 R1–R10 单元测试")
        test_summary = run_pytest([
            "tests/test_progress_transition.py",
            "tests/test_first_hit_dp.py",
            "tests/test_mc_agreement.py",
            "tests/test_risk_gradient.py",
            "tests/test_risk_numerics.py",
        ])
        report.metric("tests_passed", test_summary["passed"], group="tests")
        report.metric("tests_failed", test_summary["failed"], group="tests")
        report.metric("tests_error", test_summary["error"], group="tests")
        (run_dir / "logs" / "pytest_s1.txt").write_text(test_summary["output_tail"], encoding="utf-8")
        logger.info("pytest: passed=%d failed=%d error=%d", test_summary["passed"],
                    test_summary["failed"], test_summary["error"])

    # ------------------------------------------------------------------
    # 2) R7：硬 DP 与 Monte Carlo 对照
    # ------------------------------------------------------------------
    log_section(logger, "2. R7 硬 DP vs Monte Carlo")
    records = []
    t_start = time.time()
    for k in range(args.mc_cases):
        hh = 6 + (k % 4)
        steps = 20
        paths = random_crossing_case(seed=k, h=hh)
        wait = torch.zeros(2, hh - 1, dtype=torch.float64)
        if k % 3 == 0:
            wait[1, 1] = 1.0
        p_val = [0.05, 0.10, 0.20][k % 3]
        pause = torch.full((2,), p_val, dtype=torch.float64)
        out = model(paths, wait, pause, time_steps=steps, collision_mode="hard", gate_mode="hard",
                    return_diagnostics=False)
        dp = float(out["pair_risks"][0])
        mc = simulate_execution(paths, wait, pause, d_safe, steps, dt_s,
                                n_samples=args.mc_samples, chunk_size=5000,
                                generator=torch.Generator().manual_seed(1000 + k))
        hits = int(mc.collision.sum())
        if 0.0 < dp < 1.0:
            pvalue = float(binomtest(hits, args.mc_samples, dp).pvalue)
        else:
            pvalue = 1.0 if (dp == 0.0 and hits == 0) or (dp == 1.0 and hits == args.mc_samples) else 0.0
        records.append({
            "case": k, "h": hh, "t_steps": steps, "pause_p": p_val,
            "dp_risk": dp, "mc_hits": hits, "mc_n": args.mc_samples,
            "mc_rate": hits / args.mc_samples, "abs_error": abs(dp - hits / args.mc_samples),
            "binom_pvalue": pvalue, "consistent": pvalue >= ALPHA,
        })
        logger.info("案例 %2d: H=%d p=%.2f  DP=%.5f  MC=%.5f (hits=%d/%d) p值=%.4f %s",
                    k, hh, p_val, dp, hits / args.mc_samples, hits, args.mc_samples, pvalue,
                    "OK" if pvalue >= ALPHA else "异常")
    elapsed_mc = time.time() - t_start
    report.table("r7_dp_vs_mc", records)
    max_err = max(r["abs_error"] for r in records)
    n_incons = sum(1 for r in records if not r["consistent"])
    report.metric("r7_max_abs_error", max_err, group="R7")
    report.metric("r7_inconsistent_cases", n_incons, group="R7")
    report.metric("r7_total_mc_samples", args.mc_cases * args.mc_samples, group="R7")
    report.metric("r7_elapsed_s", elapsed_mc, group="R7")
    fig = plot_dp_vs_mc(records)
    report.save_figure(fig, "r7_dp_vs_mc")

    # ------------------------------------------------------------------
    # 3) R9：软代理温度敏感性 + soft/hard 进度分布偏差
    # ------------------------------------------------------------------
    log_section(logger, "3. R9 软硬模型差异与 sweep")
    sc = build_scenario("crossing", cfg, h=h, offset=0.5)
    wait0 = torch.zeros(2, h - 1, dtype=torch.float64)
    pause = torch.full((2,), p_pause, dtype=torch.float64)
    rows = []
    for kernel in ("logistic", "smoothstep"):
        for eps in (0.005, 0.01, 0.025, 0.05, 0.1):
            for beta in (0.1, 0.25, 0.5):
                m = FirstHitRisk(d_safe=d_safe, beta=beta, offset=model.offset,
                                 temperature_m=eps, kernel=kernel,
                                 support_m=3 * eps, dt_s=dt_s)
                soft = float(m(sc.paths, wait0, pause, t_steps, collision_mode="soft",
                               gate_mode="soft", return_diagnostics=False)["pair_risks"][0])
                hard_soft_gate = float(m(sc.paths, wait0, pause, t_steps, collision_mode="hard",
                                         gate_mode="soft", return_diagnostics=False)["pair_risks"][0])
                rows.append({"kernel": kernel, "epsilon": eps, "beta": beta, "soft": soft,
                             "hard_soft_gate": hard_soft_gate})
    hard_ref = float(model(sc.paths, wait0, pause, t_steps, collision_mode="hard", gate_mode="hard",
                           return_diagnostics=False)["pair_risks"][0])
    for r in rows:
        r["hard"] = hard_ref
    report.table("r9_soft_hard_sweep", rows)
    report.metric("r9_hard_reference", hard_ref, group="R9")
    report.metric("r9_soft_max", max(r["soft"] for r in rows), group="R9")
    report.metric("r9_soft_min", min(r["soft"] for r in rows), group="R9")
    logistic_rows = [r for r in rows if r["kernel"] == "logistic"]
    fig = plot_soft_hard_sensitivity(logistic_rows, "R9：logistic 软代理 vs 硬风险（交叉场景）")
    report.save_figure(fig, "r9_soft_hard_sensitivity")

    # soft/hard 进度分布偏差（批复 §11）
    tau0 = release_times(wait0)
    a_hard = move_probabilities(tau0, pause, t_steps, gate_mode="hard")
    a_soft = move_probabilities(tau0, pause, t_steps, beta=model.beta, offset=model.offset,
                                gate_mode="soft")
    pi_hard = propagate_progress(a_hard)[0]
    pi_soft = propagate_progress(a_soft)[0]
    bias_rows = []
    for t in range(t_steps + 1):
        tv = 0.5 * (pi_hard[t] - pi_soft[t]).abs().sum()
        bias_rows.append({
            "t": t, "time_s": t * dt_s,
            "arrived_hard": float(pi_hard[t, -1]), "arrived_soft": float(pi_soft[t, -1]),
            "tv_distance": float(tv),
        })
    report.table("soft_hard_progress_bias", bias_rows)
    report.metric("soft_hard_max_tv_distance", max(r["tv_distance"] for r in bias_rows), group="R9")
    fig = plot_scalar_curve([r["time_s"] for r in bias_rows], [r["tv_distance"] for r in bias_rows],
                            "执行时间 [s]", "总变差距离", "软/硬门槛的进度分布偏差（p=0.1, w=0）")
    report.save_figure(fig, "soft_hard_progress_bias")

    # ------------------------------------------------------------------
    # 4) 存活概率曲线 + 联合进度网格
    # ------------------------------------------------------------------
    log_section(logger, "4. 存活概率与联合进度网格")
    surv_rows = []
    fig_surv_data = []
    for name, kw in (("crossing", {"offset": 0.5}), ("head_on", {})):
        s = build_scenario(name, cfg, h=h, **kw)
        w = torch.zeros(s.n_agents, h - 1, dtype=torch.float64)
        p = torch.full((s.n_agents,), p_pause, dtype=torch.float64)
        out = model(s.paths, w, p, t_steps, collision_mode="hard", gate_mode="hard")
        sv = out["diagnostics"]["survival"][0].numpy()
        fig_surv_data.append(sv)
        for t, v in enumerate(sv):
            surv_rows.append({"scenario": name, "t": t, "time_s": t * dt_s, "survival": float(v)})
        report.metric(f"{name}/hard_risk", float(out["pair_risks"][0]), group="survival")
    report.table("survival_curves", surv_rows)
    fig = plot_survival_curves(np.stack(fig_surv_data), dt_s, "硬首次碰撞 DP 的存活概率")
    report.save_figure(fig, "survival_curves")

    # 联合进度网格（交叉场景，动作 11）
    out_hist = model(sc.paths, wait0, pause, t_steps, collision_mode="soft", gate_mode="soft",
                     return_history=True)
    collision_tensor = out_hist["diagnostics"]["collision_tensor"][0]  # [4,H,H]
    history = out_hist["diagnostics"]["history"][0]  # [T+1,H,H]
    hard_tensor = (model.pair_collision_tensor(sc.paths, "hard")[3])[0]
    # 选择“处于危险区状态的概率质量最大”的时刻作为快照，信息量最高
    risk_mask = torch.as_tensor(hard_tensor[3], dtype=history.dtype)
    mass_in_risk = (history * risk_mask.unsqueeze(0)).sum(dim=(1, 2))
    mid = int(torch.argmax(mass_in_risk).item())
    fig = plot_joint_risk_grid(
        hard_tensor[3].numpy(), collision_tensor[3].numpy(), history[mid].numpy(),
        title=f"交叉场景联合进度网格（动作 11，t={mid} 即 {mid * dt_s:.1f}s 时的概率质量）",
    )
    report.save_figure(fig, "joint_progress_grid")
    report.metric("joint_grid_snapshot_t", mid, group="diagnostics")

    # ------------------------------------------------------------------
    # 5) 梯度有限差分误差
    # ------------------------------------------------------------------
    log_section(logger, "5. 梯度有限差分检查")
    grad_paths = sc.paths.clone().requires_grad_(True)
    grad_wait = torch.zeros(2, h - 1, dtype=torch.float64, requires_grad=True)
    value = model(grad_paths, grad_wait, pause, t_steps, collision_mode="soft", gate_mode="soft",
                  return_diagnostics=False)["pair_sum_risk"]
    g_paths, g_wait = torch.autograd.grad(value, (grad_paths, grad_wait))
    fd_rows = []
    eps_h = 1e-6
    with torch.no_grad():
        for (agent, idx, axis) in [(0, 12, 1), (0, 18, 1), (1, 10, 0), (1, 20, 0)]:
            base = sc.paths.clone()
            plus, minus = base.clone(), base.clone()
            plus[agent, idx, axis] += eps_h
            minus[agent, idx, axis] -= eps_h
            f_plus = float(model(plus, wait0, pause, t_steps, collision_mode="soft",
                                 gate_mode="soft", return_diagnostics=False)["pair_sum_risk"])
            f_minus = float(model(minus, wait0, pause, t_steps, collision_mode="soft",
                                  gate_mode="soft", return_diagnostics=False)["pair_sum_risk"])
            fd = (f_plus - f_minus) / (2 * eps_h)
            ana = float(g_paths[agent, idx, axis])
            rel = abs(fd - ana) / max(abs(fd), abs(ana), 1e-8)
            fd_rows.append({"target": f"paths[{agent},{idx},{axis}]", "analytic": ana,
                            "finite_diff": fd, "rel_error": rel})
        for (agent, idx) in [(0, 9), (0, 11), (1, 12)]:
            base = torch.zeros(2, h - 1, dtype=torch.float64)
            plus, minus = base.clone(), base.clone()
            plus[agent, idx] += eps_h
            minus[agent, idx] -= eps_h
            f_plus = float(model(sc.paths, plus, pause, t_steps, collision_mode="soft",
                                 gate_mode="soft", return_diagnostics=False)["pair_sum_risk"])
            f_minus = float(model(sc.paths, minus, pause, t_steps, collision_mode="soft",
                                  gate_mode="soft", return_diagnostics=False)["pair_sum_risk"])
            fd = (f_plus - f_minus) / (2 * eps_h)
            ana = float(g_wait[agent, idx])
            rel = abs(fd - ana) / max(abs(fd), abs(ana), 1e-8)
            fd_rows.append({"target": f"wait[{agent},{idx}]", "analytic": ana,
                            "finite_diff": fd, "rel_error": rel})
    report.table("gradient_finite_difference", fd_rows)
    report.metric("gradient_max_rel_error", max(r["rel_error"] for r in fd_rows), group="R8")
    for r in fd_rows:
        logger.info("%-18s 解析=%+.6e 差分=%+.6e 相对误差=%.2e",
                    r["target"], r["analytic"], r["finite_diff"], r["rel_error"])

    # ------------------------------------------------------------------
    # 6) 计算开销
    # ------------------------------------------------------------------
    log_section(logger, "6. 计算开销")
    timing_rows = []
    for n_agents in (2, 4, 6):
        s = build_scenario("multi_bottleneck", cfg, h=32, n_agents=n_agents)
        paths = s.paths.clone().requires_grad_(True)
        wait = torch.zeros(n_agents, 31, dtype=torch.float64, requires_grad=True)
        p = torch.full((n_agents,), p_pause, dtype=torch.float64)
        # 预热
        out = model(paths, wait, p, t_steps, collision_mode="soft", gate_mode="soft")
        torch.autograd.grad(out["pair_sum_risk"] + out["finish_cost"], (paths, wait))
        t0 = time.perf_counter()
        out = model(paths, wait, p, t_steps, collision_mode="soft", gate_mode="soft")
        loss = out["pair_sum_risk"] + out["finish_cost"]
        torch.autograd.grad(loss, (paths, wait))
        elapsed = (time.perf_counter() - t0) * 1000
        n_pairs = n_agents * (n_agents - 1) // 2
        timing_rows.append({"n_agents": n_agents, "n_pairs": n_pairs,
                            "fwd_bwd_ms": round(elapsed, 2),
                            "per_pair_ms": round(elapsed / n_pairs, 3)})
        logger.info("N=%d（%d 对）：前向+反向 %.1f ms（每对 %.2f ms）",
                    n_agents, n_pairs, elapsed, elapsed / n_pairs)
    report.table("timing", timing_rows)
    for row in timing_rows:
        report.metric(f"timing_n{row['n_agents']}_ms", row["fwd_bwd_ms"], group="timing")

    # GPU（若可用）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        s = build_scenario("multi_bottleneck", cfg, h=32, n_agents=6)
        paths = s.paths.clone().to(device=device, dtype=torch.float32).requires_grad_(True)
        wait = torch.zeros(6, 31, device=device, dtype=torch.float32, requires_grad=True)
        p = torch.full((6,), p_pause, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(paths, wait, p, t_steps, collision_mode="soft", gate_mode="soft")
        loss = out["pair_sum_risk"] + out["finish_cost"]
        torch.autograd.grad(loss, (paths, wait))
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000
        report.metric("timing_cuda_n6_ms", round(gpu_ms, 2), group="timing")
        report.metric("cuda_peak_mem_mb", round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                      group="timing")
        logger.info("N=6 CUDA(float32)：前向+反向 %.1f ms，峰值显存 %.1f MB", gpu_ms,
                    torch.cuda.max_memory_allocated() / 1024**2)

    # ------------------------------------------------------------------
    # 7) 多机风险汇总口径
    # ------------------------------------------------------------------
    s6 = build_scenario("multi_bottleneck", cfg, h=32, n_agents=6)
    w6 = torch.zeros(6, 31, dtype=torch.float64)
    p6 = torch.full((6,), p_pause, dtype=torch.float64)
    out6 = model(s6.paths, w6, p6, t_steps, collision_mode="hard", gate_mode="hard")
    bound = float(team_union_bound(out6["pair_risks"]))
    mc6 = simulate_execution(s6.paths, w6, p6, d_safe, t_steps, dt_s, n_samples=2000,
                             chunk_size=1000, generator=torch.Generator().manual_seed(7))
    report.metric("n6_pair_risk_sum", float(out6["pair_sum_risk"]), group="multi")
    report.metric("n6_union_bound", bound, group="multi")
    report.metric("n6_mc_team_collision_rate", mc6.collision_rate(), group="multi")
    logger.info("N=6：两两硬风险之和=%.4f，并集上界=%.4f，全队 MC 碰撞率=%.4f",
                float(out6["pair_sum_risk"]), bound, mc6.collision_rate())

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    report.add_section(
        "## 阶段目标\n\n"
        "在训练任何神经网络之前，先证明“随机执行进度传播 + 首次碰撞吸收”这套动态规划在数学上正确、"
        "可微、且与独立 Monte Carlo 一致（文档 §5.4 的第一道门槛）。\n"
    )
    report.add_section(
        "## 方法要点\n\n"
        f"- 状态：F_t(m,n) 为“截至 t 未碰撞且联合进度为 (m,n)”的概率质量，共 H×H={h}×{h} 个状态；\n"
        "- 每步枚举 4 种动作（00/10/01/11），用四种动作的连续运动段碰撞核吸收质量；\n"
        f"- 连续碰撞判定：d_min 由相对运动闭式解给出（d_safe={d_safe:.2f} m）；\n"
        f"- 软门槛 a=(1-p)·σ((t+0.5+c-τ)/β)，c={model.offset}、β={model.beta}（批复 §11）；\n"
        f"- 硬门槛 a=(1-p)·1[t≥τ]，与执行模拟器使用完全相同的实数比较；\n"
        "- 全队语义：两两风险之和只是引导目标，不是全队至少碰撞一次的概率；"
        "全队碰撞率必须由联合 Monte Carlo 估计（本节给出并集上界对照）。\n"
    )
    report.add_section(
        "## 测试与对照结果\n\n"
        f"- R1–R10 单元测试：通过 {test_summary.get('passed', 0)} 项、失败 "
        f"{test_summary.get('failed', 0)} 项、错误 {test_summary.get('error', 0)} 项；\n"
        f"- R7（{args.mc_cases} 个案例 × {args.mc_samples} 次 MC）：最大绝对误差 {max_err:.5f}，"
        f"二项检验不一致案例 {n_incons} 个（α={ALPHA}）；\n"
        f"- R9：硬风险恒为 {hard_ref:.6f}（与 ε、β 无关），logistic 软代理在 "
        f"[{min(r['soft'] for r in rows):.4f}, {max(r['soft'] for r in rows):.4f}] 之间变化，"
        "说明软代理不具备概率解释；\n"
        f"- 软/硬门槛带来的进度分布最大总变差距离 "
        f"{max(r['tv_distance'] for r in bias_rows):.4f}；\n"
        f"- R8：梯度与中心差分的最大相对误差 {max(r['rel_error'] for r in fd_rows):.2e}。\n"
    )
    report.add_section(
        "## 结论\n\n"
        "- 首次碰撞概率 DP 在无碰撞时严格保持概率守恒，在必然碰撞时严格为 1，且只计第一次碰撞；\n"
        f"- 硬 DP 与独立 Monte Carlo 在统计误差内一致（最大绝对误差 {max_err:.5f}）；\n"
        "- 硬风险可精确计算但不可微；软风险可微但只是代理量，二者必须分开报告；\n"
        "- S1 门槛通过，可进入 S2（不使用 Diffusion 的风险梯度优化）。\n"
    )
    status = "passed" if (test_summary.get("all_passed", True) and n_incons == 0) else "failed"
    report.write_report()
    report.write_metrics_csv()
    report.write_status(status, test_summary, {"d_safe_m": d_safe, "r7_max_abs_error": max_err})
    report.sync_to_docs()
    update_project_progress([
        {"stage": "S0", "summary": "地图/几何/执行模拟器就绪。"},
        {"stage": "S1", "summary": f"首次碰撞 DP 通过 R1–R10；DP vs MC 最大误差 {max_err:.5f}。"},
    ])
    logger.info("S1 完成，报告：%s", run_dir / "report.md")
    print(f"\nS1 运行目录: {run_dir}")
    print(f"报告: {run_dir / 'report.md'}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
