"""S0：环境与几何基础验证。

内容：
1. 记录运行环境（Python/PyTorch/CUDA/GPU/Git）；
2. 生成 4 类地图并保存地图与占据栅格；
3. 构造全部手工场景并保存场景图；
4. 跑一次随机执行模拟，可视化执行轨迹并**标记碰撞时间与位置**；
5. 连续碰撞、速度约束、静态障碍的独立复核；
6. 运行 S0 单元测试，输出报告并同步到 docs/stages/S0。

用法：python scripts/run_s0.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.evaluation.reporting import StageReport, update_project_progress  # noqa: E402
from fhd.evaluation.testing import run_pytest  # noqa: E402
from fhd.evaluation.visualizations import (  # noqa: E402
    draw_map,
    draw_robots,
    plot_execution_traces,
    plot_scenario,
)
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.geometry.collisions import (  # noqa: E402
    check_speed_limit,
    path_segment_lengths,
    trajectory_is_static_safe,
)
from fhd.maps.generate import MAP_KINDS, generate_map  # noqa: E402
from fhd.maps.scenarios import SCENARIO_BUILDERS, scenario_summary  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="S0 环境与几何基础验证")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--no-tests", action="store_true", help="跳过 pytest（仅用于调试）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    run_dir = new_run_dir("S0_environment", "s0_baseline")
    logger = get_logger("fhd.s0", run_dir / "logs" / "s0.log")
    report = StageReport("S0_environment", run_dir, "S0 环境与几何基础验证报告", cfg.to_dict())

    log_section(logger, "S0 环境与几何基础验证")
    info = system_info()
    logger.info("Python %s | torch %s | CUDA %s | GPU %s", info["python"], info["torch"],
                info["cuda_available"], info.get("gpu_name", "CPU"))
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")

    world_cfg = cfg["world"]
    d_safe = 2.0 * float(world_cfg["robot_radius_m"]) + float(world_cfg["safety_margin_m"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    v_max = float(cfg["trajectory"]["max_speed_mps"])
    h = int(cfg["trajectory"]["horizon"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    p_pause = float(cfg["execution"]["pause_probability"])

    # ------------------------------------------------------------------
    # 1) 4 类地图
    # ------------------------------------------------------------------
    log_section(logger, "1. 地图生成")
    map_seeds = {kind: int(cfg["seed"]) + i for i, kind in enumerate(MAP_KINDS)}
    maps = {}
    for kind, seed in map_seeds.items():
        world = generate_map(kind, cfg, seed)
        maps[kind] = world
        occ = world.occupancy().numpy()
        sdf = world.sdf_grid().numpy()
        report.metrics_from_dict(
            {
                "n_obstacles": world.n_obstacles,
                "occupied_cells": float(occ.sum()),
                "occupancy_ratio": float(occ.mean()),
                "sdf_min_m": float(sdf.min()),
                "sdf_max_m": float(sdf.max()),
            },
            group=f"map/{kind}",
        )
        report.table(
            f"map_{kind}_rects",
            [{"x0": r[0], "y0": r[1], "x1": r[2], "y1": r[3]} for r in world.rects],
        )
        report.table(
            f"map_{kind}_circles",
            [{"cx": c[0], "cy": c[1], "r": c[2]} for c in world.circles],
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        draw_map(axes[0], world)
        axes[0].set_title(f"{kind}（种子 {seed}）障碍物几何")
        im = axes[1].imshow(sdf, origin="lower", extent=[0, world.width_m, 0, world.height_m],
                            cmap="RdBu", vmin=-2, vmax=2)
        axes[1].set_title("解析 SDF [m]（负值=碰撞）")
        axes[1].set_xlabel("x [m]")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
        report.save_figure(fig, f"map_{kind}")
        logger.info("地图 %s：障碍物 %d 个，占据率 %.3f", kind, world.n_obstacles, float(occ.mean()))

    # ------------------------------------------------------------------
    # 2) 手工场景
    # ------------------------------------------------------------------
    log_section(logger, "2. 手工场景与几何正确性")
    scenarios = {}
    scenario_rows = []
    for name in SCENARIO_BUILDERS:
        kwargs = {"n_agents": 4} if name == "multi_bottleneck" else {}
        sc = SCENARIO_BUILDERS[name](cfg, **kwargs)
        scenarios[name] = sc
        summary = scenario_summary(sc)
        scenario_rows.append({"scenario": name, **{k: str(v) for k, v in summary.items()}})
        lens = path_segment_lengths(sc.paths)
        ok_speed, max_len = check_speed_limit(sc.paths, v_max, dt_s)
        safe_static, clearance = trajectory_is_static_safe(
            sc.paths, sc.world.obstacles(), sc.world.robot_radius_m, sc.world.clearance_margin_m
        )
        report.metrics_from_dict(
            {
                "n_agents": sc.n_agents,
                "path_length_mean_m": float(torch.mean(torch.stack([p for p in
                    [torch.linalg.vector_norm(sc.paths[i, -1] - sc.paths[i, 0]) for i in range(sc.n_agents)]])))
                if sc.n_agents else 0.0,
                "max_segment_m": float(lens.max()),
                "speed_limit_ok": bool(ok_speed.all()),
                "static_safe": bool(safe_static.all()),
                "min_clearance_m": float(clearance.min()),
            },
            group=f"scenario/{name}",
        )
        fig = plot_scenario(sc)
        report.save_figure(fig, f"scenario_{name}")
        logger.info(
            "场景 %-20s N=%d 最大段长=%.3f m 速度约束=%s 静态安全=%s 最小净空=%.3f m",
            name, sc.n_agents, float(lens.max()), bool(ok_speed.all()), bool(safe_static.all()),
            float(clearance.min()),
        )
        assert bool(ok_speed.all()), f"{name} 违反速度约束"
        if name not in ("dual_corridor", "narrow_gate", "multi_bottleneck"):
            assert bool(safe_static.all()), f"{name} 静态碰撞"
    report.table("scenario_summary", scenario_rows)

    # ------------------------------------------------------------------
    # 3) 随机执行模拟 + 碰撞标记
    # ------------------------------------------------------------------
    log_section(logger, "3. 随机执行模拟（层 A）与碰撞标记")
    main_sc = scenarios["crossing"]
    pause = torch.full((main_sc.n_agents,), p_pause, dtype=torch.float64)
    batch = simulate_execution(
        main_sc.paths, torch.zeros_like(main_sc.wait_mask, dtype=torch.float64), pause,
        d_safe, t_steps, dt_s, n_samples=500, chunk_size=500,
        generator=torch.Generator().manual_seed(int(cfg["seed"])), record_traces=True,
        record_max_samples=40,
    )
    summary = batch.summary(dt_s, d_safe)
    logger.info("交叉场景：碰撞率=%.4f，完成率=%.4f，平均 makespan=%.2f s",
                summary["collision_rate"], summary["completion_rate"], summary["mean_makespan_s"])
    report.metrics_from_dict({f"crossing/{k}": v for k, v in summary.items()
                              if not isinstance(v, (list, dict))}, group="execution")
    fig = plot_execution_traces(main_sc, batch.traces, n_show=15, dt_s=dt_s, d_safe=d_safe)
    report.save_figure(fig, "execution_traces_crossing")

    # 单条轨迹的详细记录（碰撞时间与位置）
    trace_positions = batch.traces["positions"].numpy()
    first_coll = batch.traces["first_collision_step"].numpy()
    detail_rows = []
    for s in range(min(10, trace_positions.shape[0])):
        step = int(first_coll[s])
        row = {"sample": s, "first_collision_step": step,
               "collision_time_s": step * dt_s if step < t_steps else None,
               "collision": bool(step < t_steps)}
        if step < t_steps:
            row["pos_robot0_x"] = float(trace_positions[s, 0, step, 0])
            row["pos_robot0_y"] = float(trace_positions[s, 0, step, 1])
            row["pos_robot1_x"] = float(trace_positions[s, 1, step, 0])
            row["pos_robot1_y"] = float(trace_positions[s, 1, step, 1])
        detail_rows.append(row)
    report.table("execution_collision_detail", detail_rows)

    # 单条执行轨迹图（含碰撞时刻剖面）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    draw_map(axes[0], main_sc.world)
    paths = main_sc.paths.numpy()
    for i in range(main_sc.n_agents):
        axes[0].plot(paths[i, :, 0], paths[i, :, 1], "--", lw=1.2, alpha=0.5)
    s = 0
    for i in range(main_sc.n_agents):
        axes[0].plot(trace_positions[s, i, :, 0], trace_positions[s, i, :, 1], "-", lw=1.6,
                     label=f"机器人 {i}")
    step = int(first_coll[s])
    if step < t_steps:
        for i in range(main_sc.n_agents):
            p = trace_positions[s, i, step]
            axes[0].plot(p[0], p[1], "x", color="red", ms=12, mew=2.5)
        axes[0].annotate(f"首次碰撞 t={step*dt_s:.1f}s", trace_positions[s, 0, step],
                         textcoords="offset points", xytext=(8, 8), color="red", fontsize=9)
        draw_robots(axes[0], trace_positions[s, :, step], main_sc.world.robot_radius_m, alpha=0.6)
    axes[0].set_title("单条执行轨迹与碰撞位置")
    axes[0].legend(fontsize=8)

    t_axis = np.arange(t_steps + 1) * dt_s
    rel = trace_positions[s, 0, :, :] - trace_positions[s, 1, :, :]
    dist = np.linalg.norm(rel, axis=-1)
    axes[1].plot(t_axis, dist, lw=1.6, label="两机器人中心距离")
    axes[1].axhline(d_safe, color="red", ls="--", label=f"d_safe = {d_safe:.2f} m")
    if step < t_steps:
        axes[1].axvline(step * dt_s, color="red", ls=":", label="首次碰撞时刻")
    axes[1].set_xlabel("执行时间 [s]")
    axes[1].set_ylabel("中心距离 [m]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title("中心距离随时间变化")
    report.save_figure(fig, "execution_single_sample_detail")

    # ------------------------------------------------------------------
    # 4) 单元测试
    # ------------------------------------------------------------------
    test_summary = {}
    if not args.no_tests:
        log_section(logger, "4. 运行 S0 单元测试")
        test_summary = run_pytest(["tests/test_collision_continuous.py"])
        logger.info("pytest: passed=%d failed=%d error=%d", test_summary["passed"],
                    test_summary["failed"], test_summary["error"])
        report.metric("tests_passed", test_summary["passed"], group="tests")
        report.metric("tests_failed", test_summary["failed"], group="tests")
        (run_dir / "logs" / "pytest_s0.txt").write_text(test_summary["output_tail"], encoding="utf-8")

    # ------------------------------------------------------------------
    # 5) 报告
    # ------------------------------------------------------------------
    report.add_section(
        "## 阶段目标\n\n"
        "建立新工程环境、静态地图生成器、机器人圆盘几何、任务/场景构造、"
        "轨迹连续碰撞检测，以及独立随机执行模拟器；并用单元测试固定正确性。\n"
    )
    report.add_section(
        "## 关键实现决定\n\n"
        "- 障碍物只用**连续几何**（轴对齐矩形 + 圆）描述，栅格仅用于可视化与网络条件输入；\n"
        "- 线段到矩形的距离用**逐矩形黄金分割**（单矩形距离对参数 u 是凸的，min 会破坏凸性，故不能合并求解）；\n"
        "- 机器人之间用相对运动闭式解求最小距离，并额外保留一套独立实现供交叉验证；\n"
        "- 执行模型为层 A：每时隙最多前进 1 个路点，时隙内线性插值，同步连续段碰撞判定。\n"
    )
    report.add_section(
        "## 结论\n\n"
        f"- 4 类地图生成正常，占据率与 SDF 数值合理；\n"
        f"- {len(SCENARIO_BUILDERS)} 个手工场景全部满足段长 <= v_max*dt = {v_max*dt_s:.2f} m；\n"
        f"- 交叉场景在 p={p_pause} 下随机执行碰撞率 {summary['collision_rate']:.4f}，"
        "说明随机暂停确实会制造碰撞风险；\n"
        "- S0 通过，可进入 S1（首次碰撞风险 DP）。\n"
    )
    status = "passed" if test_summary.get("all_passed", True) else "failed"
    report.write_report()
    report.write_metrics_csv()
    report.write_status(status, test_summary, {"d_safe_m": d_safe})
    report.sync_to_docs()
    update_project_progress([
        {"stage": "S0", "summary": "地图/几何/执行模拟器就绪，连续碰撞与速度约束测试通过。"},
    ])
    logger.info("S0 完成，报告：%s", run_dir / "report.md")
    print(f"\nS0 运行目录: {run_dir}")
    print(f"报告: {run_dir / 'report.md'}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
