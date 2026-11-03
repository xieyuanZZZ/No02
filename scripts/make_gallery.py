"""生成可视化图集：场景仿真快照、时空图、动画、方法对比。

输出到 docs/gallery/，并生成 docs/VISUALIZATIONS.md 索引。

用法：
  python scripts/make_gallery.py --config configs/guidance.yaml
  python scripts/make_gallery.py --no-gif      # 跳过动画（更快）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle as MplCircle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.evaluation.visualizations import AGENT_COLORS, draw_map  # noqa: E402
from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.maps.generate import MAP_KINDS, generate_map  # noqa: E402
from fhd.maps.scenarios import SCENARIO_BUILDERS, build_scenario  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "docs" / "gallery"


# --------------------------------------------------------------------------- #
def load_model(cfg, device):
    from fhd.diffusion.schedule import DDPMSchedule
    from fhd.diffusion.train import build_model

    cands = sorted((ROOT / "results" / "S3_diffusion_training").glob("*/checkpoints/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None, None
    model = build_model(cfg, device)
    ckpt = torch.load(cands[0], map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    except RuntimeError:
        return None, None
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                            kind=str(cfg["diffusion"]["beta_schedule"]), device=device)
    return model, schedule


def run_simulation(sc, cfg, n_samples=300, seed=0, record=32, wait=None):
    d_safe = 2.0 * float(cfg["world"]["robot_radius_m"]) + float(cfg["world"]["safety_margin_m"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    if wait is None:
        wait = getattr(sc, "wait", None)
    if wait is None:
        wait = torch.zeros(sc.n_agents, sc.horizon - 1, dtype=torch.float64)
    else:
        wait = torch.as_tensor(wait, dtype=torch.float64).reshape(sc.n_agents, sc.horizon - 1)
    pause = torch.full((sc.n_agents,), float(cfg["execution"]["pause_probability"]),
                       dtype=torch.float64)
    batch = simulate_execution(
        sc.paths, wait, pause, d_safe, t_steps, dt_s, n_samples=n_samples, chunk_size=n_samples,
        generator=torch.Generator().manual_seed(seed), record_traces=True,
        record_max_samples=record,
    )
    return batch, d_safe, dt_s, t_steps


# --------------------------------------------------------------------------- #
def fig_overview(cases: list[tuple[str, object]], cfg, path: Path) -> Path:
    """4 类地图：左列场景（路径+机器人圆盘），右列一条随机执行的多帧快照叠加。"""
    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.6 * n))
    d_safe = 2.0 * float(cfg["world"]["robot_radius_m"]) + float(cfg["world"]["safety_margin_m"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    for row, (name, sc) in enumerate(cases):
        batch, _, _, _ = run_simulation(sc, cfg, n_samples=200, seed=hash(name) % 1000, record=8)
        ax = axes[row, 0]
        draw_map(ax, sc.world)
        for i in range(sc.n_agents):
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            p = sc.paths[i].detach().numpy()
            ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.8, label=f"机器人 {i}")
            ax.plot(p[0, 0], p[0, 1], "o", color=c, ms=8, mec="black")
            ax.plot(p[-1, 0], p[-1, 1], "s", color=c, ms=8, mec="black")
        ax.set_title(f"{name}：场景与几何路径（N={sc.n_agents}）", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

        ax = axes[row, 1]
        draw_map(ax, sc.world, alpha_obstacle=0.4)
        pos = batch.traces["positions"].numpy()
        first = batch.traces["first_collision_step"].numpy()
        for s in range(min(6, pos.shape[0])):
            for i in range(sc.n_agents):
                ax.plot(pos[s, i, :, 0], pos[s, i, :, 1], "-",
                        color=AGENT_COLORS[i % len(AGENT_COLORS)], lw=0.8, alpha=0.45)
        n_coll = 0
        for s in range(min(6, pos.shape[0])):
            step = int(first[s])
            if step < pos.shape[2] - 1:
                n_coll += 1
                for i in range(sc.n_agents):
                    ax.plot(pos[s, i, step, 0], pos[s, i, step, 1], "x", color="red", ms=11, mew=2.4)
                ax.annotate(f"碰撞 t={step * dt_s:.1f}s", pos[s, 0, step],
                            textcoords="offset points", xytext=(7, 5), color="red", fontsize=8)
        rate = batch.collision_rate()
        ax.set_title(f"{name}：随机执行（6 条轨迹，碰撞率 {rate:.2f}，{n_coll} 条碰撞）", fontsize=10)
    fig.suptitle("场景仿真总览：4 类地图 × 2 台机器人（红线=碰撞位置，d_safe=%.2f m）" % d_safe,
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_spacetime(sc, batch, dt_s: float, d_safe: float, path: Path) -> Path:
    """时空图：进度 z(t) 与两机器人中心距离 d(t)，直接展示“时序错开决定碰撞”。"""
    import torch as _t

    pos = batch.traces["positions"].numpy()
    prog = batch.traces.get("progress")
    first = batch.traces["first_collision_step"].numpy()
    k, n, t1, _ = pos.shape
    steps = np.arange(t1) * dt_s
    coll_mask = first < t1 - 1

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # 左：进度 z(t)
    ax = axes[0]
    show = min(3, k)
    for s in range(show):
        for i in range(n):
            if prog is not None:
                z = prog[s, i].numpy()
            else:
                z = np.zeros(t1)
            ax.step(steps, z, where="post", color=AGENT_COLORS[i % len(AGENT_COLORS)],
                    lw=1.7, alpha=0.85, ls="-" if i == 0 else "--")
    ax.plot([], [], color=AGENT_COLORS[0], ls="-", label="机器人 0 的进度 z(t)")
    ax.plot([], [], color=AGENT_COLORS[1 % len(AGENT_COLORS)], ls="--",
            label="机器人 1 的进度 z(t)")
    ax.set_xlabel("执行时间 [s]")
    ax.set_ylabel("进度索引 m（平台 = 被暂停在该路点）")
    ax.set_title(f"进度随时间：{show} 条随机执行（阶梯平台即随机暂停）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # 右：中心距离 d(t)
    ax = axes[1]
    ii, jj = 0, 1
    for s in range(min(24, k)):
        if coll_mask[s]:
            continue
        d = np.linalg.norm(pos[s, ii] - pos[s, jj], axis=-1)
        ax.plot(steps, d, color="#1f77b4", lw=0.8, alpha=0.35)
    for s in range(min(24, k)):
        if not coll_mask[s]:
            continue
        d = np.linalg.norm(pos[s, ii] - pos[s, jj], axis=-1)
        ax.plot(steps, d, color="#d62728", lw=1.6, alpha=0.9)
    ax.axhline(d_safe, color="black", ls="--", lw=1.2, label=f"d_safe = {d_safe:.2f} m")
    ax.plot([], [], color="#d62728", label="发生碰撞的样本")
    ax.plot([], [], color="#1f77b4", label="未碰撞的样本")
    ax.set_xlabel("执行时间 [s]")
    ax.set_ylabel("两机器人中心距离 [m]")
    ax.set_ylim(0, max(1.5, float(np.nanmax(np.linalg.norm(pos[:24, ii] - pos[:24, jj], axis=-1)))))
    ax.set_title(f"中心距离：样本碰撞率 {batch.collision_rate():.3f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("同一个安全几何轨迹，随机暂停让两人同时挤到交叉口 —— 这才是风险来源", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_snapshots(sc, batch, dt_s: float, d_safe: float, path: Path, n_frames: int = 6) -> Path:
    """一条碰撞执行的快照序列，帧集中在碰撞前后。"""
    first = batch.traces["first_collision_step"].numpy()
    pos = batch.traces["positions"].numpy()
    t_steps = pos.shape[2] - 1
    has_coll = first < t_steps
    s = int(np.argmax(has_coll)) if has_coll.any() else 0
    coll = int(first[s]) if has_coll.any() else t_steps
    lo = max(0, coll - 7)
    hi = min(t_steps, coll + 6)
    frames = np.unique(np.linspace(lo, hi, n_frames).astype(int))
    cols = 3
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 5.0 * rows))
    axes = np.atleast_2d(axes)
    for idx, step in enumerate(frames):
        ax = axes[idx // cols, idx % cols]
        draw_map(ax, sc.world, alpha_obstacle=0.5)
        for i in range(sc.n_agents):
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(pos[s, i, :step + 1, 0], pos[s, i, :step + 1, 1], "-", color=c, lw=1.4,
                    alpha=0.85)
            ax.add_patch(MplCircle((pos[s, i, step, 0], pos[s, i, step, 1]),
                                   sc.world.robot_radius_m, facecolor=c, edgecolor="black",
                                   alpha=0.95, zorder=6))
            ax.add_patch(MplCircle((pos[s, i, step, 0], pos[s, i, step, 1]), d_safe,
                                   facecolor="none", edgecolor=c, ls="--", lw=1.0, alpha=0.85,
                                   zorder=5))
        d = float(np.linalg.norm(pos[s, 0, step] - pos[s, 1, step]))
        hit = step >= coll and coll < t_steps
        ax.set_title(f"t = {step * dt_s:4.1f} s   中心距 {d:.2f} m"
                     + ("   ← 碰撞" if hit else ""),
                     fontsize=9.5, color="red" if hit else "black")
        ax.tick_params(labelsize=7)
    for k in range(len(frames), rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle(f"一条发生碰撞的执行：快照集中在碰撞前后（虚线圆半径 = d_safe = {d_safe:.3f} m（另一台机器人中心不得进入））",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_animation(sc, batch, dt_s: float, path: Path, fps: int = 12) -> Path:
    """随机执行动画（GIF）：机器人圆盘运动、暂停、碰撞高亮。"""
    first = batch.traces["first_collision_step"].numpy()
    pos = batch.traces["positions"].numpy()
    t_steps = pos.shape[2] - 1
    has_coll = first < t_steps
    s = int(np.argmax(has_coll)) if has_coll.any() else 0
    d_safe = 0.35
    n = sc.n_agents

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    draw_map(ax, sc.world, alpha_obstacle=0.5)
    trails = []
    for i in range(n):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(sc.paths[i, :, 0].detach().numpy(), sc.paths[i, :, 1].detach().numpy(), "--",
                color=c, lw=0.9, alpha=0.45)
        (trail,) = ax.plot([], [], "-", color=c, lw=1.6, alpha=0.9)
        trails.append(trail)
    robots = [MplCircle((0, 0), sc.world.robot_radius_m, facecolor=AGENT_COLORS[i % len(AGENT_COLORS)],
                        edgecolor="black", zorder=6) for i in range(n)]
    for r in robots:
        ax.add_patch(r)
    # 安全半径圈（半径 d_safe/2）：两圈相交即等价于中心距离 d < d_safe
    safety = [MplCircle((0, 0), d_safe, facecolor="none",
                        edgecolor=AGENT_COLORS[i % len(AGENT_COLORS)], ls="--", lw=1.1,
                        alpha=0.8, zorder=5) for i in range(n)]
    for c in safety:
        ax.add_patch(c)
    timer = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=11, va="top")
    status = ax.text(0.02, 0.885, "", transform=ax.transAxes, fontsize=11, va="top", color="green")
    ax.set_title(f"{sc.name}：随机执行动画（p={float(0.1):.2f}，虚线圆半径 = d_safe）")
    ax.legend([plt.Line2D([], [], color=AGENT_COLORS[i % len(AGENT_COLORS)]) for i in range(n)],
              [f"机器人 {i}" for i in range(n)], fontsize=8, loc="lower right")

    coll = int(first[s])

    def update(frame):
        for i in range(n):
            p = pos[s, i, frame]
            robots[i].center = (p[0], p[1])
            safety[i].center = (p[0], p[1])
            trails[i].set_data(pos[s, i, :frame + 1, 0], pos[s, i, :frame + 1, 1])
        timer.set_text(f"t = {frame * dt_s:5.1f} s   （{frame}/{t_steps} 时隙）")
        hit = frame >= coll and coll < t_steps
        d = float(np.linalg.norm(pos[s, 0, frame] - pos[s, 1, frame])) if n >= 2 else 0.0
        status.set_text(f"{'[碰撞] 已发生碰撞' if hit else '[正常] 未碰撞'}   d = {d:.2f} m")
        status.set_color("#d62728" if hit else "#2ca02c")
        for c in safety:
            c.center = c.center
        return [*robots, *safety, *trails, timer, status]

    anim = FuncAnimation(fig, update, frames=t_steps + 1, interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def fig_method_compare(cfg, device, path: Path) -> Path | None:
    """同一噪声下 B0（无机器人间引导）与 A3（首次碰撞引导）的对比。"""
    from fhd.evaluation.baselines import plan_with_method
    from fhd.risk.multi_pair import FirstHitRisk

    model, schedule = load_model(cfg, device)
    if model is None:
        return None
    risk = FirstHitRisk.from_config(cfg)
    sc = build_scenario("crossing", cfg)
    n = sc.n_agents
    h = int(cfg["trajectory"]["horizon"])
    start = sc.starts.float()
    goal = sc.goals.float()
    grid = sc.world.occupancy(torch.float32).unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1)
    pause = torch.full((n,), float(cfg["execution"]["pause_probability"]), dtype=torch.float32)
    x0 = torch.randn(n, h, 2, generator=torch.Generator().manual_seed(20261103))
    results = {}
    for m in ("B0", "A3"):
        results[m] = plan_with_method(
            m, model, schedule, risk, cfg, start, goal, grid, pause, sc.world.obstacles(),
            num_steps=40, x_init=x0.clone(), d_safe=risk.d_safe,
            time_steps=int(cfg["execution"]["horizon_steps"]),
            dt_s=float(cfg["trajectory"]["dt_s"]))
    d_safe = risk.d_safe
    dt_s = float(cfg["trajectory"]["dt_s"])
    t_steps = int(cfg["execution"]["horizon_steps"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for row, m in enumerate(("B0", "A3")):
        res = results[m]
        ax = axes[row, 0]
        draw_map(ax, sc.world)
        p = res.paths_m.detach().numpy()
        w = res.wait.detach().numpy()
        for i in range(n):
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(p[i, :, 0], p[i, :, 1], "-", color=c, lw=1.9, label=f"机器人 {i}")
            ax.add_patch(MplCircle((p[i, 0, 0], p[i, 0, 1]), sc.world.robot_radius_m,
                                   facecolor=c, edgecolor="black", zorder=6))
            idx = np.where(w[i] > 1e-6)[0]
            if len(idx):
                # 用一条跨越等待区间的粗线 + 一个总时长标注，避免标签堆叠
                ax.plot(p[i, idx[0]:idx[-1] + 1, 0], p[i, idx[0]:idx[-1] + 1, 1],
                        "-", color="black", lw=5.0, alpha=0.35, zorder=5)
                mid = idx[len(idx) // 2]
                ax.annotate(f"等待 {w[i].sum() * dt_s:.2f} s\n（{len(idx)} 个路点）",
                            p[i, mid], textcoords="offset points", xytext=(8, -26),
                            fontsize=8, color="black",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9),
                            zorder=8)
        ax.set_title(f"{m}：硬风险 {res.final['hard_risk_sum']:.4f}，"
                     f"等待 {res.final['wait_total']:.2f} 时隙", fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        batch, _, _, _ = run_simulation(
            type("S", (), {"paths": res.paths_m, "n_agents": n, "horizon": h,
                           "world": sc.world, "wait": res.wait})(), cfg,
            n_samples=400, seed=7, record=10)
        pos = batch.traces["positions"].numpy()
        first = batch.traces["first_collision_step"].numpy()
        steps = np.arange(pos.shape[2]) * dt_s
        coll_mask = first < pos.shape[2] - 1
        for s in range(min(10, pos.shape[0])):
            for i in range(n):
                c = "#d62728" if coll_mask[s] else AGENT_COLORS[i % len(AGENT_COLORS)]
                lw = 1.3 if coll_mask[s] else 0.9
                alpha = 0.85 if coll_mask[s] else 0.35
                ax.plot(steps, pos[s, i, :, 0], color=c, lw=lw, alpha=alpha)
        ax.plot([], [], color="#d62728", lw=1.3, label="发生碰撞的样本")
        ax.plot([], [], color=AGENT_COLORS[0], lw=0.9, label="未碰撞的样本")
        ax.set_xlabel("时间 [s]")
        ax.set_ylabel("x 坐标 [m]")
        ax.set_title(f"x(t)：实测碰撞率 {batch.collision_rate():.3f}"
                     f"（{int(coll_mask.sum())}/{pos.shape[0]} 条）", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"同一噪声种子下的方法对比（d_safe={d_safe:.2f} m，黑色粗线=主动等待区间）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def build_index(path: Path) -> Path:
    """生成 docs/VISUALIZATIONS.md：按阶段索引全部图片。"""
    lines = ["# 可视化图集", "",
             "本文件索引仓库中所有由实验自动生成的图。图片源文件在 `docs/stages/Sx/plots/`，",
             "仿真专题图在 `docs/gallery/`。", ""]
    gallery = ROOT / "docs" / "gallery"
    if gallery.exists():
        lines += ["## 场景仿真（专题）", ""]
        captions = {
            "overview_scenarios.png": "4 类地图的场景与随机执行总览",
            "spacetime_crossing.png": "时空图：几何相同、时序不同",
            "snapshots_crossing.png": "一条执行的等比时刻快照",
            "compare_B0_A3.png": "同一噪声下 B0 与 A3 的路径/等待/时空对比",
            "animation_crossing.gif": "随机执行动画（机器人圆盘、暂停、碰撞高亮）",
        }
        for png in sorted(gallery.glob("*")):
            if png.suffix.lower() not in (".png", ".gif"):
                continue
            cap = captions.get(png.name, png.name)
            lines.append(f"### {cap}")
            lines.append("")
            lines.append(f"![{cap}](gallery/{png.name})")
            lines.append("")
    names = {"S0": "环境与几何", "S1": "风险 DP", "S2": "梯度优化", "S3": "Diffusion",
             "S4": "引导接入", "S5": "对照实验"}
    for key in ("S0", "S1", "S2", "S3", "S4", "S5"):
        plots = sorted((ROOT / "docs" / "stages" / key / "plots").glob("*.png"))
        if not plots:
            continue
        lines += [f"## {key}：{names[key]}", ""]
        for png in plots:
            lines.append(f"![{png.stem}](stages/{key}/plots/{png.name})")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="生成可视化图集")
    ap.add_argument("--config", default=str(ROOT / "configs" / "guidance.yaml"))
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    OUT.mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.gallery")
    device = torch.device("cpu")

    cases = []
    for kind in MAP_KINDS:
        name = {"open": "空旷交叉", "dual_corridor": "双通路绕障",
                "narrow_gate": "窄通道/门口", "bottleneck": "多交叉/瓶颈"}[kind]
        if kind == "open":
            sc = build_scenario("crossing", cfg)
        elif kind == "dual_corridor":
            sc = build_scenario("dual_corridor", cfg, same_side=True)
        elif kind == "narrow_gate":
            sc = build_scenario("narrow_gate", cfg)
        else:
            sc = build_scenario("multi_bottleneck", cfg, n_agents=4)
        cases.append((name, sc))

    logger.info("生成场景总览 ...")
    fig_overview(cases, cfg, OUT / "overview_scenarios.png")

    crossing = build_scenario("crossing", cfg)
    batch, d_safe, dt_s, t_steps = run_simulation(crossing, cfg, n_samples=400, seed=11, record=24)
    logger.info("交叉场景：碰撞率 %.3f", batch.collision_rate())
    fig_spacetime(crossing, batch, dt_s, d_safe, OUT / "spacetime_crossing.png")
    fig_snapshots(crossing, batch, dt_s, d_safe, OUT / "snapshots_crossing.png")

    if not args.no_gif:
        logger.info("生成动画 ...")
        fig_animation(crossing, batch, dt_s, OUT / "animation_crossing.gif", fps=12)

    if not args.no_compare:
        logger.info("生成 B0/A3 对比 ...")
        try:
            fig_method_compare(cfg, device, OUT / "compare_B0_A3.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("方法对比图跳过：%s", exc)

    index = build_index(ROOT / "docs" / "VISUALIZATIONS.md")
    logger.info("图集完成：%s", index)
    print(f"图集目录：{OUT}")
    print(f"索引：{index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
