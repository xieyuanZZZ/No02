"""可视化：地图、场景、随机执行轨迹、DP/MC 对照、软硬敏感性、生存曲线、联合进度网格。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle as MplCircle  # noqa: E402
from matplotlib.patches import Rectangle as MplRect  # noqa: E402

AGENT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def _setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_setup_fonts()


def draw_map(ax: plt.Axes, world, alpha_obstacle: float = 0.55) -> None:
    """在给定坐标轴上画障碍物与地图边界（物理单位：米）。"""
    for rect in np.asarray(world.rects).reshape(-1, 4):
        ax.add_patch(
            MplRect((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
                    facecolor="#555555", edgecolor="black", alpha=alpha_obstacle, zorder=2)
        )
    for circ in np.asarray(world.circles).reshape(-1, 3):
        ax.add_patch(
            MplCircle((circ[0], circ[1]), circ[2], facecolor="#555555", edgecolor="black",
                      alpha=alpha_obstacle, zorder=2)
        )
    ax.set_xlim(0, world.width_m)
    ax.set_ylim(0, world.height_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2, linestyle=":")


def draw_robots(ax: plt.Axes, positions: np.ndarray, radius: float, colors=None, alpha: float = 0.9) -> None:
    """在给定位置画机器人圆盘。positions [N,2]。"""
    colors = colors or AGENT_COLORS
    for i, pos in enumerate(np.asarray(positions)):
        ax.add_patch(
            MplCircle((pos[0], pos[1]), radius, facecolor=colors[i % len(colors)],
                      edgecolor="black", alpha=alpha, zorder=5)
        )


def plot_scenario(
    sc,
    title: str | None = None,
    show_wait_points: bool = True,
    annotate_agents: bool = True,
) -> plt.Figure:
    """场景总览：地图 + 各机器人路径 + 起终点 + 允许等待点。"""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    draw_map(ax, sc.world)
    paths = sc.paths.detach().cpu().numpy()
    for i in range(sc.n_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(paths[i, :, 0], paths[i, :, 1], "-", color=color, lw=1.8,
                label=f"机器人 {i} 路径", zorder=4)
        ax.plot(paths[i, 0, 0], paths[i, 0, 1], "o", color=color, ms=8, mec="black", zorder=6)
        ax.plot(paths[i, -1, 0], paths[i, -1, 1], "s", color=color, ms=8, mec="black", zorder=6)
        if show_wait_points:
            mask = sc.wait_mask[i].detach().cpu().numpy()
            idx = np.where(mask)[0]
            if len(idx):
                ax.plot(paths[i, idx, 0], paths[i, idx, 1], ".", color=color, ms=5,
                        alpha=0.8, zorder=5)
        if annotate_agents:
            ax.annotate(f"起点 {i}", paths[i, 0], textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=color)
            ax.annotate(f"终点 {i}", paths[i, -1], textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color=color)
    draw_robots(ax, paths[:, 0, :], sc.world.robot_radius_m)
    ax.set_title(title or f"场景：{sc.name}（{sc.n_agents} 台机器人，H={sc.horizon}）")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_execution_traces(
    sc,
    traces: dict,
    n_show: int = 12,
    dt_s: float = 0.4,
    d_safe: float = 0.35,
    title: str | None = None,
) -> plt.Figure:
    """随机执行轨迹：每个样本的时空轨迹 + 首次碰撞位置与时间标注。"""
    positions = traces["positions"]  # [K,N,T+1,2]
    first_coll = traces["first_collision_step"]
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    else:
        positions = np.asarray(positions)
    if isinstance(first_coll, torch.Tensor):
        first_coll = first_coll.detach().cpu().numpy()
    else:
        first_coll = np.asarray(first_coll)

    k = min(n_show, positions.shape[0])
    fig, ax = plt.subplots(figsize=(7, 6.5))
    draw_map(ax, sc.world)
    paths = sc.paths.detach().cpu().numpy()
    for i in range(sc.n_agents):
        ax.plot(paths[i, :, 0], paths[i, :, 1], "--", color=AGENT_COLORS[i % len(AGENT_COLORS)],
                lw=1.0, alpha=0.45, zorder=3)

    n_coll = 0
    for s in range(k):
        for i in range(sc.n_agents):
            ax.plot(positions[s, i, :, 0], positions[s, i, :, 1], "-",
                    color=AGENT_COLORS[i % len(AGENT_COLORS)], lw=0.7, alpha=0.35, zorder=4)
    for s in range(k):
        step = int(first_coll[s])
        if step < positions.shape[2] - 1:
            n_coll += 1
            for i in range(sc.n_agents):
                p = positions[s, i, step]
                ax.plot(p[0], p[1], "x", color="red", ms=9, mew=2.0, zorder=8)
            ax.annotate(f"t={step * dt_s:.1f}s", positions[s, 0, step],
                        textcoords="offset points", xytext=(6, 4), fontsize=7, color="red")
    ax.set_title(title or f"随机执行轨迹（显示 {k} 条，其中 {n_coll} 条发生碰撞，d_safe={d_safe} m）")
    return fig


def plot_dp_vs_mc(records: list[dict], title: str = "硬 DP 风险 vs Monte Carlo 实测碰撞率") -> plt.Figure:
    """误差棒散点：横轴 DP 风险，纵轴 MC 频率，误差为 Wilson 95% 区间。"""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    dp = np.array([r["dp_risk"] for r in records])
    mc = np.array([r["mc_rate"] for r in records])
    n = np.array([r["mc_n"] for r in records])
    z = 1.96
    denom = 1 + z * z / n
    center = (mc + z * z / (2 * n)) / denom
    half = z * np.sqrt(mc * (1 - mc) / n + z * z / (4 * n * n)) / denom
    lo = np.clip(center - half, 0, 1)
    hi = np.clip(center + half, 0, 1)
    ax.errorbar(dp, mc, yerr=[mc - lo, hi - mc], fmt="o", capsize=3, color="#1f77b4",
                ecolor="#999999", label="20 个随机小案例")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="y = x")
    ax.set_xlabel("硬 DP 首次碰撞风险")
    ax.set_ylabel("Monte Carlo 碰撞频率（95% Wilson 区间）")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return fig


def plot_soft_hard_sensitivity(rows: list[dict], title: str = "软代理与硬风险的差异") -> plt.Figure:
    """rows: [{'epsilon':..,'beta':..,'soft':..,'hard':..}, ...]"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    eps_vals = sorted({r["epsilon"] for r in rows})
    beta_vals = sorted({r["beta"] for r in rows})

    ax = axes[0]
    for beta in beta_vals:
        sel = [r for r in rows if r["beta"] == beta]
        sel.sort(key=lambda r: r["epsilon"])
        ax.plot([r["epsilon"] for r in sel], [r["soft"] for r in sel], "o-",
                label=f"软风险 β={beta}")
    hard = rows[0]["hard"]
    ax.axhline(hard, color="black", ls="--", label=f"硬风险（常数 {hard:.3f}）")
    ax.set_xscale("log")
    ax.set_xlabel("软碰撞温度 ε [m]")
    ax.set_ylabel("风险值")
    ax.set_title("κ 敏感性：软代理随 ε 变化，硬风险不变")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for beta in beta_vals:
        sel = [r for r in rows if r["beta"] == beta]
        sel.sort(key=lambda r: r["epsilon"])
        ax.plot([r["epsilon"] for r in sel], [r["soft"] - r["hard"] for r in sel], "s-",
                label=f"β={beta}")
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("软碰撞温度 ε [m]")
    ax.set_ylabel("软风险 − 硬风险")
    ax.set_title("软代理高估量（虚假累积）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_survival_curves(survival: np.ndarray, dt_s: float, title: str = "首次碰撞 DP 的存活概率") -> plt.Figure:
    """survival [P,T+1]。"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = np.arange(survival.shape[-1]) * dt_s
    for p in range(survival.shape[0]):
        ax.plot(t, survival[p], lw=1.6, label=f"机器人对 {p}")
    ax.set_xlabel("执行时间 [s]")
    ax.set_ylabel("存活概率 ΣF_t")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    return fig


def plot_joint_risk_grid(
    hard: np.ndarray,
    soft: np.ndarray,
    mass: np.ndarray,
    title: str = "联合进度状态的碰撞区域与概率质量",
) -> plt.Figure:
    """三张热图：硬碰撞区域、软碰撞强度、某个时刻的联合概率质量。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, data, name, cmap in zip(
        axes,
        [hard, soft, mass],
        ["硬碰撞 C（0/1，动作 11）", "软碰撞代理（动作 11）", "某时刻联合概率质量 F_t"],
        ["Greys", "inferno", "viridis"],
    ):
        im = ax.imshow(np.asarray(data), origin="lower", cmap=cmap, aspect="equal")
        ax.set_xlabel("机器人 j 的进度 n")
        ax.set_ylabel("机器人 i 的进度 m")
        ax.set_title(name, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_scalar_curve(x, y, xlabel: str, ylabel: str, title: str, label: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, y, "o-", lw=1.6, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    if label:
        ax.legend(fontsize=8)
    return fig


def plot_timing_bar(names: list[str], values_ms: list[float], title: str = "计算开销") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(names)), values_ms, color="#4c72b0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("耗时 [ms]")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    return fig


def plot_path_optimization(
    world,
    paths_init: torch.Tensor,
    paths_opt: torch.Tensor,
    wait_opt: torch.Tensor | None = None,
    dt_s: float = 0.4,
    d_safe: float = 0.35,
    title: str = "风险梯度优化前后的几何轨迹",
) -> plt.Figure:
    """优化前后路径对比；在有主动等待的路点处标注等待时长。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    paths_init = paths_init.detach().cpu().numpy()
    paths_opt = paths_opt.detach().cpu().numpy()
    wait_np = wait_opt.detach().cpu().numpy() if wait_opt is not None else None

    ax = axes[0]
    draw_map(ax, world)
    for i in range(paths_init.shape[0]):
        ax.plot(paths_init[i, :, 0], paths_init[i, :, 1], "--",
                color=AGENT_COLORS[i % len(AGENT_COLORS)], lw=1.4, alpha=0.75,
                label=f"机器人 {i} 初始")
    ax.set_title("优化前")
    ax.legend(fontsize=8)

    ax = axes[1]
    draw_map(ax, world)
    for i in range(paths_opt.shape[0]):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(paths_opt[i, :, 0], paths_opt[i, :, 1], "-", color=color, lw=1.8,
                label=f"机器人 {i} 优化后")
        ax.plot(paths_opt[i, 0, 0], paths_opt[i, 0, 1], "o", color=color, ms=7, mec="black")
        ax.plot(paths_opt[i, -1, 0], paths_opt[i, -1, 1], "s", color=color, ms=7, mec="black")
        if wait_np is not None:
            idx = np.where(wait_np[i] > 1e-6)[0]
            for m in idx:
                p = paths_opt[i, m]
                ax.plot(p[0], p[1], "h", color=color, ms=4 + 3 * float(wait_np[i, m]),
                        mec="black", zorder=7)
                ax.annotate(f"{wait_np[i, m] * dt_s:.1f}s", p, textcoords="offset points",
                            xytext=(5, 5), fontsize=7, color=color)
    ax.set_title("优化后（六边形=主动等待点，数字=等待时长）")
    ax.legend(fontsize=8)
    fig.suptitle(f"{title}（d_safe={d_safe:.2f} m）", fontsize=12)
    return fig


def plot_risk_convergence(
    histories: dict[str, list[dict]],
    dt_s: float,
    title: str = "风险与完成时间随优化步数的变化",
) -> plt.Figure:
    """多方法的收敛曲线：软风险、硬风险、等待总量、完成时间代价。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    keys = [("soft_risk_sum", "软风险 ΣR_soft"), ("hard_risk_sum", "硬风险 ΣR_hard"),
            ("wait_total", "主动等待总量（时隙）"), ("finish_cost_s", "完成时间代价 J_finish [s]")]
    for ax, (key, label) in zip(axes.ravel(), keys):
        for name, hist in histories.items():
            xs = [row["step"] for row in hist if key in row]
            ys = [row[key] for row in hist if key in row]
            ax.plot(xs, ys, lw=1.6, label=name)
        ax.set_xlabel("优化步数")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.set_title(label, fontsize=10)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_pareto(points: list[dict], title: str = "硬风险—完成时间 Pareto") -> plt.Figure:
    """points: [{'method','hard_risk','mean_makespan_s','wait_total'}]"""
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = sorted({p["method"] for p in points})
    for idx, name in enumerate(methods):
        sel = [p for p in points if p["method"] == name]
        sel.sort(key=lambda r: r["mean_makespan_s"])
        ax.plot([p["mean_makespan_s"] for p in sel], [p["hard_risk"] for p in sel], "o-",
                color=AGENT_COLORS[idx % len(AGENT_COLORS)], label=name, ms=6)
    ax.set_xlabel("平均完成时间（Monte Carlo） [s]")
    ax.set_ylabel("硬 DP 首次碰撞风险之和")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_progress_distribution(
    pi_before: np.ndarray,
    pi_after: np.ndarray,
    dt_s: float,
    title: str = "进度分布变化",
) -> plt.Figure:
    """两张热图：优化前后单机进度分布 π_i^t(m)。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, data, name in zip(axes, [pi_before, pi_after], ["优化前", "优化后"]):
        extent = [0, data.shape[0] * dt_s, 0, data.shape[1] - 1]
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="viridis", extent=extent)
        ax.set_xlabel("执行时间 [s]")
        ax.set_ylabel("进度 m")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=12)
    return fig


def save_fig(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
