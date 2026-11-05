"""Online_FirstHit_v2 结果总览图（只读已有结果 + 少量可视化用仿真）。

生成：
  fig01_scene_plans.png   规划场景：路径、起终点、障碍、原始/W2+ 等待点
  fig02_mpc_tracking.png  MPC 跟踪：x-y 轨迹、s(t)+模式、关键 pair 距离
  fig03_m0m3.png          M0–M3 修正效果：预测-实测、分组 MAE、可靠性
  fig04_case_study.png    两个碰撞例 + 一个 M3 虚假碰撞例
  fig05_execution_rates.png D1/D2 碰撞率与安全完成率

本脚本不做任何模型改动，不写 summary，只生成图片。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle as MplCircle  # noqa: E402
from matplotlib.patches import Rectangle as MplRect  # noqa: E402

from fhd.evaluation.visualizations import AGENT_COLORS, _setup_fonts  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc  # noqa: E402
from fhd.geometry.static_maps import load_obstacles_for_map  # noqa: E402
from fhd.risk.conditional_execution_model import ExecutionModelParams  # noqa: E402
from fhd.risk.dynamic_first_hit import predict_pair_risks  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402

_setup_fonts()

MODE_COLORS = {0: "#4c72b0", 1: "#dd8452", 2: "#c44e52", 3: "#8172b3", 4: "#937860"}
MODE_NAMES = {0: "NORMAL", 1: "BRAKE_DWELL", 2: "BRAKE_PAUSE", 3: "HOLD", 4: "CREEP"}
PAIR_COLORS = {"m0": "#4c72b0", "m3": "#dd8452"}


# --------------------------------------------------------------------------- #
def load_tasks(n: int):
    bank_json = "plan_bank.json" if n == 2 else "plan_bank_n4.json"
    trajs_npz = "plan_bank_trajs.npz" if n == 2 else "plan_bank_n4_trajs.npz"
    bank = json.loads((ROOT / "data" / "mpd_delay_plan_bank" / bank_json)
                      .read_text(encoding="utf-8"))
    trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / trajs_npz)
    return {t["task_id"]: t for t in bank["tasks"]}, trajs


def draw_obstacles(ax, obstacles):
    if obstacles is None:
        return
    for c in obstacles.circles.numpy():
        ax.add_patch(MplCircle((c[0], c[1]), c[2], facecolor="#555555",
                               edgecolor="black", alpha=0.45, zorder=1))
    for r in obstacles.rects.numpy():
        ax.add_patch(MplRect((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                             facecolor="#555555", edgecolor="black", alpha=0.45, zorder=1))


def style_scene(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.18, linestyle=":")


def plot_paths(ax, paths, waits, title):
    for i in range(paths.shape[0]):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(paths[i, :, 0], paths[i, :, 1], "-", color=c, lw=1.6, alpha=0.9,
                label=f"机器人 {i}")
        ax.scatter(paths[i, 0, 0], paths[i, 0, 1], marker="o", s=45, color=c,
                   edgecolor="black", zorder=5)
        ax.scatter(paths[i, -1, 0], paths[i, -1, 1], marker="*", s=130, color=c,
                   edgecolor="black", zorder=5)
        stop = np.where(np.asarray(waits[i]) > 0)[0]
        if len(stop):
            ax.scatter(paths[i, stop, 0], paths[i, stop, 1], marker="X", s=55,
                       facecolor="white", edgecolor=c, linewidth=1.2, zorder=6)
        ax.annotate(f"G{i}", paths[i, -1], textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color=c)


def first_pair_event(run):
    if not run.collision or run.first_collision_pair is None:
        return None, None
    i, j = run.first_collision_pair
    return run.first_collision_time, (i, j)


def interp_traj(traj, t, dt=0.1):
    k = min(max(int(round(t / dt)), 0), traj.shape[0] - 1)
    return traj[k]


# --------------------------------------------------------------------------- #
def fig_scene_plans(out, n2_task, n4_task, pred):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.8))
    for row, (n, tid) in enumerate(((2, n2_task), (4, n4_task))):
        tasks, trajs = load_tasks(n)
        jt = tasks[tid]
        paths = np.asarray(trajs[tid], dtype=np.float64)
        wait_o = np.asarray(jt["baseline_wait"], dtype=np.float64)
        key = f"{tid}|w2plus|0.0"
        wait_w = np.asarray(pred[key]["wait"], dtype=np.float64)
        obs, _ = load_obstacles_for_map(jt["map_id"])

        ax = axes[row, 0]
        draw_obstacles(ax, obs)
        for i in range(n):
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(paths[i, :, 0], paths[i, :, 1], "-", color=c, lw=1.6, alpha=0.85,
                    label=f"机器人 {i}")
            ax.scatter(paths[i, 0, 0], paths[i, 0, 1], marker="o", s=45, color=c,
                       edgecolor="black", zorder=5)
            ax.scatter(paths[i, -1, 0], paths[i, -1, 1], marker="*", s=130, color=c,
                       edgecolor="black", zorder=5)
            ax.annotate(f"G{i}", paths[i, -1], textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color=c)
            idx_o = np.where(wait_o[i] > 0)[0]
            idx_w = np.where(wait_w[i] > 0)[0]
            if len(idx_o):
                ax.scatter(paths[i, idx_o, 0], paths[i, idx_o, 1], marker="o",
                           s=18 + 14 * wait_o[i, idx_o], facecolor="none",
                           edgecolor=c, linewidth=1.1, zorder=6)
            if len(idx_w):
                ax.scatter(paths[i, idx_w, 0], paths[i, idx_w, 1], marker="^",
                           s=22 + 14 * wait_w[i, idx_w], facecolor=c,
                           edgecolor="black", alpha=0.65, zorder=7)
        style_scene(ax, f"{tid} | N={n} | {jt['map_id']}")
        ax.set_title(f"{tid}\n路径/起终点；圆=原始等待点，三角=W2+等待点", fontsize=9)
        ax.legend(fontsize=6, loc="best")

        ax = axes[row, 1]
        mat = np.vstack([wait_o, wait_w])
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(np.arange(2 * n))
        ax.set_yticklabels([f"orig r{i}" for i in range(n)] +
                           [f"W2+ r{i}" for i in range(n)], fontsize=7)
        ax.set_xlabel("航点索引")
        ax.set_title(f"{tid}\n等待预算矩阵（时隙；0=不停）", fontsize=9)
        for i in range(2 * n):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                            fontsize=6, color="black")
        fig.colorbar(im, ax=ax, shrink=0.75, label="等待时隙")
    fig.suptitle("图 1  规划场景：冻结几何路径、起终点、障碍与等待计划",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    path = out / "fig01_scene_plans.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def fig_mpc_tracking(out, cases, cfg, pred, params):
    fig, axes = plt.subplots(len(cases), 3, figsize=(15.5, 4.4 * len(cases)))
    for r, case in enumerate(cases):
        n, tid, plan, p, seed = case[:5]
        pair_override = case[5] if len(case) > 5 else None
        tasks, trajs = load_tasks(n)
        jt = tasks[tid]
        paths = np.asarray(trajs[tid], dtype=np.float64)
        wait = np.asarray(pred[f"{tid}|{plan}|{p}"]["wait"], dtype=np.float64)
        obs, _ = load_obstacles_for_map(jt["map_id"])
        run = simulate_mpc(paths, wait, p,
                           MPCSimConfig(d_safe=cfg["d_safe"], horizon_steps=cfg["T"]),
                           seed=seed, record=True, obstacles=obs)
        traj = run.trajectories
        ax = axes[r, 0]
        draw_obstacles(ax, obs)
        for i in range(n):
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(paths[i, :, 0], paths[i, :, 1], "--", color=c, lw=0.9, alpha=0.45)
            ax.plot(traj[:, i, 0], traj[:, i, 1], "-", color=c, lw=1.6,
                    label=f"机器人 {i} MPC")
            ax.scatter(paths[i, 0, 0], paths[i, 0, 1], marker="o", s=30, color=c,
                       edgecolor="black", zorder=5)
            ax.scatter(paths[i, -1, 0], paths[i, -1, 1], marker="*", s=100, color=c,
                       edgecolor="black", zorder=5)
        t_hit, pair = first_pair_event(run)
        if pair is not None:
            pt = interp_traj(traj, t_hit)
            ax.scatter([pt[pair[0], 0], pt[pair[1], 0]],
                       [pt[pair[0], 1], pt[pair[1], 1]], marker="x", s=140,
                       color="red", linewidth=3, zorder=8,
                       label=f"首次碰撞 t={t_hit:.2f}s")
        style_scene(ax, f"{tid} | {plan} | p={p:g} | seed={seed}")
        ax.legend(fontsize=6, loc="best")

        # s(t) + 模式
        ax = axes[r, 1]
        if pair_override is not None:
            pair_plot = pair_override
        elif pair is None:
            pair_plot = (0, min(1, n - 1))
        else:
            pair_plot = pair
        for i in pair_plot:
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            t = np.arange(run.s_trace.shape[0]) * 0.1
            ax.plot(t, run.s_trace[:, i], "-", color=c, lw=1.4, label=f"机器人 {i} s(t)")
            modes = run.modes[:, i]
            hold = modes == 3
            ax.scatter(t[hold], run.s_trace[hold, i], s=6, color="black", alpha=0.35,
                       label=f"机器人 {i} HOLD" if i == pair_plot[0] else None)
        ax.set_xlabel("时间 [s]")
        ax.set_ylabel("弧长 s [m]")
        ax.set_title("MPC 真实进度 s(t)；黑点=停稳等待", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        # 关键 pair 距离
        ax = axes[r, 2]
        d = np.linalg.norm(traj[:, pair_plot[0], :] - traj[:, pair_plot[1], :], axis=1)
        t = np.arange(len(d)) * 0.1
        ax.plot(t, d, "-", color="#333333", lw=1.5,
                label=f"D2 实际 pair {pair_plot[0]}-{pair_plot[1]}")
        if pair is not None:
            ax.axvline(t_hit, color="red", ls="--", lw=1.0, label="D2 首次碰撞")
        ax.axhline(cfg["d_safe"], color="#c44e52", ls=":", lw=1.2, label="d_safe")
        ax.set_xlabel("时间 [s]")
        ax.set_ylabel("中心距离 [m]")
        ax.set_title(f"关键机器人对 {pair_plot[0]}-{pair_plot[1]} 距离", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("图 2  MPC 动力学跟踪：规划路径（虚线） vs 实际轨迹（实线）", fontsize=13, y=0.995)
    fig.tight_layout()
    path = out / "fig02_mpc_tracking.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def fig_m0m3(out, pairs, summary):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
    ax = axes[0, 0]
    for m, c in (("m0", "#4c72b0"), ("m3", "#dd8452")):
        ax.scatter(pairs[m], pairs["obs"], s=18, alpha=0.55, color=c,
                   label=f"{m.upper()}  MAE={np.mean(np.abs(pairs[m]-pairs['obs'])):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("预测两两首次碰撞概率")
    ax.set_ylabel("冻结测试 MPC 两两碰撞频率")
    ax.set_title("(a) 预测 vs 实测：M0 与 M3", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ps = sorted(summary["by_p"], key=float)
    x = np.arange(len(ps))
    width = 0.2
    for k, m in enumerate(("m0", "m1", "m2", "m3")):
        vals = [summary["by_p"][p][m]["mae"] for p in ps]
        ax.bar(x + (k - 1.5) * width, vals, width, label=m.upper(),
               color=["#4c72b0", "#5d8bc4", "#7aa3d3", "#dd8452"][k])
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in ps])
    ax.set_ylabel("pair MAE")
    ax.set_title("(b) 按暂停概率 p 分组", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 0]
    groups = ["N2", "N4"]
    x = np.arange(len(groups))
    for k, m in enumerate(("m0", "m1", "m2", "m3")):
        vals = [summary["by_group"][g][m]["mae"] for g in groups]
        ax.bar(x + (k - 1.5) * width, vals, width, label=m.upper(),
               color=["#4c72b0", "#5d8bc4", "#7aa3d3", "#dd8452"][k])
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("pair MAE")
    ax.set_title("(c) 按机器人数量分组", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 1]
    for m, c in (("m0", "#4c72b0"), ("m3", "#dd8452")):
        rows = summary["reliability"][m]
        xs = [r["mean_pred"] for r in rows]
        ys = [r["empirical_rate"] for r in rows]
        ns = [r["n"] for r in rows]
        ax.plot(xs, ys, "o-", color=c, label=m.upper(), lw=1.2, ms=5)
        for xx, yy, nn in zip(xs, ys, ns):
            ax.annotate(str(nn), (xx, yy), textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color=c)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("分箱平均预测风险")
    ax.set_ylabel("分箱经验碰撞频率")
    ax.set_title("(d) 可靠性曲线（数字=样本数）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle("图 3  M0–M3 风险模型修正效果（同一冻结测试集，30 MC/格）",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    path = out / "fig03_m0m3.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def fig_case_study(out, pairs, examples, fp_case):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
    for ax, (key, title) in zip(axes.ravel()[:3], examples):
        tid, plan, p = key
        sel = pairs[(pairs.task_id == tid) & (pairs.plan == plan)
                    & (np.abs(pairs.p - p) < 1e-9)].sort_values("pair")
        x = np.arange(len(sel))
        width = 0.17
        ax.bar(x - 2 * width, sel["m0"], width, label="M0", color="#4c72b0")
        ax.bar(x - width, sel["m1"], width, label="M1", color="#6b93c9")
        ax.bar(x, sel["m2"], width, label="M2", color="#9bb7da")
        ax.bar(x + width, sel["m3"], width, label="M3", color="#dd8452")
        ax.bar(x + 2 * width, sel["obs"], width, label="实测 MC", color="#55a868")
        ax.set_xticks(x)
        ax.set_xticklabels(sel["pair"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("首次碰撞概率 / 频率")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, axis="y")

    # 第 4 格：M3 虚假碰撞例的距离曲线（D2 vs M3 samples）
    ax = axes[1, 1]
    n, tid, plan, p, seed = fp_case[:5]
    tasks, trajs = load_tasks(n)
    paths = np.asarray(trajs[tid], dtype=np.float64)
    wait = np.asarray(EXTERNAL_PRED[f"{tid}|{plan}|{p}"]["wait"], dtype=np.float64)
    run = simulate_mpc(paths, wait, p,
                       MPCSimConfig(d_safe=EXTERNAL_CFG["d_safe"],
                                    horizon_steps=EXTERNAL_CFG["T"]),
                       seed=seed, record=True)
    m3 = predict_pair_risks(paths, wait, p, 38.4, EXTERNAL_PARAMS[n],
                            d_safe=EXTERNAL_CFG["d_safe"], n_samples=20, seed=7,
                            return_traces=True)
    tr = m3["trace_arrays"]
    # 每个样本对 1-2 的距离曲线
    pos = tr["positions"]
    d_m3 = np.linalg.norm(pos[:, :, 1, :] - pos[:, :, 2, :], axis=-1)
    t_m3 = tr["times"]
    for k in range(min(6, d_m3.shape[1])):
        ax.plot(t_m3, d_m3[:, k], color="#dd8452", lw=0.7, alpha=0.35)
    d_d2 = np.linalg.norm(run.trajectories[:, 1, :] - run.trajectories[:, 2, :], axis=1)
    ax.plot(np.arange(len(d_d2)) * 0.1, d_d2, color="#333333", lw=1.8,
            label="D2 实际 1-2 距离")
    ax.axhline(EXTERNAL_CFG["d_safe"], color="#c44e52", ls=":", lw=1.2, label="d_safe")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("距离 [m]")
    ax.set_title("M3 虚假碰撞例：橙线=20 条预测样本，黑线=D2 实际", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.suptitle("图 4  典型 case：两个真实碰撞 + 一个 M3 虚假碰撞", fontsize=13, y=0.995)
    fig.tight_layout()
    path = out / "fig04_case_study.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def fig_execution(out, p1):
    ps = p1["ps"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for g in ("N2", "N4"):
        for plan, ls in (("orig", "-"), ("w2plus", "--")):
            c = "#4c72b0" if g == "N2" else "#dd8452"
            y1 = [p1["by_group"][g][plan][str(p)]["d1_collision_rate"] for p in ps]
            y2 = [p1["by_group"][g][plan][str(p)]["d2_collision_rate"] for p in ps]
            axes[0].plot(ps, y1, ls, marker="o", color=c, alpha=0.45)
            axes[0].plot(ps, y2, ls, marker="s", color=c,
                         label=f"{g} {plan} (实线 D2 / 虚线 D1)")
            axes[1].plot(ps, [p1["by_group"][g][plan][str(p)]["d2_safe_rate"]
                              for p in ps], ls, marker="s", color=c,
                         label=f"{g} {plan}")
    axes[0].set_title("碰撞率：虚线 D1 / 实线 D2")
    axes[1].set_title("D2 安全完成率（无碰撞且全部到达）")
    for ax in axes:
        ax.set_xlabel("暂停概率 p")
        ax.set_ylabel("比例")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle("图 5  阶段一正式集：D1 理想执行 vs D2 MPC 动力学执行", fontsize=12)
    fig.tight_layout()
    path = out / "fig05_execution_rates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
EXTERNAL_PRED = {}
EXTERNAL_CFG = {}
EXTERNAL_PARAMS = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/Online_FirstHit_v2/figures_overview")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / "configs" / "base.yaml")
    risk = FirstHitRisk.from_config(cfg)
    T = int(cfg["execution"]["horizon_steps"])
    d_safe = risk.d_safe
    EXTERNAL_CFG.update({"T": T, "d_safe": d_safe})

    pred = json.loads((ROOT / "results" / "Online_FirstHit_v2" / "03_risk_baselines"
                       / "predictions_m0m1m2.json").read_text(encoding="utf-8"))["predictions"]
    EXTERNAL_PRED.update(pred)
    EXTERNAL_PARAMS.update({
        2: ExecutionModelParams.load(ROOT / "results" / "Online_FirstHit_v2"
                                     / "04_dynamic_risk_m3" / "m3_params_n2.json"),
        4: ExecutionModelParams.load(ROOT / "results" / "Online_FirstHit_v2"
                                     / "04_dynamic_risk_m3" / "m3_params_n4.json"),
    })

    p1 = json.loads((ROOT / "results" / "Online_FirstHit_v2" / "02_mpc_benchmark"
                     / "summary.json").read_text(encoding="utf-8"))
    m3 = json.loads((ROOT / "results" / "Online_FirstHit_v2" / "04_dynamic_risk_m3"
                     / "summary.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(ROOT / "results" / "Online_FirstHit_v2" / "04_dynamic_risk_m3"
                        / "risk_pairs.csv")

    paths = []
    paths.append(fig_scene_plans(out, "open_crossing_300009_pair07_13",
                                 "open_crossing_300000_n4_0", pred))

    cases = [
        (4, "open_crossing_300000_n4_0", "orig", 0.1, 20282915),
        (2, "open_crossing_300009_pair07_13", "orig", 0.2, 20347915),
        (4, "open_crossing_300012_n4_0", "w2plus", 0.0, 20282915, (1, 2)),
    ]
    paths.append(fig_mpc_tracking(out, cases, {"T": T, "d_safe": d_safe}, pred,
                                  EXTERNAL_PARAMS))
    paths.append(fig_m0m3(out, pairs, m3))

    examples = [
        (("open_crossing_300000_n4_0", "orig", 0.1),
         "N4 orig p=0.1：真实高碰撞, M3 低估"),
        (("open_crossing_300009_pair07_13", "orig", 0.2),
         "N2 orig p=0.2：真实碰撞, M0/M3 都低估"),
        (("open_crossing_300012_n4_0", "w2plus", 0.0),
         "N4 W2+ p=0：M0 正确为 0，M3 虚假预测 1"),
    ]
    paths.append(fig_case_study(out, pairs, examples, cases[2]))
    paths.append(fig_execution(out, p1))

    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
