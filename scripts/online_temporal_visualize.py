"""A/B/C 典型 case 可视化：x-y 轨迹 + s(t)/模式 + 在线决策时刻。

用法示例：
  python scripts/online_temporal_visualize.py --task-id open_crossing_300000_n4_0 \
      --p 0.1 --seed 20281116
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fhd.evaluation.visualizations import AGENT_COLORS, _setup_fonts  # noqa: E402
from fhd.execution.mpc_simulator import (MPCSimConfig, make_pause_events,  # noqa: E402
                                         simulate_mpc)
from fhd.geometry.static_maps import load_obstacles_for_map  # noqa: E402
from fhd.optim.online_baseline import SimpleConflictCoordinator  # noqa: E402
from fhd.optim.online_firsthit import OnlineFirstHitCoordinator  # noqa: E402
from fhd.optim.wait_policies import (WaitBudget, allowed_positions, make_mask,  # noqa: E402
                                     policy_w0, policy_w2plus)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402

_setup_fonts()


def _w2plus(paths, w0, p_pause, risk, T, H, n, seed=0):
    budget = WaitBudget(time_steps=T, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), n)
    paths_t = torch.tensor(paths, dtype=torch.float64)
    w0_t = torch.tensor(w0, dtype=torch.float64)
    pause01 = torch.full((n,), p_pause, dtype=torch.float64)
    return policy_w2plus(w0_t, mask, budget, risk, paths_t, pause01, T,
                         lam_T=0.05, max_evals=300, restarts=2, seed=seed).wait.numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default="open_crossing_300000_n4_0")
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20281116)
    ap.add_argument("--out", default="results/Online_Temporal/figures")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    base = load_config(ROOT / "configs" / "base.yaml")
    T = int(base["execution"]["horizon_steps"])
    H = int(base["trajectory"]["horizon"])
    dt_slot = float(base["trajectory"]["dt_s"])
    risk = FirstHitRisk.from_config(base)
    d_safe = risk.d_safe
    bank = json.loads((ROOT / "data" / "mpd_delay_plan_bank" / "plan_bank_n4.json")
                      .read_text(encoding="utf-8"))
    trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / "plan_bank_n4_trajs.npz")
    jt = [t for t in bank["tasks"] if t["task_id"] == args.task_id][0]
    paths = np.asarray(trajs[args.task_id], dtype=np.float64)
    w0 = np.asarray(jt["baseline_wait"], dtype=np.float64)
    w2 = _w2plus(paths, w0, 0.1, risk, T, H, 4, seed=0)
    obs, meta = load_obstacles_for_map(jt["map_id"])
    cfg = MPCSimConfig(d_safe=d_safe, horizon_steps=T)
    events = make_pause_events(args.seed, args.p, 4, T, cfg)

    methods = [
        ("A 离线 W2+", w2, None),
        ("B 简单在线", w0, SimpleConflictCoordinator(
            paths, w0, d_safe, dt_slot, 3.0, args.p, {"hold_slots": 1})),
        ("C First-Hit 在线", w0, OnlineFirstHitCoordinator(
            paths, w0, d_safe, dt_slot, 3.2, args.p, risk_model=risk,
            config={"hold_slots": 1})),
    ]
    runs = []
    for name, wait, coord in methods:
        r = simulate_mpc(paths, wait, args.p, cfg, seed=args.seed, record=True,
                         events=events, obstacles=obs, obstacle_meta=meta,
                         coordinator=coord)
        runs.append((name, r))

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for row, (name, r) in enumerate(runs):
        ax = axes[row, 0]
        for i in range(4):
            c = AGENT_COLORS[i]
            ax.plot(paths[i, :, 0], paths[i, :, 1], "--", color=c, lw=0.8, alpha=0.35)
            ax.plot(r.trajectories[:, i, 0], r.trajectories[:, i, 1], "-", color=c,
                    lw=1.5, label=f"机器人 {i}")
            ax.scatter(paths[i, 0, 0], paths[i, 0, 1], marker="o", s=30, color=c,
                       edgecolor="black", zorder=5)
            ax.scatter(paths[i, -1, 0], paths[i, -1, 1], marker="*", s=100, color=c,
                       edgecolor="black", zorder=5)
        if r.first_collision_pair is not None:
            pa = r.first_collision_time
            k = int(min(max(pa / 0.1, 0), len(r.trajectories) - 1))
            i, j = r.first_collision_pair
            ax.scatter([r.trajectories[k, i, 0], r.trajectories[k, j, 0]],
                       [r.trajectories[k, i, 1], r.trajectories[k, j, 1]],
                       marker="x", s=150, color="red", linewidth=3, zorder=8,
                       label=f"碰撞 t={pa:.2f}s")
        else:
            ax.set_title(f"{name}：无机器人间碰撞", fontsize=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        ax = axes[row, 1]
        for i in range(4):
            c = AGENT_COLORS[i]
            t = np.arange(r.s_trace.shape[0]) * 0.1
            ax.plot(t, r.s_trace[:, i], "-", color=c, lw=1.4, label=f"r{i}")
            hold = r.modes[:, i] == 3
            ax.scatter(t[hold], r.s_trace[hold, i], s=7, color="black", alpha=0.35)
        for d in (r.coordinator_log or []):
            for a in d.get("actions", []):
                if a.get("accepted"):
                    ax.axvline(d["t"], color="#c44e52", ls=":", lw=0.8, alpha=0.6)
        ax.set_xlabel("时间 [s]")
        ax.set_ylabel("弧长 s [m]")
        ax.set_title(f"{name}：s(t)，黑点=HOLD，红线=在线停车命令\n"
                     f"碰撞={r.collision} 安全完成={r.safe_completion} "
                     f"在线等待={r.online_wait_time:.1f}s 决策P95="
                     f"{r.coordinator_stats.get('p95_ms', 0):.1f}ms", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle(f"A/B/C 典型 case：{args.task_id}  p={args.p}  seed={args.seed}",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    path = out / f"online_case_{args.task_id}_p{args.p}_seed{args.seed}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
