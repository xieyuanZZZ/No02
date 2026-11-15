"""Make quick-look figures from frozen_v1 NPZ trajectories."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(r"D:\paper2\_figs"); OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results/Continuous_Baseline_Comparison/credible_v1/frozen_v1"
MAN = ROOT / "results/Continuous_Baseline_Comparison/credible_v1/task_manifest.json"
manifest = {t["task_id"]: t for t in json.loads(MAN.read_text(encoding="utf-8"))["tasks"]}
methods = [("ours_frozen", "Ours"), ("orca_rvo2", "ORCA"), ("mpc_cbf", "MPC-CBF")]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
tasks = ["A1_open_orthogonal", "B1_block_below_above"]
p = 0.1
task_order = list(manifest)
for ti, tid in enumerate(tasks):
    task = manifest[tid]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharex=True, sharey=True)
    seed = 20281701 + task_order.index(tid) * 100
    for ax, (m, label) in zip(axes, methods):
        f = RES / "raw" / f"{m}_{tid}_p{p}_s{seed}.npz"
        d = np.load(f)
        pos = d["positions"]
        for i in range(pos.shape[1]):
            ax.plot(pos[:, i, 0], pos[:, i, 1], color=colors[i], lw=1.4,
                    label=f"R{i}")
            ax.scatter(pos[0, i, 0], pos[0, i, 1], color=colors[i], s=28,
                       marker="o", edgecolor="k", zorder=5)
            ax.scatter(task["paths"][i][-1][0], task["paths"][i][-1][1],
                       color=colors[i], s=70, marker="*", edgecolor="k", zorder=6)
        for r in task["rects"]:
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                                   facecolor="0.55", edgecolor="k", alpha=0.6))
        ax.set_title(f"{label}  {tid}  p={p}")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Actual trajectories, p={p}, seed={seed}")
    fig.tight_layout()
    fig.savefig(OUT / f"traj_{tid}.png", dpi=150)
    plt.close(fig)

# min distance curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ti, (ax, tid) in enumerate(zip(axes, tasks)):
    seed = 20281701 + task_order.index(tid) * 100
    for m, label in methods:
        d = np.load(RES / "raw" / f"{m}_{tid}_p{p}_s{seed}.npz")
        pos = d["positions"]; t = d["times"]
        n = pos.shape[1]
        vals = []
        for k in range(len(t)):
            dd = [np.linalg.norm(pos[k, i] - pos[k, j])
                  for i in range(n) for j in range(i + 1, n)]
            vals.append(min(dd))
        ax.plot(t, vals, label=label, lw=1.6)
    ax.axhline(0.35, color="k", ls="--", lw=1, label="d_safe=0.35")
    ax.set_title(tid); ax.set_xlabel("time [s]"); ax.set_ylabel("min pair distance [m]")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "min_distance.png", dpi=150)
plt.close(fig)

# summary bars
s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["by"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
labels = ["Ours", "ORCA", "MPC-CBF"]
keys = [("ours_frozen", "open"), ("orca_rvo2", "open"), ("mpc_cbf", "open")]
keys2 = [("ours_frozen", "static"), ("orca_rvo2", "static"), ("mpc_cbf", "static")]
for ax, kk, title in [(axes[0], keys, "Open"), (axes[1], keys2, "Static obstacles")]:
    vals = [s[f"{m}|{sc}"]["safe_rate"] for m, sc in kk]
    ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylim(0, 1.05); ax.set_title(title); ax.set_ylabel("safe completion rate")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "safe_rates.png", dpi=150)
plt.close(fig)
print("saved", OUT)
