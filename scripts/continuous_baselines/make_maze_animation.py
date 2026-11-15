"""Animate the curved maze scene, p=0.1 seed 20281701."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/Continuous_Baseline_Comparison/maze_v1/small_pilot"
MAN = ROOT / "results/Continuous_Baseline_Comparison/maze_v1/task_manifest.json"
OUT = Path(r"D:\paper2\_figs"); OUT.mkdir(parents=True, exist_ok=True)
task = json.loads(MAN.read_text(encoding="utf-8"))["tasks"][0]
methods = [("ours_frozen", "Ours"), ("orca_rvo2", "ORCA"), ("mpc_cbf", "MPC-CBF")]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
p, seed = 0.1, 20281701
data = {m: np.load(RES / "raw" / f"{m}_{task['task_id']}_p{p}_s{seed}.npz")
        for m, _ in methods}
tmax = max(float(d["times"][-1]) for d in data.values())
frames = np.arange(0, tmax + 1e-9, 0.2)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, (m, label) in zip(axes, methods):
    for r in task["rects"]:
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                               color="0.55", ec="k", alpha=0.7, zorder=1))
    paths = np.asarray(task["paths"])
    for i in range(4):
        ax.plot(paths[i, :, 0], paths[i, :, 1], color="0.85", lw=0.8, zorder=0)
        ax.scatter(paths[i, -1, 0], paths[i, -1, 1], marker="*", s=90,
                   color=colors[i], edgecolor="k", zorder=6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title(f"{label}  maze  p={p}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=0.2)
lines = {}; dots = {}
for ax, (m, _) in zip(axes, methods):
    lines[m] = [ax.plot([], [], color=colors[i], lw=1.4)[0] for i in range(4)]
    dots[m] = [ax.scatter([], [], color=colors[i], s=90, ec="k", zorder=8)
               for i in range(4)]
tt = [ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
              bbox=dict(fc="white", alpha=0.75, ec="none")) for ax in axes]
dd = [ax.text(0.02, 0.91, "", transform=ax.transAxes, va="top", fontsize=9,
              bbox=dict(fc="white", alpha=0.75, ec="none")) for ax in axes]


def update(fi):
    t = frames[fi]
    for j, (m, _) in enumerate(methods):
        d = data[m]; times = d["times"]; pos = d["positions"]
        k = max(0, min(int(np.searchsorted(times, t, side="right") - 1), len(times) - 1))
        for i in range(4):
            lines[m][i].set_data(pos[:k + 1, i, 0], pos[:k + 1, i, 1])
            dots[m][i].set_offsets(pos[k, i])
        pair = min(np.linalg.norm(pos[k, i] - pos[k, jj])
                   for i in range(4) for jj in range(i + 1, 4))
        tt[j].set_text(f"t = {t:4.1f} s")
        dd[j].set_text(f"min pair = {pair:.2f} m")
    return []


anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
anim.save(OUT / "anim_maze.gif", writer=PillowWriter(fps=8))
plt.close(fig)
print("saved", OUT / "anim_maze.gif", len(frames))
