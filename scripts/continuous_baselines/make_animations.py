"""Animated side-by-side comparison from frozen_v1 NPZ trajectories."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/Continuous_Baseline_Comparison/credible_v1/frozen_v1"
MAN = ROOT / "results/Continuous_Baseline_Comparison/credible_v1/task_manifest.json"
OUT = Path(r"D:\paper2\_figs"); OUT.mkdir(parents=True, exist_ok=True)
manifest = {t["task_id"]: t for t in json.loads(MAN.read_text(encoding="utf-8"))["tasks"]}
methods = [("ours_frozen", "Ours"), ("orca_rvo2", "ORCA"), ("mpc_cbf", "MPC-CBF")]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
task_order = list(manifest)


def load(m, tid, p, seed):
    d = np.load(RES / "raw" / f"{m}_{tid}_p{p}_s{seed}.npz")
    return d["times"], d["positions"], d["velocities"]


def make(tid, p, seed, name):
    task = manifest[tid]
    data = {m: load(m, tid, p, seed) for m, _ in methods}
    tmax = max(float(d[0][-1]) for d in data.values())
    frames = np.arange(0.0, tmax + 1e-9, 0.2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharex=True, sharey=True)
    paths = np.asarray(task["paths"], float)
    for ax, (m, label) in zip(axes, methods):
        for r in task["rects"]:
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                                   facecolor="0.55", edgecolor="k", alpha=0.65, zorder=1))
        for i in range(paths.shape[0]):
            ax.plot(paths[i, :, 0], paths[i, :, 1], color="0.82", lw=1.0, zorder=0)
            ax.scatter(paths[i, -1, 0], paths[i, -1, 1], marker="*", s=100,
                       color=colors[i], edgecolor="k", zorder=6)
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.set_title(f"{label}  {tid}  p={p}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(alpha=0.2)
    lines = {}
    dots = {}
    for ax, (m, _) in zip(axes, methods):
        lines[m] = [ax.plot([], [], color=colors[i], lw=1.3, alpha=0.8)[0]
                    for i in range(4)]
        dots[m] = [ax.scatter([], [], color=colors[i], s=90, edgecolor="k", zorder=8)
                   for i in range(4)]
    time_txt = [ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                        fontsize=9, bbox=dict(fc="white", alpha=0.75, ec="none"))
                for ax in axes]
    dist_txt = [ax.text(0.02, 0.91, "", transform=ax.transAxes, va="top",
                        fontsize=9, bbox=dict(fc="white", alpha=0.75, ec="none"))
                for ax in axes]

    def update(fi):
        t = frames[fi]
        for ax, (m, _) in zip(axes, methods):
            tt, pos, _ = data[m]
            k = int(np.searchsorted(tt, t, side="right") - 1)
            k = max(0, min(k, len(tt) - 1))
            for i in range(4):
                upto = pos[:k + 1, i]
                lines[m][i].set_data(upto[:, 0], upto[:, 1])
                dots[m][i].set_offsets(pos[k, i])
            d = [np.linalg.norm(pos[k, i] - pos[k, j])
                 for i in range(4) for j in range(i + 1, 4)]
            time_txt[methods.index((m, dict(methods)[m]))].set_text(f"t = {t:4.1f} s")
            dist_txt[methods.index((m, dict(methods)[m]))].set_text(f"min pair = {min(d):.2f} m")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    anim.save(OUT / name, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("saved", OUT / name, "frames", len(frames))


if __name__ == "__main__":
    make("A1_open_orthogonal", 0.1, 20281701, "anim_A1_open.gif")
    make("B1_block_below_above", 0.1, 20281901, "anim_B1_static.gif")
    make("B1_block_below_above", 0.2, 20281901, "anim_B1_static_p02.gif")
