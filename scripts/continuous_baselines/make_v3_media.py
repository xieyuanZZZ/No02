"""Generate 4 representative v3 animations and summary charts."""
from pathlib import Path
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("V3_SRC", r"D:\paper2\_v3_medium1"))
MAN = ROOT / "results/Continuous_Baseline_Comparison/credible_v3/task_manifest.json"
OUT = Path(r"D:\paper2\_figs"); OUT.mkdir(parents=True, exist_ok=True)
tasks = json.loads(MAN.read_text(encoding="utf-8"))["tasks"]
tmap = {t["task_id"]: t for t in tasks}
order = [t["task_id"] for t in tasks]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
methods = [("ours_frozen", "Ours"), ("orca_rvo2", "ORCA"), ("mpc_cbf", "MPC-CBF")]
chosen = ["A3_open_multi_cross", "B4_staggered_blocks", "B6_two_blocks_detour", "B8_l_shape"]
p = 0.1


def animate(tid):
    seed = 20281701 + order.index(tid) * 100
    data = {m: np.load(RES / "raw" / f"{m}_{tid}_p{p}_s{seed}.npz") for m, _ in methods}
    tmax = max(float(d["times"][-1]) for d in data.values())
    frames = np.arange(0, tmax + 1e-9, 0.2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    t = tmap[tid]
    for ax, (m, label) in zip(axes, methods):
        for r in t["rects"]:
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                                   color="0.58", ec="k", alpha=0.75, zorder=1))
        for i, pth in enumerate(t["paths"]):
            pth = np.asarray(pth)
            ax.plot(pth[:, 0], pth[:, 1], color="0.86", lw=0.8, zorder=0)
            ax.scatter(pth[-1, 0], pth[-1, 1], marker="*", s=80,
                       color=colors[i], ec="k", zorder=5)
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.set_title(f"{label}  {tid}  p={p}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=0.2)
    lines = {}; dots = {}
    for ax, (m, _) in zip(axes, methods):
        lines[m] = [ax.plot([], [], color=colors[i], lw=1.3)[0] for i in range(4)]
        dots[m] = [ax.scatter([], [], color=colors[i], s=80, ec="k", zorder=8)
                   for i in range(4)]
    tt = [ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
                  bbox=dict(fc="white", alpha=0.75, ec="none")) for ax in axes]
    dd = [ax.text(0.02, 0.91, "", transform=ax.transAxes, va="top", fontsize=9,
                  bbox=dict(fc="white", alpha=0.75, ec="none")) for ax in axes]

    def update(fi):
        tcur = frames[fi]
        for j, (m, _) in enumerate(methods):
            d = data[m]; k = max(0, min(int(np.searchsorted(d["times"], tcur, side="right") - 1),
                                        len(d["times"]) - 1))
            pos = d["positions"]
            for i in range(4):
                lines[m][i].set_data(pos[:k + 1, i, 0], pos[:k + 1, i, 1])
                dots[m][i].set_offsets(pos[k, i])
            pair = min(np.linalg.norm(pos[k, i] - pos[k, jj])
                       for i in range(4) for jj in range(i + 1, 4))
            tt[j].set_text(f"t = {tcur:4.1f} s")
            dd[j].set_text(f"min pair = {pair:.2f} m")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    fn = OUT / f"anim_v3_{tid}.gif"
    anim.save(fn, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("saved", fn)


if __name__ == "__main__":
    for tid in chosen:
        animate(tid)
