"""Final 12-scene overview with a new filename to avoid image cache confusion."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "results/Continuous_Baseline_Comparison/credible_v3/task_manifest.json"
tasks = json.loads(MAN.read_text(encoding="utf-8"))["tasks"]
out = Path(r"D:\paper2\_figs")
out.mkdir(parents=True, exist_ok=True)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def draw(ax, t):
    for r in t["rects"]:
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                               color="0.58", ec="k", alpha=0.8, zorder=1))
    for i, p in enumerate(t["paths"]):
        p = np.asarray(p)
        ax.plot(p[:, 0], p[:, 1], color=colors[i], lw=1.5, zorder=3)
        ax.scatter(p[0, 0], p[0, 1], color=colors[i], s=28, ec="k", zorder=5)
        ax.scatter(p[-1, 0], p[-1, 1], marker="*", color=colors[i], s=85,
                   ec="k", zorder=6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
    ax.grid(alpha=0.2)
    ax.set_xlabel("x [m]", fontsize=8); ax.set_ylabel("y [m]", fontsize=8)


fig, axes = plt.subplots(3, 4, figsize=(17, 12))
for ax, t in zip(axes.ravel(), tasks):
    draw(ax, t)
    ax.set_title(t["task_id"], fontsize=9,
                 fontweight="bold" if t["task_id"] in
                 ("B4_staggered_blocks", "B6_two_blocks_detour", "B8_l_shape") else "normal")
fig.suptitle("credible_v3 FINAL — 12 tasks: 4 open + 8 static (B4/B6/B8 redesigned)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out / "credible_v3_scenes_FINAL_20261115.png", dpi=170)
plt.close(fig)

# B4/B6/B8 detail with new filename
ids = ["B4_staggered_blocks", "B6_two_blocks_detour", "B8_l_shape"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, tid in zip(axes, ids):
    t = [x for x in tasks if x["task_id"] == tid][0]
    draw(ax, t)
    ax.set_title(f"{tid}\nclearance={t['min_path_obstacle_clearance']:.2f} m",
                 fontsize=10, fontweight="bold")
fig.suptitle("Redesigned B4 (Z), B6 (diagonal/detour), B8 (nested L)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out / "credible_v3_B4_B6_B8_FINAL_20261115.png", dpi=170)
plt.close(fig)
print("saved final overviews")
