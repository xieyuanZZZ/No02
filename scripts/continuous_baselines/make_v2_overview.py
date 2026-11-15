from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "results/Continuous_Baseline_Comparison/credible_v2/task_manifest.json"
OUT = Path(r"D:\paper2\_figs"); OUT.mkdir(parents=True, exist_ok=True)
tasks = json.loads(MAN.read_text(encoding="utf-8"))["tasks"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for ax, t in zip(axes.ravel(), tasks):
    for r in t["rects"]:
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                               color="0.6", ec="k", alpha=0.7, zorder=1))
    for i, p in enumerate(t["paths"]):
        p = np.asarray(p)
        ax.plot(p[:, 0], p[:, 1], color=colors[i], lw=1.3, label=f"R{i}")
        ax.scatter(p[0, 0], p[0, 1], color=colors[i], s=25, ec="k", zorder=5)
        ax.scatter(p[-1, 0], p[-1, 1], marker="*", color=colors[i], s=75,
                   ec="k", zorder=6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title(t["task_id"], fontsize=9)
    ax.grid(alpha=0.2); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
axes[0, 0].legend(fontsize=7, loc="upper left")
fig.suptitle("credible_v2: 4 open crossings + 4 static-obstacle tasks (N=4)")
fig.tight_layout()
fig.savefig(OUT / "credible_v2_scenes.png", dpi=160)
plt.close(fig)
print("saved", OUT / "credible_v2_scenes.png")
