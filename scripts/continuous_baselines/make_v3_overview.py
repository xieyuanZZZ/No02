from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fhd.adapters.scene_variation import sample_variant_retry

MAN = ROOT / "results/Continuous_Baseline_Comparison/credible_v3/task_manifest.json"
OUT = Path(r"D:\paper2\_figs"); OUT.mkdir(parents=True, exist_ok=True)
tasks = json.loads(MAN.read_text(encoding="utf-8"))["tasks"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def draw(ax, t, seed=None):
    if seed is not None:
        t = sample_variant_retry(t, seed)
    for r in t["rects"]:
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                               color="0.6", ec="k", alpha=0.7, zorder=1))
    for i, p in enumerate(t["paths"]):
        p = np.asarray(p)
        ax.plot(p[:, 0], p[:, 1], color=colors[i], lw=1.2)
        ax.scatter(p[0, 0], p[0, 1], color=colors[i], s=20, ec="k", zorder=5)
        ax.scatter(p[-1, 0], p[-1, 1], marker="*", color=colors[i], s=65,
                   ec="k", zorder=6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.grid(alpha=0.2); ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])


fig, axes = plt.subplots(3, 4, figsize=(16, 11))
for ax, t in zip(axes.ravel(), tasks):
    draw(ax, t); ax.set_title(t["task_id"], fontsize=9)
fig.suptitle("credible_v3: 4 open + 8 static tasks, A*-planned reference paths")
fig.tight_layout()
fig.savefig(OUT / "credible_v3_scenes.png", dpi=150)
plt.close(fig)

base = [t for t in tasks if t["task_id"] == "B8_l_shape"][0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
draw(axes[0], base); axes[0].set_title("B8 base")
for j, seed in enumerate([11, 22, 33], start=1):
    draw(axes[j], base, seed); axes[j].set_title(f"B8 variant seed={seed}")
fig.suptitle("Per-run jitter: starts/goals/obstacles + re-planned A* paths")
fig.tight_layout()
fig.savefig(OUT / "credible_v3_variants.png", dpi=150)
plt.close(fig)
print("saved")
