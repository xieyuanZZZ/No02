from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
old = json.loads(Path(r"D:\paper2\_old_manifest.json").read_text(encoding="utf-8-sig"))
new = json.loads((ROOT / "results/Continuous_Baseline_Comparison/credible_v3/task_manifest.json").read_text(encoding="utf-8"))
ids = ["B4_staggered_blocks", "B6_two_blocks_detour", "B8_l_shape"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
for col, tid in enumerate(ids):
    for row, (data, tag) in enumerate([(old, "BEFORE"), (new, "AFTER")]):
        ax = axes[row, col]
        t = [x for x in data["tasks"] if x["task_id"] == tid][0]
        for r in t["rects"]:
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                                   color="0.62", ec="k", alpha=0.7))
        for i, p in enumerate(t["paths"]):
            p = np.asarray(p)
            ax.plot(p[:, 0], p[:, 1], color=colors[i], lw=1.4)
            ax.scatter(p[0, 0], p[0, 1], color=colors[i], s=22, ec="k")
            ax.scatter(p[-1, 0], p[-1, 1], marker="*", color=colors[i], s=65, ec="k")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"{tag}: {tid}\nclearance {t['min_path_obstacle_clearance']:.2f} m", fontsize=9)
fig.suptitle("B4/B6/B8 redesign: before vs after (same 10x10, 4 agents)")
fig.tight_layout()
out = Path(r"D:\paper2\_figs\credible_v3_redesign_compare.png")
fig.savefig(out, dpi=160)
plt.close(fig)
print("saved", out)
