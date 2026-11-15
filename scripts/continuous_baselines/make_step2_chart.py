from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(r"D:\paper2\_figs"); OUT.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

ax = axes[0]
ax.bar(["hard CBF", "soft CBF"], [63, 1], color=["#d62728", "#2ca02c"])
ax.set_title("MPC-CBF, B1 p=0")
ax.set_ylabel("IPOPT failures")
for i, v in enumerate([63, 1]):
    ax.text(i, v + 1, str(v), ha="center")
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
ax.bar(["before", "after"], [0.42, 0.019], color=["#d62728", "#2ca02c"])
ax.set_title("ORCA endpoint error")
ax.set_ylabel("max goal error [m]")
for i, v in enumerate([0.42, 0.019]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
ax.grid(axis="y", alpha=0.25)

ax = axes[2]
ax.bar(["passed", "failed"], [11, 0], color=["#2ca02c", "#d62728"])
ax.set_title("common evaluator tests")
ax.set_ylabel("tests")
for i, v in enumerate([11, 0]):
    ax.text(i, v + 0.2, str(v), ha="center")
ax.set_ylim(0, 13)
ax.grid(axis="y", alpha=0.25)

fig.suptitle("Step 2: baseline and evaluator fixes (B1, p=0)")
fig.tight_layout()
fig.savefig(OUT / "step2_fixes.png", dpi=160)
plt.close(fig)
print("saved", OUT / "step2_fixes.png")
