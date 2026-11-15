from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
SRC = Path(os.environ.get("V3_SRC", r"D:\paper2\_v3_medium1"))
rows = [json.loads(x) for x in (SRC / "per_run.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
rows = [r for r in rows if r.get("error") is None]
methods = [("ours_frozen", "Ours"), ("orca_rvo2", "ORCA"), ("mpc_cbf", "MPC-CBF")]
tasks = sorted({r["task_id"] for r in rows})
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
# summary
out = {"by_method": {}, "by_task_method": {}}
for m, label in methods:
    sel = [r for r in rows if r["method"] == m]
    out["by_method"][label] = {
        "n": len(sel),
        "safe_rate": float(np.mean([r["safe_completion"] for r in sel])),
        "collision_rate": float(np.mean([r["collision"] for r in sel])),
        "static_collision_rate": float(np.mean([r["static_collision"] for r in sel
                                                if r["static_collision"] is not None])) if any(r["static_collision"] is not None for r in sel) else None,
        "completion_mean": float(np.mean([r["completion_s"] for r in sel])),
    }
    for t in tasks:
        ss = [r for r in sel if r["task_id"] == t]
        out["by_task_method"][f"{label}|{t}"] = {
            "n": len(ss),
            "safe_rate": float(np.mean([r["safe_completion"] for r in ss])) if ss else None,
        }
(Path(r"D:\paper2\_v3_summary.json")).write_text(json.dumps(out, indent=2), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2.4, 1]})
ax = axes[0]
x = np.arange(len(tasks)); w = 0.25
for j, (m, label) in enumerate(methods):
    vals = [out["by_task_method"].get(f"{label}|{t}", {}).get("safe_rate") or 0 for t in tasks]
    ax.bar(x + (j - 1) * w, vals, w, label=label, color=colors[j])
ax.set_xticks(x); ax.set_xticklabels([t.split("_")[0] for t in tasks], rotation=0)
ax.set_ylim(0, 1.05); ax.set_ylabel("safe completion rate")
ax.set_title("credible_v3: safe completion by task")
ax.legend(); ax.grid(axis="y", alpha=0.25)
ax = axes[1]
labels = [label for _, label in methods]
vals = [out["by_method"][label]["safe_rate"] for label in labels]
bars = ax.bar(labels, vals, color=colors)
ax.set_ylim(0, 1.05); ax.set_ylabel("safe completion rate")
ax.set_title("overall")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
ax.grid(axis="y", alpha=0.25)
fig.suptitle("Randomized medium pilot — 12 tasks, p=0/0.1/0.2")
fig.tight_layout()
fig.savefig(r"D:\paper2\_figs\v3_medium_summary.png", dpi=160)
plt.close(fig)
print(json.dumps(out["by_method"], indent=1))
