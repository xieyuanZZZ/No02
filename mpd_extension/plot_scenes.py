"""画 A(prior) vs B(mpd) 在我们三类场景上的轨迹对比图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle, Rectangle as MplRect

ROOT = Path("/mnt/d/mpd_official/results_mpd")
OUT = Path("/mnt/d/paper2/docs/mpd_figures")
OUT.mkdir(parents=True, exist_ok=True)

KIND_CN = {"open_crossing": "Open crossing", "single_obstacle": "Single obstacle",
           "bottleneck_channel": "Bottleneck channel"}


def load(planner, kind, limit=1):
    d = ROOT / f"scenes_{planner}"
    out = []
    for f in sorted(d.glob("traj_task*.npz")):
        z = np.load(f, allow_pickle=True)
        if str(z["kind"][0]) == kind:
            out.append(z)
            if len(out) >= limit:
                break
    return out


def draw(ax, z, title, show_guidance=False):
    boxes = z["boxes"]
    circles = z["circles"]
    for b in boxes:
        ax.add_patch(MplRect((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                             facecolor="#888", edgecolor="k", alpha=0.6, zorder=2))
    for c in circles:
        ax.add_patch(MplCircle((c[0], c[1]), c[2], facecolor="#888", edgecolor="k",
                               alpha=0.6, zorder=2))
    trajs = z["trajs"]
    for k in range(trajs.shape[0]):
        ax.plot(trajs[k, :, 0], trajs[k, :, 1], lw=1.2, alpha=0.85)
    s, g = z["start"], z["goal"]
    ax.plot(s[0], s[1], "o", ms=9, mfc="#2ca02c", mec="k", zorder=5)
    ax.plot(g[0], g[1], "s", ms=9, mfc="#d62728", mec="k", zorder=5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, ls=":")


def main():
    kinds = ["open_crossing", "single_obstacle", "bottleneck_channel"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for j, kind in enumerate(kinds):
        for i, planner in enumerate(["diffusion_prior", "mpd"]):
            zs = load(planner, kind, 1)
            ax = axes[i, j]
            if not zs:
                ax.set_axis_off()
                continue
            label = "A: Diffusion Prior" if planner == "diffusion_prior" else "B: MPD (SDF guide)"
            draw(ax, zs[0], f"{KIND_CN[kind]} — {label}")
    fig.suptitle("Official MPD on No02 scenes (8 candidates per task; green=start, red=goal, grey=obstacles)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "mpd_scenes_AB.png", dpi=130)
    plt.close(fig)

    # 汇总柱状图
    summ = {}
    for planner in ["diffusion_prior", "mpd"]:
        p = ROOT / f"scenes_{planner}" / "metrics.json"
        if p.exists():
            summ[planner] = json.loads(p.read_text(encoding="utf-8"))["summary"]
    if len(summ) == 2:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        xs = np.arange(len(kinds))
        w = 0.36
        a = [summ["diffusion_prior"].get(k, {}).get("single_traj_safe_rate", 0) for k in kinds]
        b = [summ["mpd"].get(k, {}).get("single_traj_safe_rate", 0) for k in kinds]
        ax.bar(xs - w / 2, a, w, label="A: Diffusion Prior", color="#4c72b0")
        ax.bar(xs + w / 2, b, w, label="B: MPD (SDF guide)", color="#dd8452")
        for i, (va, vb) in enumerate(zip(a, b)):
            ax.text(i - w / 2, va + 0.02, f"{va:.2f}", ha="center", fontsize=8)
            ax.text(i + w / 2, vb + 0.02, f"{vb:.2f}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([KIND_CN[k] for k in kinds])
        ax.set_ylabel("Per-trajectory static-safe rate")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title("Official MPD ported to No02 scenes: effect of SDF guidance", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "mpd_scenes_AB_bar.png", dpi=130)
        plt.close(fig)
    print("图已生成：", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
