"""生成 M0/M1/M2 对照图（供阶段报告使用）。

输入：results/Online_FirstHit/04_dynamic_firsthit/m1m2/m1_pairs.csv
输出：同目录 figures/m0m1m2_comparison.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402

_setup_fonts()

RUN = ROOT / "results" / "Online_FirstHit" / "04_dynamic_firsthit" / "m1m2"


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main() -> int:
    rows = list(csv.DictReader((RUN / "m1_pairs.csv").open(encoding="utf-8")))
    obs = np.array([float(r["obs"]) for r in rows])
    models = [("m0", "M0 原始 First-Hit"), ("m1", "M1 均值段时长"),
              ("m2", "M2 分布采样")]
    groups = [("N2", "N=2"), ("N4", "N=4")]

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    for col, (key, name) in enumerate(models):
        pred = np.array([float(r[key]) for r in rows])
        ax = axes[0, col]
        ax.scatter(pred, obs, s=16, alpha=0.7)
        lim = [0, max(0.8, float(max(pred.max(), obs.max())) * 1.05)]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{name} 预测")
        if col == 0:
            ax.set_ylabel("MPC MC 两两碰撞频率")
        ax.set_title(f"{name}\nMAE={np.mean(np.abs(pred - obs)):.3f}  "
                     f"ρ={spearman(pred, obs):.3f}", fontsize=10)
        ax.grid(alpha=0.3)

    # 第四列：分组 MAE 柱状
    ax = axes[0, 3]
    width = 0.26
    xs = np.arange(len(groups))
    for mi, (key, name) in enumerate(models):
        vals = []
        for g, _ in groups:
            sel = [r for r in rows if r["group"] == g]
            vals.append(float(np.mean([abs(float(r[key]) - float(r["obs"])) for r in sel])))
        ax.bar(xs + (mi - 1) * width, vals, width, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_ylabel("MAE")
    ax.set_title("分组 MAE", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # 第二行：按 p 分组的偏差与排序相关
    ps = sorted({r["p"] for r in rows}, key=float)
    ax = axes[1, 0]
    for key, name in models:
        bias = []
        for p in ps:
            sel = [r for r in rows if r["p"] == p]
            bias.append(float(np.mean([float(r[key]) - float(r["obs"]) for r in sel])))
        ax.plot([float(p) for p in ps], bias, "o-", label=name)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("暂停概率 p")
    ax.set_ylabel("偏差（预测 − 实测）")
    ax.set_title("偏差随 p 的变化", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for key, name in models:
        rho = []
        for p in ps:
            sel = [r for r in rows if r["p"] == p]
            rho.append(spearman([float(r[key]) for r in sel], [float(r["obs"]) for r in sel]))
        ax.plot([float(p) for p in ps], rho, "o-", label=name)
    ax.set_xlabel("暂停概率 p")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("风险排序相关性", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 风险分布对比
    ax = axes[1, 2]
    bins = np.linspace(0, 0.8, 25)
    ax.hist(obs, bins=bins, alpha=0.45, label="MPC MC 实测", color="k")
    for key, name in models:
        ax.hist([float(r[key]) for r in rows], bins=bins, histtype="step", lw=1.4, label=name)
    ax.set_xlabel("两两碰撞概率")
    ax.set_ylabel("机器人对数量")
    ax.set_title("预测与实测的分布", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 汇总表
    ax = axes[1, 3]
    ax.axis("off")
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    lines = ["model        MAE     bias      rho"]
    for key, label in (("M0_vs_MPC", "M0"), ("M1_vs_MPC", "M1"), ("M2_vs_MPC", "M2")):
        v = summary[key]
        lines.append(f"{label:<6} {v['mae']:>10.4f} {v['bias']:>+7.4f} {v['spearman']:>6.3f}")
    ax.text(0.02, 0.75, "\n".join(lines), family="monospace", fontsize=11,
            transform=ax.transAxes, va="top")
    ax.text(0.02, 0.30, "grouped MAE\n"
                        "N=2 : M0 0.058  M1 0.053  M2 0.112\n"
                        "N=4 : M0 0.135  M1 0.174  M2 0.119",
            family="monospace", fontsize=9, transform=ax.transAxes, va="top")

    fig.suptitle("M0/M1/M2 风险预测与 MPC 执行结果的对照（112 个机器人对）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RUN / "figures" / "m0m1m2_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"输出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
