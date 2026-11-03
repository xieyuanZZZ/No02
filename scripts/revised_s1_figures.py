"""重绘 S1 三类场景示意图（中文字体、含多任务交叉演示）。

单独成脚本的原因：数据集生成很慢，但改图只需要秒级重跑。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle as MplCircle  # noqa: E402
from matplotlib.patches import Rectangle as MplRect  # noqa: E402

from fhd.data.astar import build_grid, plan_path  # noqa: E402
from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402
from fhd.maps.generate import WorldMap  # noqa: E402
from fhd.maps.simple_scenes import KIND_LABELS, SIMPLE_KINDS  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402

_setup_fonts()


def sample_task_demo(world, kind: str, rng) -> tuple[np.ndarray, np.ndarray] | None:
    w, h = world.width_m, world.height_m
    cx, cy = w / 2, h / 2
    for _ in range(60):
        if kind == "open_crossing":
            theta = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(3.4, 4.3)
            s = np.array([cx + r * np.cos(theta), cy + r * np.sin(theta)])
            g = np.array([cx - r * np.cos(theta), cy - r * np.sin(theta)])
        elif kind == "single_obstacle":
            axis = rng.integers(0, 2)
            r = rng.uniform(3.2, 4.2)
            off = rng.uniform(-1.6, 1.6)
            if axis == 0:
                s = np.array([cx - r, cy + off])
                g = np.array([cx + r, cy - off])
            else:
                s = np.array([cx + off, cy - r])
                g = np.array([cx - off, cy + r])
        else:
            horizontal = bool(world.meta.get("horizontal", True))
            ccx, ccy = world.meta.get("center", [cx, cy])
            r = rng.uniform(3.2, 4.2)
            off = rng.uniform(-1.2, 1.2)
            if horizontal:
                s = np.array([ccx + off, ccy - r])
                g = np.array([ccx - off, ccy + r])
            else:
                s = np.array([ccx - r, ccy + off])
                g = np.array([ccx + r, ccy - off])
        s = np.clip(s, 0.6, w - 0.6)
        g = np.clip(g, 0.6, w - 0.6)
        if np.linalg.norm(s - g) < 4.0:
            continue
        if world.is_free(s[None])[0] and world.is_free(g[None])[0]:
            return s, g
    return None


def main() -> int:
    cfg = load_config(ROOT / "configs" / "base.yaml")
    data = ROOT / "data" / "revised_s1"
    maps = {}
    for split in ("train", "val", "test"):
        maps[split] = []
        for m in json.loads((data / f"maps_{split}.json").read_text(encoding="utf-8")):
            wm = WorldMap(**m)
            wm.rects = np.asarray(wm.rects, dtype=np.float64).reshape(-1, 4)
            wm.circles = np.asarray(wm.circles, dtype=np.float64).reshape(-1, 3)
            maps[split].append(wm)
    run_dir = new_run_dir("S1_scene_redesign", "s1_figures")
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    color = plt.cm.tab10(np.linspace(0, 1, 10))
    for kind in SIMPLE_KINDS:
        # 找一个该类型、冲突最明显的测试地图
        cand = [m for m in maps["test"] if m.kind == kind]
        if not cand:
            continue
        world = cand[0]
        grid = build_grid(world, 256)
        rng = np.random.default_rng(3)
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
        for ax, n_show, title in zip(
                axes, (1, 2, 4),
                ("单任务", "2 台机器人：路径交叉/共享通道", "4 台机器人：多向冲突")):
            ax.set_xlim(0, world.width_m)
            ax.set_ylim(0, world.height_m)
            ax.set_aspect("equal")
            for rect in world.rects.reshape(-1, 4):
                ax.add_patch(MplRect((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
                                     facecolor="#555", edgecolor="k", alpha=0.55, zorder=2))
            for c in world.circles.reshape(-1, 3):
                ax.add_patch(MplCircle((c[0], c[1]), c[2], facecolor="#555", edgecolor="k",
                                       alpha=0.55, zorder=2))
            shown = 0
            while shown < n_show:
                sg = sample_task_demo(world, kind, rng)
                if sg is None:
                    break
                p = plan_path(world, sg[0], sg[1], 256, 32, rng=rng, noise_scale=0.0, grid=grid)
                if p is None:
                    continue
                pts = p.numpy()
                ax.plot(pts[:, 0], pts[:, 1], "-", color=color[shown], lw=2.0,
                        label=f"机器人 {shown + 1}")
                ax.plot(pts[0, 0], pts[0, 1], "o", color=color[shown], ms=9, mec="k", zorder=5)
                ax.plot(pts[-1, 0], pts[-1, 1], "s", color=color[shown], ms=9, mec="k", zorder=5)
                shown += 1
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.25, ls=":")
            if shown > 1:
                ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"S1 场景：{KIND_LABELS[kind]}（{kind}）", fontsize=13)
        fig.tight_layout()
        fig.savefig(run_dir / "plots" / f"scene_{kind}.png", dpi=140)
        plt.close(fig)

    import shutil
    docs = ROOT / "docs" / "revised" / "figures"
    docs.mkdir(parents=True, exist_ok=True)
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, docs / png.name)
    print(f"图已更新：{run_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
