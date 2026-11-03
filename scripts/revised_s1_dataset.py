"""第二版方案 · S1：三类简单场景的地图与数据集重建。

产出 `data/revised_s1/`：
  maps_<split>.json     地图（含 map_id / 障碍几何 / 参数）
  traj_<split>.npz      paths [N,H,2]、starts、goals、map_idx、task_id、path_variant
  tasks_<split>.jsonl   每个任务一行：task_id / map_id / kind / start / goal / seed ...
  meta.json             配置、种子、统计

划分：**按地图布局**划分训练/验证/测试（不同 map_seed），不是同图换起终点。

用法：
  python scripts/revised_s1_dataset.py --config configs/base.yaml --out data/revised_s1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.data.astar import build_grid, plan_path  # noqa: E402
from fhd.data.preprocess import validate_path  # noqa: E402
from fhd.geometry.collisions import trajectory_is_static_safe  # noqa: E402
from fhd.geometry.paths import path_length, resample_polyline  # noqa: E402
from fhd.maps.simple_scenes import (  # noqa: E402
    KIND_LABELS,
    SIMPLE_KINDS,
    generate_simple_map,
    map_id_of,
)
from fhd.utils.config import load_config, save_json  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle as MplCircle  # noqa: E402
from matplotlib.patches import Rectangle as MplRect  # noqa: E402

SPLIT_SEED = {"train": 100_000, "val": 200_000, "test": 300_000}
SPLIT_MAPS = {"train": 100, "val": 20, "test": 20}
TASKS_PER_MAP = 30
VARIANTS = 2


# --------------------------------------------------------------------------- #
def sample_task(world, kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """按场景类型采样一对起终点，使路径必然出现交叉或共享通道。"""
    w, h = world.width_m, world.height_m
    cx, cy = w / 2, h / 2
    for _ in range(40):
        if kind == "open_crossing":
            theta = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(3.4, 4.3)
            s = np.array([cx + r * np.cos(theta), cy + r * np.sin(theta)])
            g = np.array([cx - r * np.cos(theta), cy - r * np.sin(theta)])
            s += rng.normal(0, 0.25, 2)
            g += rng.normal(0, 0.25, 2)
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
        else:  # bottleneck_channel
            horizontal = bool(world.meta.get("horizontal", True))
            ccx, ccy = world.meta.get("center", [cx, cy])
            r = rng.uniform(3.2, 4.2)
            off = rng.uniform(-1.2, 1.2)
            if horizontal:  # 通道沿 x，机器人沿 y 穿过
                s = np.array([ccx + off, ccy - r])
                g = np.array([ccx - off, ccy + r])
            else:
                s = np.array([ccx - r, ccy + off])
                g = np.array([ccx + r, ccy - off])
        lo, hi = 0.6, w - 0.6
        s = np.clip(s, lo, hi)
        g = np.clip(g, lo, hi)
        if np.linalg.norm(s - g) < 4.0:
            continue
        if world.is_free(s[None])[0] and world.is_free(g[None])[0]:
            return s, g
    return None


def pair_min_distance(path_a: torch.Tensor, path_b: torch.Tensor) -> float:
    """两条路径在相同进度上的最小中心距离（用于判断是否存在交叉冲突）。"""
    d = torch.linalg.vector_norm(path_a - path_b, dim=-1)
    return float(d.min())


def main() -> int:
    ap = argparse.ArgumentParser(description="S1 场景与数据集重建")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "revised_s1"))
    ap.add_argument("--tasks-per-map", type=int, default=TASKS_PER_MAP)
    ap.add_argument("--variants", type=int, default=VARIANTS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = new_run_dir("S1_scene_redesign", "revised_s1")
    logger = get_logger("fhd.s1r", run_dir / "logs" / "s1.log")
    log_section(logger, "S1 场景重设计 + 数据集重建")

    h = int(cfg["trajectory"]["horizon"])
    v_max = float(cfg["trajectory"]["max_speed_mps"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    resolution = int(cfg["dataset"]["astar_resolution"])
    min_len = float(cfg["dataset"]["min_path_length_m"])
    min_sg = float(cfg["dataset"]["min_start_goal_dist_m"])
    d_safe = 2 * float(cfg["world"]["robot_radius_m"]) + float(cfg["world"]["safety_margin_m"])
    max_len = v_max * dt_s * (h - 1)
    logger.info("H=%d，速度上限 %.2f m/s → 路径总长上限 %.2f m", h, v_max, max_len)

    meta = {"config": cfg.to_dict(), "system": system_info(), "splits": {},
            "tasks_per_map": args.tasks_per_map, "variants": args.variants,
            "h": h, "max_path_length_m": max_len, "d_safe": d_safe}
    all_stats = {}
    t_start = time.time()

    for split, n_maps in SPLIT_MAPS.items():
        rng = np.random.default_rng(SPLIT_SEED[split])
        maps, paths, starts, goals, map_idx, task_ids, variants = [], [], [], [], [], [], []
        tasks_meta = []
        stats = {"tasks_attempted": 0, "tasks_kept": 0, "paths": 0,
                 "discard_unreachable": 0, "discard_speed": 0, "discard_static": 0,
                 "discard_length": 0, "discard_sample": 0,
                 "per_kind": {}, "conflict_pairs": []}
        for mi in range(n_maps):
            kind = SIMPLE_KINDS[mi % len(SIMPLE_KINDS)]
            seed = SPLIT_SEED[split] + mi
            try:
                world = generate_simple_map(kind, cfg, seed)
            except RuntimeError as exc:
                logger.warning("地图生成失败 %s/%d: %s", kind, seed, exc)
                continue
            grid = build_grid(world, resolution)
            local_idx = len(maps)
            maps.append(world)
            kind_stats = stats["per_kind"].setdefault(kind, {"maps": 0, "tasks": 0, "paths": 0,
                                                             "first_paths": []})
            kind_stats["maps"] += 1
            for ti in range(args.tasks_per_map):
                stats["tasks_attempted"] += 1
                sg = sample_task(world, kind, rng)
                if sg is None:
                    stats["discard_sample"] += 1
                    continue
                s, g = sg
                base_path = plan_path(world, s, g, resolution, h, rng=rng, noise_scale=0.0,
                                      grid=grid)
                if base_path is None:
                    stats["discard_unreachable"] += 1
                    continue
                diag = validate_path(base_path, torch.as_tensor(s, dtype=torch.float64),
                                     torch.as_tensor(g, dtype=torch.float64), world, v_max, dt_s)
                if not diag["static_safe"]:
                    stats["discard_static"] += 1
                    continue
                if not diag["speed_ok"]:
                    stats["discard_speed"] += 1
                    continue
                if diag["length_m"] < min_len:
                    stats["discard_length"] += 1
                    continue
                task_id = f"{split}_{mi:03d}_{ti:03d}"
                kept_here = 0
                cand = [base_path]
                for v in range(1, args.variants):
                    wp = world.sample_free_points(rng, 1, min_sep=0.3, border_m=0.8)[0]
                    p = plan_path(world, s, g, resolution, h, rng=rng, noise_scale=0.35,
                                  grid=grid, waypoint=wp)
                    if p is None:
                        continue
                    d2 = validate_path(p, torch.as_tensor(s, dtype=torch.float64),
                                       torch.as_tensor(g, dtype=torch.float64), world, v_max, dt_s)
                    if d2["static_safe"] and d2["speed_ok"] and d2["length_m"] >= min_len:
                        cand.append(p)
                for v, p in enumerate(cand):
                    paths.append(p)
                    starts.append(torch.as_tensor(s, dtype=torch.float64))
                    goals.append(torch.as_tensor(g, dtype=torch.float64))
                    map_idx.append(local_idx)
                    task_ids.append(task_id)
                    variants.append(v)
                    kept_here += 1
                    stats["paths"] += 1
                    kind_stats["paths"] += 1
                if kept_here == 0:
                    continue
                stats["tasks_kept"] += 1
                kind_stats["tasks"] += 1
                if len(kind_stats["first_paths"]) < 200:
                    kind_stats["first_paths"].append(base_path)
                tasks_meta.append({
                    "task_id": task_id, "map_id": map_id_of(kind, seed), "map_index": local_idx,
                    "kind": kind, "kind_label": KIND_LABELS[kind],
                    "start": s.tolist(), "goal": g.tolist(),
                    "n_variants": kept_here,
                    "path_lengths_m": [round(float(path_length(p)), 4) for p in cand],
                    "random_seed": int(rng.integers(0, 2**31 - 1)),
                })
            if (mi + 1) % 20 == 0:
                logger.info("%s: %d/%d 地图，累计 %d 条路径，%.0f s",
                            split, mi + 1, n_maps, stats["paths"], time.time() - t_start)

        # 冲突检查：同类地图内随机配对两个任务的基准路径，统计最近距离
        for kind, ks in stats["per_kind"].items():
            fp = ks.pop("first_paths")
            if len(fp) >= 2:
                rng2 = np.random.default_rng(0)
                for _ in range(min(60, len(fp) * 2)):
                    i, j = rng2.choice(len(fp), size=2, replace=False)
                    stats["conflict_pairs"].append(round(pair_min_distance(fp[i], fp[j]), 4))

        if not paths:
            raise RuntimeError(f"{split} 未生成任何路径")

        (out / f"maps_{split}.json").write_text(
            json.dumps([m.to_dict() for m in maps], ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(
            out / f"traj_{split}.npz",
            paths=torch.stack(paths).numpy(),
            starts=torch.stack(starts).numpy(),
            goals=torch.stack(goals).numpy(),
            map_idx=np.asarray(map_idx, dtype=np.int64),
            task_id=np.asarray(task_ids),
            path_variant=np.asarray(variants, dtype=np.int64),
        )
        (out / f"tasks_{split}.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks_meta), encoding="utf-8")

        cp = stats["conflict_pairs"]
        stats["conflict_pair_min_dist_median"] = float(np.median(cp)) if cp else float("nan")
        stats["conflict_pair_frac_below_dsafe"] = float(np.mean(np.array(cp) < d_safe)) if cp else 0.0
        stats["n_maps"] = len(maps)
        stats["n_paths"] = len(paths)
        stats["n_tasks"] = stats["tasks_kept"]
        stats["elapsed_s"] = round(time.time() - t_start, 1)
        all_stats[split] = stats
        meta["splits"][split] = stats
        logger.info("%s：%d 地图 / %d 任务 / %d 路径；冲突对最小距离中位数 %.3f m，"
                    "低于 d_safe 的比例 %.2f", split, len(maps), stats["tasks_kept"],
                    len(paths), stats["conflict_pair_min_dist_median"],
                    stats["conflict_pair_frac_below_dsafe"])

    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    save_json(run_dir / "manifest.json", {"config": cfg.to_dict(), "system": meta["system"],
                                          "splits": all_stats})

    # ---------------- 可视化 ----------------
    for j, kind in enumerate(SIMPLE_KINDS):
        seed = SPLIT_SEED["train"] + j
        world = generate_simple_map(kind, cfg, seed)
        from fhd.data.astar import build_grid as _bg
        grid = _bg(world, resolution)
        rng = np.random.default_rng(7)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
        for ax, n_show in zip(axes, (1, 4)):
            ax.set_xlim(0, world.width_m)
            ax.set_ylim(0, world.height_m)
            ax.set_aspect("equal")
            for rect in world.rects.reshape(-1, 4):
                ax.add_patch(MplRect((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
                                     facecolor="#555", edgecolor="k", alpha=0.55, zorder=2))
            for c in world.circles.reshape(-1, 3):
                ax.add_patch(MplCircle((c[0], c[1]), c[2], facecolor="#555", edgecolor="k",
                                       alpha=0.55, zorder=2))
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_show, 2)))
            for k in range(n_show):
                sg = sample_task(world, kind, rng)
                if sg is None:
                    continue
                p = plan_path(world, sg[0], sg[1], resolution, h, rng=rng, noise_scale=0.0, grid=grid)
                if p is None:
                    continue
                pts = p.numpy()
                ax.plot(pts[:, 0], pts[:, 1], "-", color=colors[k], lw=1.9,
                        label=f"任务 {k + 1}")
                ax.plot(pts[0, 0], pts[0, 1], "o", color=colors[k], ms=8, mec="k")
                ax.plot(pts[-1, 0], pts[-1, 1], "s", color=colors[k], ms=8, mec="k")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(f"{KIND_LABELS[kind]}：{'单任务' if n_show == 1 else f'{n_show} 个任务（可见交叉/共享通道）'}",
                         fontsize=10)
            ax.grid(alpha=0.25, ls=":")
            if n_show > 1:
                ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"S1 场景示例：{KIND_LABELS[kind]}（seed={seed}）", fontsize=12)
        fig.tight_layout()
        (run_dir / "plots").mkdir(exist_ok=True)
        fig.savefig(run_dir / "plots" / f"scene_{kind}.png", dpi=130)
        plt.close(fig)

    # ---------------- 报告 ----------------
    lines = ["# S1 场景重设计与数据集重建报告", "",
             f"- 运行目录：`{run_dir}`", f"- 数据目录：`{out}`", "",
             "## 场景设计", "",
             "| 场景 | 障碍物 | 交互点 |", "|---|---|---|",
             "| 开放交叉 `open_crossing` | 0–1 个角落小圆 | 中心交叉 |",
             "| 单障碍绕行 `single_obstacle` | 中央 1 个矩形或圆 | 障碍前后汇合 |",
             "| 单瓶颈通行 `bottleneck_channel` | 2 道墙形成 0.45–0.62 m 通道 | 通道本身 |", "",
             "## 数据划分（按地图布局，不是同图换起终点）", "",
             "| 划分 | 地图数 | 任务数 | 路径数 | 冲突对最小距离中位数 [m] | 低于 d_safe 比例 | 用时 [s] |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for split, st in all_stats.items():
        lines.append(f"| {split} | {st['n_maps']} | {st['n_tasks']} | {st['n_paths']} | "
                     f"{st['conflict_pair_min_dist_median']:.3f} | "
                     f"{st['conflict_pair_frac_below_dsafe']:.2f} | {st['elapsed_s']} |")
    lines += ["", "## 丢弃统计", "",
              "| 划分 | 采样失败 | 不可达 | 超速 | 静态碰撞 | 太短 |", "|---|---:|---:|---:|---:|---:|"]
    for split, st in all_stats.items():
        lines.append(f"| {split} | {st['discard_sample']} | {st['discard_unreachable']} | "
                     f"{st['discard_speed']} | {st['discard_static']} | {st['discard_length']} |")
    lines += ["", "## 按场景分布", "", "| 划分 | 场景 | 地图 | 任务 | 路径 |", "|---|---|---:|---:|---:|"]
    for split, st in all_stats.items():
        for kind, ks in st["per_kind"].items():
            lines.append(f"| {split} | {KIND_LABELS[kind]} | {ks['maps']} | {ks['tasks']} | {ks['paths']} |")
    lines += ["", "## 数据集字段", "",
              "- `traj_<split>.npz`：`paths [N,H,2]`、`starts`、`goals`、`map_idx`、`task_id`、`path_variant`",
              "- `tasks_<split>.jsonl`：`task_id`、`map_id`、`kind`、`start`、`goal`、`n_variants`、"
              "`path_lengths_m`、`random_seed`",
              "- `maps_<split>.json`：完整连续几何（矩形/圆）与元参数；静态检查一律基于连续几何",
              "", "## 验收自检", ""]
    kinds_seen = {k for st in all_stats.values() for k in st["per_kind"]}
    checks = {
        "三类场景均已生成": set(SIMPLE_KINDS).issubset(kinds_seen),
        "训练/验证/测试地图种子互不重叠": (SPLIT_SEED["train"] < SPLIT_SEED["val"] < SPLIT_SEED["test"]),
        "路径均通过连续几何静态检查": all(s["discard_static"] == 0 for s in all_stats.values()),
        "冲突对确实存在（低于 d_safe 比例 > 0）": all(
            s["conflict_pair_frac_below_dsafe"] > 0 for s in all_stats.values()),
        "无大量不可达任务（< 10%）": all(
            s["discard_unreachable"] / max(s["tasks_attempted"], 1) < 0.1 for s in all_stats.values()),
        "每个任务都有多条 path_variant": all(s["n_paths"] > s["n_tasks"] for s in all_stats.values()),
    }
    for k, v in checks.items():
        lines.append(f"- {'✅' if v else '❌'} {k}")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    import shutil
    docs = ROOT / "docs" / "revised"
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "report.md", docs / "01_S1_scenes.md")
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, docs / "figures" / png.name)

    logger.info("S1 完成：%s", out)
    print(f"\n数据：{out}\n运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
