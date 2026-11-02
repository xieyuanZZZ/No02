"""S3 前置：从零生成单机器人路径数据集（A* + 多样化扰动 + 连续几何复核）。

生成规则（文档 §6.2）：
  1. 地图生成器创建矩形/圆形障碍（按机器人半径与净空膨胀）；
  2. 在自由空间采样起点/终点，剔除不可达或过近任务；
  3. A* 与带随机边代价的加权 A* 生成多样化路径，并随机插入途经点制造同伦分支；
  4. 路径按弧长重采样到 H 个路点，逐段检查速度上限；
  5. 任何一条路径都要通过**连续几何**碰撞复核；
  6. 训练 / 验证 / 测试使用互不重叠的地图种子。

用法：
  python scripts/generate_dataset.py --config configs/diffusion_train.yaml --split-size small
  python scripts/generate_dataset.py --config configs/diffusion_train.yaml --split-size full
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
from fhd.data.dataset import dataset_stats, save_split  # noqa: E402
from fhd.data.preprocess import diversity_score, validate_path  # noqa: E402
from fhd.maps.generate import MAP_KINDS, generate_map  # noqa: E402
from fhd.utils.config import load_config, save_json  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

SPLIT_SEED_OFFSET = {"train": 0, "val": 10_000, "test": 20_000}


def parse_args():
    p = argparse.ArgumentParser(description="生成单机器人轨迹数据集")
    p.add_argument("--config", default=str(ROOT / "configs" / "diffusion_train.yaml"))
    p.add_argument("--split-size", choices=["smoke", "small", "medium", "full"], default="medium")
    p.add_argument("--vectors-per-task", type=int, default=2, help="每个任务生成多少条不同路径")
    p.add_argument("--out", default=str(ROOT / "data" / "dataset"))
    return p.parse_args()


def generate_split(
    split: str,
    n_maps: int,
    tasks_per_map: int,
    cfg,
    rng_seed: int,
    vectors_per_task: int,
    logger,
) -> dict:
    h = int(cfg["trajectory"]["horizon"])
    v_max = float(cfg["trajectory"]["max_speed_mps"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    resolution = int(cfg["dataset"]["astar_resolution"])
    min_len = float(cfg["dataset"]["min_path_length_m"])
    min_sg = float(cfg["dataset"]["min_start_goal_dist_m"])
    noise_scale = 0.35

    maps, paths, starts, goals, map_idx = [], [], [], [], []
    stats = {
        "tasks_attempted": 0, "tasks_reachable": 0, "vectors_generated": 0,
        "discarded_unreachable": 0, "discarded_too_close": 0,
        "discarded_speed": 0, "discarded_static": 0, "discarded_length": 0,
        "diversity_samples": [],
    }
    t0 = time.time()
    rng = np.random.default_rng(rng_seed)

    for mi in range(n_maps):
        kind = MAP_KINDS[mi % len(MAP_KINDS)]
        map_seed = rng_seed + mi
        try:
            world = generate_map(kind, cfg, map_seed)
        except RuntimeError:
            continue
        grid = build_grid(world, resolution)
        maps.append(world)
        local_idx = len(maps) - 1
        made_for_map = 0
        attempts = 0
        while made_for_map < tasks_per_map and attempts < tasks_per_map * 6:
            attempts += 1
            stats["tasks_attempted"] += 1
            try:
                pair = world.sample_free_points(rng, 2, min_sep=min_sg, border_m=0.6)
            except RuntimeError:
                continue
            s, g = pair[0], pair[1]
            if float(np.linalg.norm(s - g)) < min_sg:
                stats["discarded_too_close"] += 1
                continue
            cand = []
            for k in range(vectors_per_task):
                if k == 0:
                    path = plan_path(world, s, g, resolution, h, rng=rng, noise_scale=0.0, grid=grid)
                else:
                    wp = None
                    if rng.random() < 0.6:
                        try:
                            wp = world.sample_free_points(rng, 1, min_sep=0.3, border_m=0.6)[0]
                        except RuntimeError:
                            wp = None
                    path = plan_path(world, s, g, resolution, h, rng=rng,
                                     noise_scale=noise_scale, grid=grid, waypoint=wp)
                if path is None:
                    stats["discarded_unreachable"] += 1
                    continue
                diag = validate_path(path, torch.as_tensor(s, dtype=torch.float64),
                                     torch.as_tensor(g, dtype=torch.float64), world, v_max, dt_s)
                if not diag["speed_ok"]:
                    stats["discarded_speed"] += 1
                    continue
                if not diag["static_safe"]:
                    stats["discarded_static"] += 1
                    continue
                if diag["length_m"] < min_len:
                    stats["discarded_length"] += 1
                    continue
                cand.append(path)
            if not cand:
                continue
            for p in cand:
                paths.append(p)
                starts.append(torch.as_tensor(s, dtype=torch.float64))
                goals.append(torch.as_tensor(g, dtype=torch.float64))
                map_idx.append(local_idx)
                stats["vectors_generated"] += 1
            if len(cand) > 1:
                stats["diversity_samples"].append(diversity_score(torch.stack(cand)))
            made_for_map += 1
            stats["tasks_reachable"] += 1
        if (mi + 1) % 20 == 0:
            logger.info("%s: %d/%d 地图，累计 %d 条路径，用时 %.0f s",
                        split, mi + 1, n_maps, len(paths), time.time() - t0)

    div = stats.pop("diversity_samples")
    stats["diversity_mean_m"] = float(np.mean(div)) if div else 0.0
    stats["n_maps_used"] = len(maps)
    stats["n_paths"] = len(paths)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return {
        "maps": maps, "paths": paths, "starts": starts, "goals": goals,
        "map_idx": map_idx, "stats": stats,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    out_root = Path(args.out)
    run_dir = new_run_dir("S3_diffusion_training", f"S3_dataset_{args.split_size}")
    logger = get_logger("fhd.s3.data", run_dir / "logs" / "dataset.log")
    log_section(logger, f"S3 数据集生成（规模={args.split_size}）")

    if args.split_size == "smoke":
        sizes = {"train": (5, 25), "val": (5, 10), "test": (6, 10)}
    elif args.split_size == "small":
        sizes = {"train": (10, 40), "val": (5, 30), "test": (5, 30)}
    elif args.split_size == "medium":
        # 批复 §25：先用 5,000 / 500 数据完成流程验证，再扩大到 20,000 / 2,000
        sizes = {"train": (50, 50), "val": (10, 25), "test": (10, 25)}
    else:
        sizes = {"train": (int(cfg["dataset"]["train_maps"]), int(cfg["dataset"]["train_tasks_per_map"])),
                 "val": (int(cfg["dataset"]["val_maps"]), int(cfg["dataset"]["val_tasks_per_map"])),
                 "test": (int(cfg["dataset"]["test_maps"]), int(cfg["dataset"]["test_tasks_per_map"]))}
    logger.info("规模设置：%s", sizes)

    meta = {"config": cfg.to_dict(), "system": system_info(), "sizes": sizes, "splits": {}}
    all_stats = {}
    for split, (n_maps, tasks) in sizes.items():
        rng_seed = int(cfg["seed"]) + SPLIT_SEED_OFFSET[split]
        result = generate_split(split, n_maps, tasks, cfg, rng_seed,
                                args.vectors_per_task, logger)
        if not result["paths"]:
            raise RuntimeError(f"{split} 未生成任何路径")
        save_split(out_root, split, result["maps"], result["paths"], result["starts"],
                   result["goals"], result["map_idx"])
        all_stats[split] = result["stats"]
        meta["splits"][split] = result["stats"]
        logger.info("%s：%d 张地图，%d 条路径，多样性 %.3f m，用时 %.0f s",
                    split, result["stats"]["n_maps_used"], result["stats"]["n_paths"],
                    result["stats"]["diversity_mean_m"], result["stats"]["elapsed_s"])
        print(f"{split}: {result['stats']}")

    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    save_json(run_dir / "dataset_stats.json", all_stats)
    (run_dir / "config_snapshot.yaml").parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(run_dir / "config_snapshot.yaml")

    # 统计与报告
    report_lines = ["# 数据集生成报告", "", f"- 规模：{args.split_size}", f"- 输出目录：`{out_root}`", ""]
    for split in sizes:
        st = all_stats[split]
        report_lines += [f"## {split}", "", "```json", json.dumps(st, ensure_ascii=False, indent=2), "```", ""]
        disc = st["discarded_speed"] + st["discarded_static"] + st["discarded_unreachable"]
        tot = max(1, st["tasks_attempted"])
        report_lines.append(f"- 丢弃率：{disc / tot:.2%}（不可达 {st['discarded_unreachable']}，"
                            f"超速 {st['discarded_speed']}，静态碰撞 {st['discarded_static']}）")
        report_lines.append("")
    report_lines += ["## 结论", "", "数据已按训练/验证/测试互不重叠的地图种子生成，"
                      "全部路径通过连续几何碰撞与逐段速度上限复核。", ""]
    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    (run_dir / "stage_status.json").write_text(json.dumps({
        "stage": "S3_dataset", "status": "completed", "split_size": args.split_size,
        "stats": all_stats, "git_commit": meta["system"]["git_commit"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("数据集生成完成：%s", out_root)
    print(f"\n数据集：{out_root}\n运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
