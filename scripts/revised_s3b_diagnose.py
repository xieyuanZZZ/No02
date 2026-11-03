"""S3 补充：为什么「单障碍绕行」只有 29.5%？

只做诊断，不改模型。三项证据：
  1. 生成路径与参考路径的「绕行比」（长度 / 起终点直线距离）对比；
  2. 生成路径距障碍物中心的最小距离，与安全阈值对比；
  3. 典型任务的可视化：参考路径绕行，生成路径直接穿过障碍。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.data.dataset import TrajectoryDataset  # noqa: E402
from fhd.diffusion.ddim import ddim_sample  # noqa: E402
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.diffusion.train import build_model  # noqa: E402
from fhd.evaluation.simple_eval import pick_tasks  # noqa: E402
from fhd.evaluation.visualizations import AGENT_COLORS, _setup_fonts, draw_map  # noqa: E402
from fhd.geometry.collisions import trajectory_is_static_safe  # noqa: E402
from fhd.geometry.paths import path_length  # noqa: E402
from fhd.maps.simple_scenes import KIND_LABELS  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import resolve_device, set_seed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_setup_fonts()


def straight_ratio(path: torch.Tensor) -> float:
    """路径长度 / 起终点直线距离。=1 表示完全笔直。"""
    length = float(path_length(path.unsqueeze(0))[0])
    d = float(torch.linalg.vector_norm(path[-1] - path[0]))
    return length / max(d, 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "s2_B_128_32.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "revised_s1"))
    ap.add_argument("--per-kind", type=int, default=25)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device("auto")
    run_dir = new_run_dir("S3_single_agent", "s3_diagnose")

    model = build_model(cfg, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                            kind=str(cfg["diffusion"]["beta_schedule"]), device=device)
    res = int(cfg["diffusion"].get("map_input_res", cfg["world"]["grid_size"]))
    ds = TrajectoryDataset(Path(args.data), "test", grid_size=res,
                           robot_radius_m=float(cfg["world"]["robot_radius_m"]),
                           clearance_margin_m=float(cfg["world"]["clearance_margin_m"]),
                           width_m=float(cfg["world"]["width_m"]),
                           height_m=float(cfg["world"]["height_m"]))
    tasks = pick_tasks(ds, args.per_kind)

    rows = []
    examples = {}
    gen = torch.Generator(device=device.type).manual_seed(0)
    for t in tasks:
        idx = t["task"]
        item = ds[idx]
        n = args.samples
        start = item["start"].unsqueeze(0).expand(n, -1).to(device)
        goal = item["goal"].unsqueeze(0).expand(n, -1).to(device)
        grid = item["grid"].unsqueeze(0).expand(n, -1, -1, -1).to(device)
        with torch.no_grad():
            out = ddim_sample(model, schedule, start, goal, grid,
                              num_steps=int(cfg["diffusion"]["ddim_sampling_steps"]),
                              generator=gen)
        paths = ds.denormalise(out.float()).cpu()
        world = ds.maps[int(item["map_idx"])]
        safe, clr = trajectory_is_static_safe(paths, world.obstacles(), world.robot_radius_m,
                                              world.clearance_margin_m)
        ref = ds.data["paths"][idx]
        rows.append({
            "task": idx, "kind": t["kind"],
            "ref_straight_ratio": straight_ratio(ref),
            "gen_straight_ratio_mean": float(np.mean([straight_ratio(p) for p in paths])),
            "static_safe_rate": float(safe.float().mean()),
            "min_clearance_mean": float(clr.min()),
            "ref_min_clearance": float(trajectory_is_static_safe(
                ref.unsqueeze(0), world.obstacles(), world.robot_radius_m,
                world.clearance_margin_m)[1].min()),
        })
        if t["kind"] not in examples:
            examples[t["kind"]] = (idx, paths, ref, world)

    # 汇总
    summary = {}
    for kind in KIND_LABELS:
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        summary[kind] = {
            "n_tasks": len(sel),
            "ref_straight_ratio": float(np.mean([r["ref_straight_ratio"] for r in sel])),
            "gen_straight_ratio": float(np.mean([r["gen_straight_ratio_mean"] for r in sel])),
            "static_safe_rate": float(np.mean([r["static_safe_rate"] for r in sel])),
        }

    # 图
    kinds = [k for k in KIND_LABELS if k in examples]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.6 * len(kinds), 5.2))
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        idx, paths, ref, world = examples[kind]
        draw_map(ax, world)
        ax.plot(ref[:, 0].numpy(), ref[:, 1].numpy(), "k--", lw=1.6, label="参考路径")
        for i in range(min(8, paths.shape[0])):
            ax.plot(paths[i, :, 0], paths[i, :, 1], color=AGENT_COLORS[i % len(AGENT_COLORS)],
                    lw=1.0, alpha=0.75)
        ax.set_title(f"{KIND_LABELS[kind]}\n生成(彩) vs 参考(黑虚)", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("S3 失败模式：模型只会生成近直线，不会绕障", fontsize=12)
    fig.tight_layout()
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "plots" / "s3_failure_mode.png", dpi=130)
    plt.close(fig)

    lines = ["# S3 补充诊断：单障碍绕行为何失败", "",
             "## 证据", "",
             "| 场景 | 任务数 | 参考路径绕行比 | 生成路径绕行比 | 静态安全率 |",
             "|---|---:|---:|---:|---:|"]
    for kind, v in summary.items():
        lines.append(f"| {KIND_LABELS[kind]} | {v['n_tasks']} | {v['ref_straight_ratio']:.3f} | "
                     f"{v['gen_straight_ratio']:.3f} | {v['static_safe_rate']:.3f} |")
    lines += ["", "绕行比 = 路径长度 / 起终点直线距离；1.000 表示完全笔直。", "",
              "## 解读", "",
              f"- 参考路径在单障碍场景的绕行比为 "
              f"{summary.get('single_obstacle', {}).get('ref_straight_ratio', float('nan')):.3f}，"
              f"而生成路径只有 "
              f"{summary.get('single_obstacle', {}).get('gen_straight_ratio', float('nan')):.3f}"
              "，说明模型几乎总是输出近直线，直接穿过障碍。",
              f"- 交叉场景参考绕行比 "
              f"{summary.get('open_crossing', {}).get('ref_straight_ratio', float('nan')):.3f}，"
              "本身就是直线，所以模型表现好；"
              f"瓶颈场景参考绕行比 "
              f"{summary.get('bottleneck_channel', {}).get('ref_straight_ratio', float('nan')):.3f}，"
              "虽然要过门但整体仍接近直线。",
              "- 结论：**模型学到的是「起点到终点的准直线插值」，没有学到「绕开障碍」**。",
              "  这是表示/训练问题，不是地图分辨率问题（S2 已证明三种分辨率无差异）。"]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    import shutil
    docs = ROOT / "docs" / "revised"
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "report.md", docs / "02c_S3_failure_mode.md")
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, docs / "figures" / png.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
