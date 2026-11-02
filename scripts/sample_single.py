"""S3：单机器人条件 Diffusion 的生成质量评估（文档 §6.5）。

统计 500 个独立测试任务（可调）：
  * 静态障碍连续碰撞率与成功率；
  * 起终点误差、路径长度、平滑度；
  * 每段速度约束违反率；
  * 同一任务多个候选的几何多样性；
  * 推理时间。

用法：python scripts/sample_single.py --config configs/diffusion_train.yaml --ckpt <best.pt>
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

from fhd.data.dataset import TrajectoryDataset  # noqa: E402
from fhd.data.preprocess import diversity_score  # noqa: E402
from fhd.diffusion.ddim import ddim_sample  # noqa: E402
from fhd.diffusion.model import ConditionalUNet1D  # noqa: E402
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.evaluation.reporting import StageReport, update_project_progress  # noqa: E402
from fhd.evaluation.visualizations import draw_map  # noqa: E402
from fhd.geometry.collisions import check_speed_limit, path_segment_lengths, trajectory_is_static_safe  # noqa: E402
from fhd.geometry.paths import path_length  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import resolve_device, set_seed  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def load_model(cfg, ckpt_path: Path, device):
    d = cfg["diffusion"]
    from fhd.diffusion.train import build_model

    model = build_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt, missing, unexpected


def main() -> int:
    p = argparse.ArgumentParser(description="单机器人生成质量评估")
    p.add_argument("--config", default=str(ROOT / "configs" / "diffusion_train.yaml"))
    p.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n-tasks", type=int, default=500)
    p.add_argument("--samples-per-task", type=int, default=8)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--split", default="test")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    run_dir = new_run_dir("S3_diffusion_training", "s3_sample")
    logger = get_logger("fhd.s3.sample", run_dir / "logs" / "sample.log")
    log_section(logger, "S3 单机器人生成质量评估")

    ds = TrajectoryDataset(Path(args.data), args.split,
                           grid_size=int(cfg["world"]["grid_size"]),
                           robot_radius_m=float(cfg["world"]["robot_radius_m"]),
                           clearance_margin_m=float(cfg["world"]["clearance_margin_m"]),
                           width_m=float(cfg["world"]["width_m"]),
                           height_m=float(cfg["world"]["height_m"]))
    ckpt_path = Path(args.ckpt) if args.ckpt else next(
        iter(sorted((ROOT / "results" / "S3_diffusion_training").glob("*/checkpoints/best.pt"),
                    key=lambda x: x.stat().st_mtime, reverse=True)), None)
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit("未找到 checkpoint，请先训练或指定 --ckpt")
    model, ckpt, missing, unexpected = load_model(cfg, ckpt_path, device)
    logger.info("加载 %s（step=%s）", ckpt_path, ckpt.get("step"))
    logger.info("测试集：%d 条轨迹，%d 张地图", len(ds), len(ds.maps))

    schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                            kind=str(cfg["diffusion"]["beta_schedule"]), device=device,
                            dtype=torch.float32)
    num_steps = int(args.steps or cfg["diffusion"]["ddim_sampling_steps"])
    v_max = float(cfg["trajectory"]["max_speed_mps"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    k = int(args.samples_per_task)

    n_tasks = min(args.n_tasks, len(ds))
    indices = np.linspace(0, len(ds) - 1, n_tasks).astype(int).tolist()
    rows = []
    all_success = []
    all_diversity = []
    all_length_err = []
    all_smooth = []
    t_start = time.time()
    gen = torch.Generator(device=device.type).manual_seed(int(cfg["seed"]))

    for idx in indices:
        item = ds[idx]
        start = item["start"].unsqueeze(0).expand(k, -1).to(device)
        goal = item["goal"].unsqueeze(0).expand(k, -1).to(device)
        grid = item["grid"].unsqueeze(0).expand(k, -1, -1, -1).to(device)
        with torch.no_grad():
            samples = ddim_sample(model, schedule, start, goal, grid, num_steps=num_steps,
                                  generator=gen)
        paths = ds.denormalise(samples.float()).detach().cpu()
        mi = int(item["map_idx"])
        world = ds.maps[mi]

        safe, clearance = trajectory_is_static_safe(
            paths, world.obstacles(), world.robot_radius_m, world.clearance_margin_m)
        speed_ok, max_seg = check_speed_limit(paths, v_max, dt_s)
        end_err = (torch.linalg.vector_norm(paths[:, 0] - ds.data["starts"][idx], dim=-1).max().item(),
                   torch.linalg.vector_norm(paths[:, -1] - ds.data["goals"][idx], dim=-1).max().item())
        lengths = path_length(paths)
        gt_len = float(path_length(ds.data["paths"][idx].unsqueeze(0))[0])
        accel = paths[:, 2:] - 2 * paths[:, 1:-1] + paths[:, :-2]
        smooth = float((accel ** 2).sum(-1).mean())
        div = diversity_score(paths)

        for s in range(k):
            rows.append({
                "task": int(idx), "sample": s,
                "static_safe": bool(safe[s]),
                "speed_ok": bool(speed_ok[s]),
                "min_clearance_m": float(clearance[s]),
                "max_segment_m": float(max_seg[s]),
                "start_err_m": float(torch.linalg.vector_norm(paths[s, 0] - ds.data["starts"][idx])),
                "goal_err_m": float(torch.linalg.vector_norm(paths[s, -1] - ds.data["goals"][idx])),
                "length_m": float(lengths[s]),
                "gt_length_m": gt_len,
                "length_ratio": float(lengths[s]) / max(gt_len, 1e-6),
                "smoothness": smooth,
                "success": bool(safe[s] and speed_ok[s]),
            })
        all_success.append(float((safe & speed_ok).float().mean()))
        all_diversity.append(div)
        all_length_err.append(float((lengths / max(gt_len, 1e-6)).mean()))
        all_smooth.append(smooth)
        if len(rows) % (k * 50) == 0:
            logger.info("已评估 %d/%d 任务，累计成功率 %.3f", len(all_success), n_tasks,
                        float(np.mean(all_success)))

    elapsed = time.time() - t_start
    report = StageReport("S3_diffusion_training", run_dir, "S3 单机器人生成质量评估报告", cfg.to_dict())
    report.table("single_robot_samples", rows)
    success_rate = float(np.mean(all_success))
    metrics = {
        "n_tasks": n_tasks, "samples_per_task": k, "ddim_steps": num_steps,
        "success_rate": success_rate,
        "static_collision_rate": float(np.mean([1 - r["static_safe"] for r in rows])),
        "speed_violation_rate": float(np.mean([1 - r["speed_ok"] for r in rows])),
        "start_err_max_m": float(max(r["start_err_m"] for r in rows)),
        "goal_err_max_m": float(max(r["goal_err_m"] for r in rows)),
        "length_ratio_mean": float(np.mean([r["length_ratio"] for r in rows])),
        "smoothness_mean": float(np.mean([r["smoothness"] for r in rows])),
        "diversity_mean_m": float(np.mean(all_diversity)),
        "inference_ms_per_sample": elapsed * 1000 / max(1, len(rows)),
        "total_seconds": elapsed,
    }
    info = system_info()
    report.metrics_from_dict({k2: v for k2, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    report.metrics_from_dict(metrics, group="sampling")
    logger.info("成功率 %.3f | 静态碰撞率 %.3f | 长度比 %.3f | 多样性 %.3f m | %.1f ms/样本",
                success_rate, metrics["static_collision_rate"], metrics["length_ratio_mean"],
                metrics["diversity_mean_m"], metrics["inference_ms_per_sample"])

    # 可视化：12 个任务，每个 4 条候选
    n_show = 12
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    show_idx = np.linspace(0, len(ds) - 1, n_show).astype(int).tolist()
    for ax, idx in zip(axes.ravel(), show_idx):
        item = ds[idx]
        start = item["start"].unsqueeze(0).expand(4, -1).to(device)
        goal = item["goal"].unsqueeze(0).expand(4, -1).to(device)
        grid = item["grid"].unsqueeze(0).expand(4, -1, -1, -1).to(device)
        with torch.no_grad():
            samples = ddim_sample(model, schedule, start, goal, grid, num_steps=num_steps,
                                  generator=torch.Generator(device=device.type).manual_seed(idx))
        paths = ds.denormalise(samples.float()).detach().cpu().numpy()
        draw_map(ax, ds.maps[int(item["map_idx"])])
        gt = ds.data["paths"][idx].numpy()
        ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.0, alpha=0.8)
        for s in range(paths.shape[0]):
            ax.plot(paths[s, :, 0], paths[s, :, 1], lw=1.0, alpha=0.85)
        ax.set_title(f"任务 {idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"单机器人条件 Diffusion 生成样例（DDIM {num_steps} 步，灰=障碍，黑虚线=真实路径）")
    report.save_figure(fig, "generated_samples_grid")

    # 成功率分布图
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(all_success, bins=20, color="#4c72b0", edgecolor="black")
    ax.set_xlabel("每个任务的候选成功率（%d 条候选）" % k)
    ax.set_ylabel("任务数")
    ax.set_title(f"成功率分布（总体 {success_rate:.2%}）")
    report.save_figure(fig, "success_rate_hist")

    report.add_section(
        "## 评估口径\n\n"
        f"- 从 `{args.split}` 划分中均匀取 {n_tasks} 个任务，每个任务采样 {k} 条候选；\n"
        f"- “成功”定义：静态障碍连续几何无碰撞 **且** 每段长度 ≤ v_max·Δt={v_max*dt_s:.2f} m；\n"
        "- 起终点由采样器强制固定，误差应接近 0；\n"
        "- 多样性 = 同一任务 k 条候选的平均两两点距。\n"
    )
    report.add_section(
        "## 结果\n\n"
        f"- 成功率 **{success_rate:.2%}**；静态碰撞率 {metrics['static_collision_rate']:.2%}；"
        f"速度越界率 {metrics['speed_violation_rate']:.2%}；\n"
        f"- 路径长度比（生成/真实）均值 {metrics['length_ratio_mean']:.3f}；\n"
        f"- 候选多样性 {metrics['diversity_mean_m']:.3f} m；\n"
        f"- 推理 {metrics['inference_ms_per_sample']:.1f} ms/样本（DDIM {num_steps} 步）。\n"
        "\n**不追求 100% 成功**：不合格候选一律记为失败，不做后验剔除。\n"
    )
    report.write_report()
    report.write_metrics_csv()
    report.write_status("completed", {}, metrics)
    report.sync_to_docs()
    update_project_progress([
        {"stage": "S3", "summary": f"单机生成成功率 {success_rate:.2%}，多样性 {metrics['diversity_mean_m']:.3f} m。"},
    ])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"运行目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
