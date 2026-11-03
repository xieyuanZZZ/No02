"""第二版方案 · S3：重新训练单机器人 Diffusion。

本阶段**不引入**多机器人风险引导，只求一个可靠的基础规划模型：
  * 固定 H=32、DDIM、起终点补偿、训练损失、运动学约束（方案 §5.1）；
  * 用 S2 选出的地图编码架构；
  * 先 8000 步验证流程，再用 30000 步完整训练；
  * 评估按三类地图分别统计，给出 Task Success@K 与聚类 CI；
  * 输出 train/val loss 与梯度范数曲线（方案 §9 要求）。

用法：
  python scripts/revised_s3_train.py --config configs/s2_C_128_64.yaml --steps 30000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.data.dataset import TrajectoryDataset  # noqa: E402
from fhd.diffusion.schedule import DDPMSchedule  # noqa: E402
from fhd.diffusion.train import Trainer, build_model  # noqa: E402
from fhd.evaluation.reporting import StageReport  # noqa: E402
from fhd.evaluation.simple_eval import evaluate, pick_tasks, summarize  # noqa: E402
from fhd.evaluation.visualizations import AGENT_COLORS, draw_map  # noqa: E402
from fhd.maps.simple_scenes import KIND_LABELS  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import resolve_device, set_seed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402

_setup_fonts()


def read_train_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def plot_curves(rows: list[dict], out: Path) -> None:
    steps = [int(r["step"]) for r in rows if r.get("loss")]
    loss = [float(r["loss"]) for r in rows if r.get("loss")]
    gnorm = [float(r["grad_norm"]) for r in rows if r.get("grad_norm")]
    vr = [(int(r["step"]), float(r["val_loss"])) for r in rows if r.get("val_loss")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(steps, loss, lw=1.0, color="#4c72b0")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("训练步")
    axes[0].set_ylabel("训练 loss（对数轴）")
    axes[0].set_title("Train Loss", fontsize=10)
    axes[0].grid(alpha=0.3)
    if vr:
        axes[1].plot([v[0] for v in vr], [v[1] for v in vr], "o-", ms=4, color="#dd8452")
        axes[1].set_yscale("log")
    axes[1].set_xlabel("训练步")
    axes[1].set_ylabel("验证 loss（对数轴）")
    axes[1].set_title("Validation Loss", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[2].plot(steps, gnorm, lw=0.8, color="#55a868")
    axes[2].set_xlabel("训练步")
    axes[2].set_ylabel("梯度范数（裁剪前）")
    axes[2].set_title("Gradient Norm", fontsize=10)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="S3 单机器人 Diffusion 训练与评估")
    ap.add_argument("--config", default=str(ROOT / "configs" / "s2_C_128_64.yaml"))
    ap.add_argument("--data", default=str(ROOT / "data" / "revised_s1"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--per-kind", type=int, default=25)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--eval-only", action="store_true", help="跳过训练，直接用已有 checkpoint 评估")
    ap.add_argument("--ckpt", default="", help="eval-only 时使用的 checkpoint")
    ap.add_argument("--run", default="", help="eval-only 时复用已有运行目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # 关键：把 CLI 的步数写回配置，保证「实际生效配置」与训练一致（方案 §2 特别提醒）
    cfg_dict = cfg.to_dict()
    cfg_dict["diffusion"]["training_steps"] = int(args.steps)
    cfg_dict["cli_overrides"] = {"steps": args.steps, "data": args.data,
                                 "config_file": args.config}
    from fhd.utils.config import Config
    cfg = Config(cfg_dict)

    set_seed(int(cfg["seed"]))
    device = resolve_device("auto")
    run_dir = new_run_dir("S3_single_agent", "revised_s3")
    logger = get_logger("fhd.s3r", run_dir / "logs" / "s3.log")
    log_section(logger, "S3 单机器人 Diffusion 训练")
    logger.info("配置=%s 数据=%s 步数=%d 设备=%s 地图输入=%s 特征=%s",
                args.config, args.data, args.steps, device,
                cfg["diffusion"].get("map_input_res"), cfg["diffusion"].get("map_spatial_res"))

    if args.eval_only:
        if not args.ckpt or not args.run:
            raise SystemExit("--eval-only 需要同时给 --ckpt 与 --run")
        run_dir = Path(args.run)
        ckpt_path = Path(args.ckpt)
        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        info = {"step": int(ckpt_meta.get("step", -1)),
                "best_val": float(ckpt_meta.get("best_val", float("nan")))}
        train_s = 0.0
        rows = read_train_log(run_dir / "train_log.csv")
        plot_curves(rows, run_dir / "plots" / "s3_train_curves.png")
    else:
        trainer = Trainer(cfg, run_dir, Path(args.data), device, resume=None)
        t0 = time.time()
        info = trainer.train(args.steps, eval_every=1000, checkpoint_every=5000,
                             sample_every=5000, log_every=100)
        train_s = time.time() - t0
        logger.info("训练完成：step=%s best_val=%.5f 用时 %.0f s", info["step"],
                    info["best_val"], train_s)
        rows = read_train_log(run_dir / "train_log.csv")
        plot_curves(rows, run_dir / "plots" / "s3_train_curves.png")
        ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg, device)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    schedule = DDPMSchedule(int(cfg["diffusion"]["num_train_timesteps"]),
                            kind=str(cfg["diffusion"]["beta_schedule"]), device=device)
    ds = TrajectoryDataset(Path(args.data), "test",
                           grid_size=int(cfg["diffusion"].get("map_input_res",
                                                              cfg["world"]["grid_size"])),
                           robot_radius_m=float(cfg["world"]["robot_radius_m"]),
                           clearance_margin_m=float(cfg["world"]["clearance_margin_m"]),
                           width_m=float(cfg["world"]["width_m"]),
                           height_m=float(cfg["world"]["height_m"]))
    tasks = pick_tasks(ds, args.per_kind)
    logger.info("评估 %d 个任务（每类 %d），每任务 %d 条候选", len(tasks), args.per_kind,
                args.samples)
    erows = evaluate(model, schedule, ds, tasks, args.samples,
                     int(cfg["diffusion"]["ddim_sampling_steps"]),
                     float(cfg["trajectory"]["max_speed_mps"]),
                     float(cfg["trajectory"]["dt_s"]), device, int(cfg["seed"]))
    s = summarize(erows, "s3")
    for kind, v in s["by_kind"].items():
        logger.info("%s：有效率 %.3f，TaskSuccess@%d %.3f，静态 %.3f，长度比 %.3f",
                    v["label"], v["success_rate"], args.samples, v["task_success_at_k"],
                    v["static_safe_rate"], v["length_ratio"])

    # 按类型柱状图
    kinds = list(s["by_kind"])
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = np.arange(len(kinds))
    ax.bar(xs - 0.2, [s["by_kind"][k]["success_rate"] for k in kinds], 0.4, label="单条有效率",
           color="#4c72b0")
    ax.bar(xs + 0.2, [s["by_kind"][k]["task_success_at_k"] for k in kinds], 0.4,
           label=f"Task Success@{args.samples}", color="#55a868")
    ax.axhline(0.95, color="r", ls="--", lw=1.2, label="初步目标 95%")
    ax.set_xticks(xs)
    ax.set_xticklabels([KIND_LABELS[k] for k in kinds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例")
    ax.set_title("S3：按地图类型的单机器人规划质量", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "s3_success_by_kind.png", dpi=130)
    plt.close(fig)

    # 样例：每类一张
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.4 * len(kinds), 5.0))
    axes = np.atleast_1d(axes)
    with torch.no_grad():
        for ax, kind in zip(axes, kinds):
            task = next(t for t in tasks if t["kind"] == kind)
            item = ds[task["task"]]
            world = ds.maps[int(item["map_idx"])]
            draw_map(ax, world)
            gt = ds.data["paths"][task["task"]].numpy()
            ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.3, label="参考路径")
            n = 4
            start = item["start"].unsqueeze(0).expand(n, -1).to(device)
            goal = item["goal"].unsqueeze(0).expand(n, -1).to(device)
            grid = item["grid"].unsqueeze(0).expand(n, -1, -1, -1).to(device)
            out = ddim_sample_wrapper(model, schedule, start, goal, grid, cfg)
            paths = ds.denormalise(out.cpu()).numpy()
            for i in range(n):
                ax.plot(paths[i, :, 0], paths[i, :, 1], color=AGENT_COLORS[i % len(AGENT_COLORS)],
                        lw=1.2, alpha=0.9)
            ax.set_title(KIND_LABELS[kind], fontsize=10)
            ax.legend(fontsize=7)
    fig.suptitle("S3：三类场景下的生成样例（实线为 4 条候选）", fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "s3_samples_by_kind.png", dpi=130)
    plt.close(fig)

    # ---------------- 报告 ----------------
    report = StageReport("S3_single_agent", run_dir, "S3 单机器人 Diffusion 训练报告", cfg.to_dict())
    sysinfo = system_info()
    report.metrics_from_dict({k: v for k, v in sysinfo.items() if not isinstance(v, (dict, list))},
                             group="environment")
    report.metric("train_steps", info["step"], group="train")
    report.metric("best_val", info["best_val"], group="train")
    report.metric("train_seconds", round(train_s, 1), group="train")
    report.metric("map_input_res", cfg["diffusion"].get("map_input_res"), group="model")
    report.metric("map_spatial_res", cfg["diffusion"].get("map_spatial_res"), group="model")
    report.metric("success_rate", s["success_rate"], group="eval")
    report.metric("success_ci_lo", s["ci"]["lo"], group="eval")
    report.metric("success_ci_hi", s["ci"]["hi"], group="eval")
    report.metric("task_success_at_k", s["task_success_at_k"], group="eval")
    for kind, v in s["by_kind"].items():
        report.metric(f"{kind}/success_rate", v["success_rate"], group="by_kind")
        report.metric(f"{kind}/task_success", v["task_success_at_k"], group="by_kind")
    report.table("s3_per_task", erows)
    report.table("s3_train_log", [{k: r[k] for k in ("step", "loss", "grad_norm", "val_loss")
                                  if k in r and r[k] != ""} for r in rows])

    lines = ["| 地图类型 | 任务数 | 单条有效率 | Task Success@%d | 静态安全率 | 速度通过率 | 长度比 | 净空 [m] | 推理 [s/条] |"
             % args.samples, "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind, v in s["by_kind"].items():
        lines.append(f"| {v['label']} | {v['n_tasks']} | {v['success_rate']:.3f} | "
                     f"{v['task_success_at_k']:.3f} | {v['static_safe_rate']:.3f} | "
                     f"{v['speed_ok_rate']:.3f} | {v['length_ratio']:.3f} | "
                     f"{v['min_clearance_m']:.3f} | {v['infer_s_per_sample']:.3f} |")
    simple_ok = (s["by_kind"].get("open_crossing", {}).get("success_rate", 0) >= 0.95
                 and s["by_kind"].get("single_obstacle", {}).get("success_rate", 0) >= 0.95)
    report.add_section(
        "## 训练\n\n"
        f"- 配置：`{args.config}`（地图输入 {cfg['diffusion'].get('map_input_res')}，"
        f"特征 {cfg['diffusion'].get('map_spatial_res')}，条件 {cfg['diffusion'].get('map_conditioning')}）\n"
        f"- 数据：`{args.data}`（S1 三类简单场景）\n"
        f"- 步数：**{info['step']}**（CLI 覆盖值已写回配置快照）\n"
        f"- best val loss：{info['best_val']:.5f}；训练用时 {train_s:.0f} s\n"
        f"- 曲线：`plots/s3_train_curves.png`（train/val loss + 梯度范数）\n"
    )
    report.add_section("## 评估（按地图类型分别统计）\n\n" + "\n".join(lines) + "\n")
    report.add_section(
        "## 验收判定\n\n"
        f"- 简单场景（开放交叉 + 单障碍绕行）是否同时达到 95% 有效率："
        f"**{'是' if simple_ok else '否'}**\n"
        f"- 总体单条有效率 {s['success_rate']:.3f}（任务级 95% CI "
        f"[{s['ci']['lo']:.3f}, {s['ci']['hi']:.3f}]）\n"
        f"- 是否满足进入 S4 的条件：**{'是' if simple_ok else '否，需先解决基础模型问题'}**\n\n"
        "瓶颈场景单独统计，不用它掩盖基础模型是否具备正常规划能力（方案 §5.3）。\n"
    )
    report.write_report()
    report.write_metrics_csv()
    report.write_status("completed" if simple_ok else "completed_with_risk",
                        {}, {"simple_scenarios_ok": simple_ok})
    report.sync_to_docs()

    import shutil
    docs = ROOT / "docs" / "revised"
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "report.md", docs / "02b_S3_training.md")
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, docs / "figures" / png.name)
    (run_dir / "manifest.json").write_text(json.dumps({
        "config_file": args.config, "effective_config": cfg.to_dict(),
        "data": args.data, "steps": args.steps, "checkpoint": str(ckpt_path),
        "checkpoint_step": info["step"], "summary": s,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"step": info["step"], "success": round(s["success_rate"], 4),
                      "by_kind": {k: round(v["success_rate"], 4) for k, v in s["by_kind"].items()},
                      "simple_ok": simple_ok}, ensure_ascii=False))
    print(f"运行目录：{run_dir}")
    return 0


def ddim_sample_wrapper(model, schedule, start, goal, grid, cfg):
    from fhd.diffusion.ddim import ddim_sample
    gen = torch.Generator(device=start.device.type).manual_seed(0)
    return ddim_sample(model, schedule, start, goal, grid,
                       num_steps=int(cfg["diffusion"]["ddim_sampling_steps"]),
                       generator=gen)


if __name__ == "__main__":
    raise SystemExit(main())
