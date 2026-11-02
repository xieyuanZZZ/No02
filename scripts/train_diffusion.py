"""S3：训练单机器人条件 DDPM。

用法：
  # smoke test（500–1000 步，验证 loss/梯度/数据加载/采样都正常）
  python scripts/train_diffusion.py --config configs/diffusion_train.yaml --smoke
  # 正式训练（可断点续训、可设置本次时间预算）
  python scripts/train_diffusion.py --config configs/diffusion_train.yaml --steps 40000 --time-budget 14400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.diffusion.train import Trainer, build_model  # noqa: E402
from fhd.evaluation.reporting import StageReport, update_project_progress  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir, system_info  # noqa: E402
from fhd.utils.seed import resolve_device, set_seed  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="训练单机器人条件 Diffusion")
    p.add_argument("--config", default=str(ROOT / "configs" / "diffusion_train.yaml"))
    p.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", default=None, help="checkpoint 路径（默认读 run 目录下 latest.pt）")
    p.add_argument("--run-dir", default=None, help="复用已有运行目录（断点续训）")
    p.add_argument("--time-budget", type=float, default=None, help="本次最长训练秒数")
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    d = cfg["diffusion"]
    if args.batch_size:
        cfg["diffusion"]["batch_size"] = args.batch_size

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = new_run_dir("S3_diffusion_training", "s3_train")
    logger = get_logger("fhd.s3", run_dir / "logs" / "s3_train.log")
    log_section(logger, "S3 单机器人条件 Diffusion 训练")
    logger.info("设备 %s | 数据 %s", device, args.data)

    resume = Path(args.resume) if args.resume else (run_dir / "checkpoints" / "latest.pt")
    if not resume.exists():
        resume = None

    try:
        trainer = Trainer(cfg, run_dir, Path(args.data), device, resume=resume)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        cfg["diffusion"]["batch_size"] = max(8, int(cfg["diffusion"]["batch_size"]) // 2)
        cfg["diffusion"]["grad_accum_steps"] = 2
        logger.warning("显存不足，自动降低 batch 到 %d 并启用梯度累积×2", cfg["diffusion"]["batch_size"])
        trainer = Trainer(cfg, run_dir, Path(args.data), device, resume=resume)

    if args.smoke:
        steps = int(args.steps or d["smoke_steps"])
        result = trainer.train(steps, eval_every=100, checkpoint_every=300, sample_every=300,
                               log_every=10, time_budget_s=args.time_budget)
        status = "completed"
    else:
        steps = int(args.steps or d["training_steps"])
        result = trainer.train(steps, eval_every=int(d["val_every"]),
                               checkpoint_every=int(d["checkpoint_every"]),
                               sample_every=int(d["sample_every"]),
                               log_every=int(d["log_every"]),
                               time_budget_s=args.time_budget)
        status = "completed" if result["step"] >= steps else "partial"

    logger.info("训练结束：step=%d，best_val=%.5f，用时 %.0f s", result["step"],
                result["best_val"], result["elapsed_s"])

    report = StageReport("S3_diffusion_training", run_dir,
                         "S3 单机器人条件 Diffusion 训练报告", cfg.to_dict())
    info = system_info()
    report.metrics_from_dict({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                             group="environment")
    report.metric("steps_completed", result["step"], group="train")
    report.metric("best_val_loss", result["best_val"], group="train")
    report.metric("train_seconds", result["elapsed_s"], group="train")
    report.metric("batch_size", trainer.batch_size_actual, group="train")
    report.metric("grad_accum", trainer.grad_accum_actual, group="train")
    report.metric("n_train_samples", len(trainer.train_ds), group="data")
    report.metric("n_val_samples", len(trainer.val_ds), group="data")

    stats_file = Path(args.data) / "meta.json"
    if stats_file.exists():
        meta = json.loads(stats_file.read_text(encoding="utf-8"))
        (run_dir / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        for split, st in meta.get("splits", {}).items():
            report.metrics_from_dict({f"{split}/{k}": v for k, v in st.items()}, group="data")

    report.add_section(
        "## 训练设置\n\n"
        f"- 去噪网络：条件 1D U-Net（通道 {d['unet_channels']}，FiLM 条件注入），"
        f"参数约 {sum(p.numel() for p in trainer.model.parameters()) / 1e6:.2f} M；\n"
        f"- 调度：{d['beta_schedule']}，T={d['num_train_timesteps']}，预测量 ε；\n"
        f"- 优化器 AdamW，lr={d['lr']}，batch={trainer.batch_size_actual}，"
        f"梯度累积 {trainer.grad_accum_actual}，EMA decay={d['ema_decay']}；\n"
        f"- 训练样本 {len(trainer.train_ds)} 条，验证样本 {len(trainer.val_ds)} 条。\n"
    )
    report.add_section(
        "## 说明\n\n"
        "- 训练过程中的 loss/梯度范数/吞吐逐步写入 `train_log.csv`（断点续训会追加）；\n"
        "- 每 `sample_every` 步用 EMA 权重做一次 DDIM 采样并出图，便于观察是否在学习；\n"
        "- 检查点：`latest.pt`（可续训）、`best.pt`（验证损失最低）、`step_XXXXXX.pt`。\n"
    )
    report.write_report()
    report.write_metrics_csv()
    report.write_status(status, {}, {"step": result["step"], "best_val": result["best_val"]})
    report.sync_to_docs()
    update_project_progress([
        {"stage": "S3", "summary": f"训练 {result['step']} 步，最佳验证损失 {result['best_val']:.5f}。"},
    ])
    print(f"\n训练运行目录: {run_dir}")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
