"""单机器人条件 DDPM 训练（文档 §6.4）。

特性（批复 §24）：
  * 先 500–1,000 步 smoke test，再正式训练；
  * 训练/验证损失、梯度范数逐步写 CSV（可断点续训）；
  * 定期输出生成轨迹图（训练过程中即可检查模型是否在学习）；
  * 保存 latest / best / 定期 checkpoint；
  * 显存不足时自动降低 batch，并用梯度累积维持有效 batch（实际配置写入报告）。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fhd.data.dataset import TrajectoryDataset
from fhd.diffusion.ddim import ddim_sample
from fhd.diffusion.model import EMA, ConditionalUNet1D, count_parameters
from fhd.diffusion.schedule import DDPMSchedule
from fhd.evaluation.reporting import PROJECT_ROOT  # noqa: F401
from fhd.utils.logging import get_logger


def build_model(cfg, device) -> ConditionalUNet1D:
    d = cfg["diffusion"]
    model = ConditionalUNet1D(
        in_channels=2,
        base_channels=int(d["unet_channels"][0]),
        mults=(1, int(d["unet_channels"][1]) // int(d["unet_channels"][0]),
               int(d["unet_channels"][2]) // int(d["unet_channels"][0])),
        cond_dim=int(d["cond_feature_dim"]),
        map_feature_dim=int(d["map_feature_dim"]),
        grid_size=int(cfg["world"]["grid_size"]),
        map_res=int(d.get("map_spatial_res", 8)),
        map_conditioning=str(d.get("map_conditioning", "grid_sample")),
    ).to(device)
    model.horizon = int(cfg["trajectory"]["horizon"])
    return model


class Trainer:
    def __init__(self, cfg, run_dir: Path, dataset_root: Path, device, resume: Path | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = device
        self.logger = get_logger("fhd.s3.train", self.run_dir / "logs" / "train.log")
        d = cfg["diffusion"]

        self.train_ds = TrajectoryDataset(
            dataset_root, "train",
            grid_size=int(cfg["world"]["grid_size"]),
            robot_radius_m=float(cfg["world"]["robot_radius_m"]),
            clearance_margin_m=float(cfg["world"]["clearance_margin_m"]),
            width_m=float(cfg["world"]["width_m"]), height_m=float(cfg["world"]["height_m"]),
        )
        self.val_ds = TrajectoryDataset(
            dataset_root, "val",
            grid_size=int(cfg["world"]["grid_size"]),
            robot_radius_m=float(cfg["world"]["robot_radius_m"]),
            clearance_margin_m=float(cfg["world"]["clearance_margin_m"]),
            width_m=float(cfg["world"]["width_m"]), height_m=float(cfg["world"]["height_m"]),
        )
        self.batch_size = int(d["batch_size"])
        self.grad_accum = int(d["grad_accum_steps"])
        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                       drop_last=True, num_workers=int(d["num_workers"]))
        self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                     num_workers=int(d["num_workers"]))

        self.schedule = DDPMSchedule(int(d["num_train_timesteps"]), kind=str(d["beta_schedule"]),
                                     device=device, dtype=torch.float32)
        self.model = build_model(cfg, device)
        self.ema = EMA(self.model, decay=float(d["ema_decay"]))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(d["lr"]),
                                           weight_decay=float(d["weight_decay"]))
        self.step = 0
        self.best_val = float("inf")
        self.log_rows: list[dict] = []
        self.grad_accum_actual = self.grad_accum
        self.batch_size_actual = self.batch_size

        if resume is not None and Path(resume).exists():
            self._load(Path(resume))

        # 固定一批验证任务用于可视化
        n_vis = min(8, len(self.val_ds))
        batch = [self.val_ds[i] for i in range(n_vis)]
        self.vis = {
            "start": torch.stack([b["start"] for b in batch]).to(device),
            "goal": torch.stack([b["goal"] for b in batch]).to(device),
            "grid": torch.stack([b["grid"] for b in batch]).to(device),
            "x0": torch.stack([b["x0"] for b in batch]).to(device),
            "path_m": torch.stack([b["path_m"] for b in batch]).to(device),
            "map_idx": [int(b["map_idx"]) for b in batch],
        }

    # ------------------------------------------------------------------
    def _save(self, name: str, extra: dict | None = None) -> Path:
        path = self.run_dir / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "best_val": self.best_val,
            "batch_size": self.batch_size_actual,
            "grad_accum": self.grad_accum_actual,
            "config": self.cfg.to_dict(),
            "extra": extra or {},
        }, path)
        return path

    def _load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.ema.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = int(ckpt.get("step", 0))
        self.best_val = float(ckpt.get("best_val", float("inf")))
        self.logger.info("从 %s 恢复训练，step=%d", path.name, self.step)

    # ------------------------------------------------------------------
    def _batch_loss(self, batch) -> tuple[torch.Tensor, dict]:
        x0 = batch["x0"].to(self.device)
        start = batch["start"].to(self.device)
        goal = batch["goal"].to(self.device)
        grid = batch["grid"].to(self.device)
        b = x0.shape[0]
        t = torch.randint(0, self.schedule.num_steps, (b,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        pred = self.model(x_t, t, start, goal, grid)
        loss = torch.nn.functional.mse_loss(pred, noise)
        return loss, {"loss": float(loss.detach()), "t_mean": float(t.float().mean())}

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict:
        self.model.eval()
        total, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if max_batches is not None and i >= max_batches:
                break
            loss, _ = self._batch_loss(batch)
            total += float(loss)
            n += 1
        self.model.train()
        return {"val_loss": total / max(n, 1)}

    # ------------------------------------------------------------------
    def train(self, total_steps: int, eval_every: int, checkpoint_every: int, sample_every: int,
              log_every: int, time_budget_s: float | None = None) -> dict:
        d = self.cfg["diffusion"]
        csv_path = self.run_dir / "train_log.csv"
        new_file = not csv_path.exists()
        csv_file = csv_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["step", "loss", "grad_norm", "lr", "val_loss", "elapsed_s", "samples_per_s"],
        )
        if new_file:
            writer.writeheader()

        self.model.train()
        started = time.time()
        running, run_count = 0.0, 0
        data_iter = iter(self.train_loader)
        ckpt_extra = {"horizon": int(self.cfg["trajectory"]["horizon"])}
        self.logger.info("开始训练：目标 %d 步，batch=%d，累积=%d，参数=%d",
                         total_steps, self.batch_size_actual, self.grad_accum_actual,
                         count_parameters(self.model))

        while self.step < total_steps:
            if time_budget_s is not None and time.time() - started > time_budget_s:
                self.logger.info("达到本次时间预算，保存并退出（step=%d）", self.step)
                break
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(self.grad_accum_actual):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                loss, info = self._batch_loss(batch)
                (loss / self.grad_accum_actual).backward()
                step_loss += info["loss"] / self.grad_accum_actual
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
            self.optimizer.step()
            self.ema.update(self.model)
            self.step += 1
            running += step_loss
            run_count += 1

            if self.step % log_every == 0:
                elapsed = time.time() - started
                row = {
                    "step": self.step,
                    "loss": running / run_count,
                    "grad_norm": grad_norm,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "val_loss": "",
                    "elapsed_s": round(elapsed, 1),
                    "samples_per_s": round(self.step * self.batch_size_actual * self.grad_accum_actual / max(elapsed, 1e-6), 1),
                }
                self.log_rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                running, run_count = 0.0, 0
                if self.step % (log_every * 20) == 0:
                    self.logger.info("step %6d | loss %.5f | grad %.3f | %.1f 样本/s",
                                     self.step, row["loss"], grad_norm, row["samples_per_s"])

            if self.step % eval_every == 0:
                val = self.evaluate(max_batches=10)
                self.log_rows.append({"step": self.step, "val_loss": val["val_loss"]})
                writer.writerow({"step": self.step, "loss": "", "grad_norm": "", "lr": "",
                                 "val_loss": val["val_loss"],
                                 "elapsed_s": round(time.time() - started, 1), "samples_per_s": ""})
                csv_file.flush()
                self.logger.info("step %6d | 验证 loss %.5f", self.step, val["val_loss"])
                if val["val_loss"] < self.best_val:
                    self.best_val = val["val_loss"]
                    self._save("best.pt", ckpt_extra)
                    (self.run_dir / "best_val.json").write_text(
                        json.dumps({"step": self.step, "val_loss": val["val_loss"]}), encoding="utf-8")

            if self.step % checkpoint_every == 0:
                self._save("latest.pt", ckpt_extra)
                self._save(f"step_{self.step:06d}.pt", ckpt_extra)

            if self.step % sample_every == 0:
                self._quick_sample_plot(tag=f"step{self.step:06d}")

        self._save("latest.pt", ckpt_extra)
        csv_file.close()
        if not (self.run_dir / "checkpoints" / "best.pt").exists():
            self._save("best.pt", ckpt_extra)
        return {"step": self.step, "best_val": self.best_val,
                "elapsed_s": round(time.time() - started, 1)}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _quick_sample_plot(self, tag: str) -> None:
        """用 EMA 权重采样几条轨迹并画图（训练过程中持续更新）。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from fhd.evaluation.visualizations import draw_map
        from fhd.maps.generate import WorldMap

        model_backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.ema.copy_to(self.model)
        gen = torch.Generator(device=self.device.type).manual_seed(0)
        samples = ddim_sample(self.model, self.schedule, self.vis["start"], self.vis["goal"],
                              self.vis["grid"], num_steps=20, generator=gen)
        for k, v in model_backup.items():
            self.model.state_dict()[k].copy_(v)
        self.model.train()

        paths = self.val_ds.denormalise(samples.float()).detach().cpu().numpy()
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax_i, idx in enumerate(range(min(4, paths.shape[0]))):
            ax = axes.ravel()[ax_i]
            world = self.val_ds.maps[self.vis["map_idx"][idx]]
            draw_map(ax, world)
            gt = self.vis["path_m"][idx].detach().cpu().numpy()
            ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.2, label="真实路径")
            ax.plot(paths[idx, :, 0], paths[idx, :, 1], "-", color="#d62728", lw=1.8, label="采样路径")
            ax.legend(fontsize=8)
            ax.set_title(f"验证样本 {idx}", fontsize=9)
        fig.suptitle(f"DDIM 采样（{tag}，20 步，EMA 权重）")
        path = self.run_dir / "plots" / f"sample_{tag}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        self.logger.info("已保存采样图：%s", path.name)
