"""DDPM/DDIM 噪声调度（文档 §6.4）。

采用余弦调度（Nichol & Dhariwal），预测量为 ε（epsilon）。
"""

from __future__ import annotations

import math

import torch


class DDPMSchedule:
    """预计算 beta / alpha / alpha_bar 及其常用组合。"""

    def __init__(self, num_steps: int = 1000, kind: str = "cosine", s: float = 0.008,
                 dtype: torch.dtype = torch.float32, device=None):
        self.num_steps = int(num_steps)
        self.kind = kind
        self.dtype = dtype
        self.device = device
        if kind == "cosine":
            steps = torch.linspace(0, self.num_steps, self.num_steps + 1, dtype=torch.float64)
            f = torch.cos(((steps / self.num_steps) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 1e-8, 0.999)
        elif kind == "linear":
            betas = torch.linspace(1e-4, 0.02, self.num_steps, dtype=torch.float64)
        else:
            raise ValueError(f"未知调度: {kind}")
        self.betas = betas.to(dtype=dtype, device=device)
        self.alphas = (1.0 - self.betas).to(dtype=dtype, device=device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(dtype=dtype, device=device)
        self.alphas_cumprod_prev = torch.cat(
            [self.alphas_cumprod.new_ones(1), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (
            1.0 - self.alphas_cumprod_prev
        ) / (1.0 - self.alphas_cumprod)

    # ------------------------------------------------------------------
    def to(self, device=None, dtype=None) -> "DDPMSchedule":
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.betas = self.betas.to(device=self.device, dtype=self.dtype)
        self.alphas = self.alphas.to(device=self.device, dtype=self.dtype)
        self.alphas_cumprod = self.alphas_cumprod.to(device=self.device, dtype=self.dtype)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device=self.device, dtype=self.dtype)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device=self.device, dtype=self.dtype)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device=self.device, dtype=self.dtype
        )
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device=self.device, dtype=self.dtype)
        self.posterior_variance = self.posterior_variance.to(device=self.device, dtype=self.dtype)
        return self

    # ------------------------------------------------------------------
    def _gather(self, values: torch.Tensor, t: torch.Tensor, dims: int) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(-1, *([1] * dims))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """前向加噪：x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps"""
        dims = x0.dim() - 1
        return (
            self._gather(self.sqrt_alphas_cumprod, t, dims) * x0
            + self._gather(self.sqrt_one_minus_alphas_cumprod, t, dims) * noise
        )

    def predict_x0(self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """由 ε 预测恢复干净样本（文档 §7.2）。"""
        dims = x_t.dim() - 1
        return (
            x_t - self._gather(self.sqrt_one_minus_alphas_cumprod, t, dims) * eps
        ) / self._gather(self.sqrt_alphas_cumprod, t, dims)

    def predict_eps(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """由干净样本反推 ε（DDIM 一致性重算用，文档 §7.3）。"""
        dims = x_t.dim() - 1
        return (
            x_t - self._gather(self.sqrt_alphas_cumprod, t, dims) * x0
        ) / self._gather(self.sqrt_one_minus_alphas_cumprod, t, dims)

    def ddim_timesteps(self, num_steps: int) -> torch.Tensor:
        """等间隔的 DDIM 子序列（含 0），返回降序索引。"""
        step_ratio = self.num_steps // num_steps
        seq = torch.arange(0, self.num_steps, step_ratio, dtype=torch.long)
        return torch.flip(seq, dims=[0])

    def ddim_step(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: int,
        t_next: int,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """确定性 DDIM（eta=0）一步。"""
        ab_t = self.alphas_cumprod[t]
        ab_next = self.alphas_cumprod[t_next] if t_next >= 0 else self.alphas_cumprod.new_ones(())
        eps = (x_t - torch.sqrt(ab_t) * x0_pred) / torch.sqrt(1.0 - ab_t)
        x_next = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1.0 - ab_next) * eps
        if eta > 0:
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next))
            noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
            x_next = x_next + sigma * noise
        return x_next


def beta_schedule_statistics(schedule: DDPMSchedule) -> dict:
    return {
        "num_steps": schedule.num_steps,
        "kind": schedule.kind,
        "beta_min": float(schedule.betas.min()),
        "beta_max": float(schedule.betas.max()),
        "alpha_bar_first": float(schedule.alphas_cumprod[0]),
        "alpha_bar_last": float(schedule.alphas_cumprod[-1]),
    }
