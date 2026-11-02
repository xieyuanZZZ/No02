"""DDIM 采样（不含风险引导），文档 §7.3。"""

from __future__ import annotations

from typing import Callable

import torch

from fhd.diffusion.schedule import DDPMSchedule


def project_endpoints(x: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """把每条轨迹的首尾点替换为条件起终点（端点固定，文档 §6.4）。"""
    out = x.clone()
    out[:, 0, :] = start
    out[:, -1, :] = goal
    return out


@torch.no_grad()
def ddim_sample(
    model,
    schedule: DDPMSchedule,
    start: torch.Tensor,
    goal: torch.Tensor,
    grid: torch.Tensor,
    num_steps: int = 40,
    generator: torch.Generator | None = None,
    x_init: torch.Tensor | None = None,
    guide_fn: Callable[[int, int, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    record: list | None = None,
) -> torch.Tensor:
    """从纯噪声出发做 DDIM 采样。

    参数
    ----
    guide_fn : 可选回调 (step_index, t, x_t, x0_pred) -> 修改后的 x0_pred，
               供 S4 的首次碰撞风险引导接入；为 None 时就是标准 DDIM。
    record   : 若提供列表，则记录每个采样步的 x0 预测（用于 S4 的画图与轨迹分析）。

    返回归一化后的干净轨迹 [B,H,2]。
    """
    model.eval()
    batch, h, _ = start.shape[0], x_init.shape[1] if x_init is not None else None, 2
    if x_init is None:
        x = torch.randn(batch, h or 32, 2, generator=generator, device=start.device, dtype=start.dtype)
    else:
        x = x_init.clone()

    ts = schedule.ddim_timesteps(num_steps)
    for i, t in enumerate(ts.tolist()):
        tt = torch.full((batch,), int(t), device=start.device, dtype=torch.long)
        eps = model(x, tt, start, goal, grid)
        x0 = schedule.predict_x0(x, eps, tt)
        x0 = project_endpoints(x0, start, goal)
        if guide_fn is not None:
            x0 = guide_fn(i, int(t), x, x0)
            x0 = project_endpoints(x0, start, goal)
        if record is not None:
            record.append({"step": i, "t": int(t), "x0": x0.detach().clone()})
        t_next = ts[i + 1].item() if i + 1 < len(ts) else -1
        x = schedule.ddim_step(x, x0, int(t), int(t_next))
    return x


@torch.no_grad()
def ddim_sample_with_eps(
    model,
    schedule: DDPMSchedule,
    start: torch.Tensor,
    goal: torch.Tensor,
    grid: torch.Tensor,
    num_steps: int = 40,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回最终轨迹与最后一步的 ε（诊断用）。"""
    x = torch.randn(start.shape[0], getattr(model, "horizon", 32), 2, generator=generator,
                    device=start.device, dtype=start.dtype)
    ts = schedule.ddim_timesteps(num_steps)
    eps = torch.zeros_like(x)
    for i, t in enumerate(ts.tolist()):
        tt = torch.full((start.shape[0],), int(t), device=start.device, dtype=torch.long)
        eps = model(x, tt, start, goal, grid)
        x0 = project_endpoints(schedule.predict_x0(x, eps, tt), start, goal)
        t_next = ts[i + 1].item() if i + 1 < len(ts) else -1
        x = schedule.ddim_step(x, x0, int(t), int(t_next))
    return x, eps
