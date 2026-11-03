"""可微分 DDIM（第二版方案 §7.2）。

与 `ddim.py` 的区别：**保留计算图**，让梯度从最终轨迹回传到初始噪声 z。

约定：
  * 网络权重冻结（`requires_grad=False`），但前向不套 `torch.no_grad()`；
  * 确定性 DDIM，eta=0；
  * 起终点在**归一化空间**用可微的方式写回（不是 clone+赋值，便于审计梯度）；
  * 与普通 `ddim_sample` 必须逐点一致（由 `tests/test_differentiable_ddim.py` 锁定）。

输入/输出都是归一化坐标 [-1,1]；米制转换由调用方负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fhd.diffusion.schedule import DDPMSchedule


def project_endpoints_diff(
    x: torch.Tensor, start: torch.Tensor, goal: torch.Tensor
) -> torch.Tensor:
    """可微的端点写回：首尾点替换为条件起终点，其余点原样。

    用乘法掩码而不是切片赋值，避免 CopySlices 的隐式梯度语义。
    """
    b, h, _ = x.shape
    mask = torch.zeros(1, h, 1, dtype=x.dtype, device=x.device)
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0
    cond = torch.zeros(b, h, 2, dtype=x.dtype, device=x.device)
    cond[:, 0, :] = start
    cond[:, -1, :] = goal
    return x * (1.0 - mask) + cond


@dataclass
class DiffDDIMResult:
    x0: torch.Tensor                       # 最终干净轨迹（归一化），保留计算图
    x0_steps: list[torch.Tensor] = field(default_factory=list)   # 每步 x̂0
    x_steps: list[torch.Tensor] = field(default_factory=list)    # 每步 x_t


def ddim_sample_diff(
    model,
    schedule: DDPMSchedule,
    start: torch.Tensor,
    goal: torch.Tensor,
    grid: torch.Tensor,
    z: torch.Tensor,
    num_steps: int = 40,
    record: bool = False,
) -> DiffDDIMResult:
    """从初始噪声 z 出发的可微 DDIM。

    参数
    ----
    start, goal : [B,2] 归一化起终点
    grid        : [B,1,G,G]
    z           : [B,H,2] 初始噪声（就是扩散链最开始的 x_T）
    """
    x = z
    ts = schedule.ddim_timesteps(num_steps)
    x0_steps, x_steps = [], []
    x0 = x
    for i, t in enumerate(ts.tolist()):
        tt = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
        eps = model(x, tt, start, goal, grid)
        x0 = schedule.predict_x0(x, eps, tt)
        x0 = project_endpoints_diff(x0, start, goal)
        if record:
            x0_steps.append(x0)
            x_steps.append(x)
        t_next = ts[i + 1].item() if i + 1 < len(ts) else -1
        x = schedule.ddim_step(x, x0, int(t), int(t_next))
    return DiffDDIMResult(x0=x0, x0_steps=x0_steps, x_steps=x_steps)


def noise_rms(z: torch.Tensor, z0: torch.Tensor) -> float:
    return float((z.detach() - z0.detach()).pow(2).mean().sqrt())


def project_trust_region(z: torch.Tensor, z0: torch.Tensor, radius: float) -> torch.Tensor:
    """把 z 投影回 RMS(z-z0) ≤ radius 的球内（超球面按比例缩放）。"""
    with torch.no_grad():
        delta = z - z0
        rms = delta.pow(2).mean().sqrt()
        if float(rms) > radius:
            delta = delta * (radius / (rms + 1e-12))
        return (z0 + delta).detach().requires_grad_(True)
