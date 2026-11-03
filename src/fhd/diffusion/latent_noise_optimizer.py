"""初始噪声优化（第二版方案 §7.3）。

思路：不改生成过程中的 x̂0，而是把整个 DDIM 链当作一个可微映射

    Q = F_θ(z; M, S, G)

对**初始噪声 z** 做梯度优化，网络权重全程冻结：

    J(z) = λ_R·R_FH(Q) + λ_T·J_finish + λ_S·L_static
           + λ_K·L_kin + λ_G·L_geom + λ_z·‖z − z0‖²

    z^{k+1} = Proj_{RMS(z−z0) ≤ ρ}( z^k − η·∇_z J )

每一次更新后都要**重新跑完整 DDIM**得到新轨迹，不能只改中间干净轨迹。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from fhd.diffusion.differentiable_ddim import (
    ddim_sample_diff,
    noise_rms,
    project_trust_region,
)
from fhd.geometry.sdf import sampled_static_penalty
from fhd.risk.objectives import geometry_cost, kinematic_cost, wait_l1


@dataclass
class NoiseOptConfig:
    steps: int = 5                 # 外层迭代次数
    lr: float = 0.01
    trust_radius: float = 0.1      # RMS(z - z0) 上限
    lambda_R: float = 1.0
    lambda_T: float = 0.05
    lambda_S: float = 10.0
    lambda_K: float = 1.0
    lambda_G: float = 0.5
    lambda_z: float = 1.0          # 噪声偏移正则
    ddim_steps: int = 40


@dataclass
class NoiseOptResult:
    z: torch.Tensor
    paths_m: torch.Tensor
    history: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0


def _loss(
    model,
    schedule,
    risk_model,
    cfg,
    start_n,
    goal_n,
    grid,
    z,
    z0,
    wait,
    pause_p,
    obstacles,
    time_steps,
    dt_s,
    opt: NoiseOptConfig,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """返回 (loss, 分项, 米制轨迹)。整条 DDIM 链可微。"""
    scale = torch.tensor([float(cfg["world"]["width_m"]) / 2.0,
                          float(cfg["world"]["height_m"]) / 2.0],
                         dtype=z.dtype, device=z.device)
    out = ddim_sample_diff(model, schedule, start_n, goal_n, grid, z,
                           num_steps=opt.ddim_steps)
    paths_m = (out.x0 + 1.0) * scale
    risk = risk_model(paths_m, wait, pause_p, time_steps, collision_mode="soft",
                      gate_mode="soft", return_diagnostics=False)
    r_static = sampled_static_penalty(paths_m, obstacles,
                                      float(cfg["world"]["robot_radius_m"]),
                                      float(cfg["world"]["clearance_margin_m"])) \
        if obstacles is not None and getattr(obstacles, "count", 0) > 0 else paths_m.new_zeros(())
    kin, _ = kinematic_cost(paths_m, float(cfg["trajectory"]["max_speed_mps"]), float(dt_s))
    geom = geometry_cost(paths_m, float(cfg["trajectory"]["max_speed_mps"]), float(dt_s))
    drift = (z - z0).pow(2).mean()
    total = (opt.lambda_R * risk["pair_sum_risk"]
             + opt.lambda_T * risk["finish_cost"]
             + opt.lambda_S * r_static
             + opt.lambda_K * kin
             + opt.lambda_G * geom
             + opt.lambda_z * drift)
    detail = {
        "risk": float(risk["pair_sum_risk"].detach()),
        "finish": float(risk["finish_cost"].detach()),
        "static": float(r_static.detach()) if torch.is_tensor(r_static) else float(r_static),
        "kin": float(kin.detach()),
        "geom": float(geom.detach()),
        "drift": float(drift.detach()),
    }
    return total, detail, paths_m


def optimize_initial_noise(
    model,
    schedule,
    risk_model,
    cfg,
    start_m: torch.Tensor,
    goal_m: torch.Tensor,
    grid: torch.Tensor,
    pause_p: torch.Tensor,
    z0: torch.Tensor,
    obstacles=None,
    opt: NoiseOptConfig | None = None,
    time_steps: int = 96,
    dt_s: float = 0.4,
    wait: torch.Tensor | None = None,
) -> NoiseOptResult:
    """对初始噪声做带信赖域的 Adam 优化。"""
    opt = opt or NoiseOptConfig()
    device, dtype = z0.device, z0.dtype
    scale = torch.tensor([float(cfg["world"]["width_m"]) / 2.0,
                          float(cfg["world"]["height_m"]) / 2.0], dtype=dtype, device=device)
    start_n = start_m.to(dtype) / scale - 1.0
    goal_n = goal_m.to(dtype) / scale - 1.0
    if wait is None:
        wait = torch.zeros(start_m.shape[0], int(cfg["trajectory"]["horizon"]) - 1,
                           dtype=dtype, device=device)

    z = z0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=opt.lr)
    history: list[dict] = []
    started = time.time()

    with torch.no_grad():
        _, detail0, paths0 = _loss(model, schedule, risk_model, cfg, start_n, goal_n, grid,
                                   z.detach(), z0, wait, pause_p, obstacles, time_steps,
                                   dt_s, opt)
    history.append({"iter": 0, "loss": float(sum(detail0.values())), **detail0,
                    "noise_rms": 0.0, "path_rms_vs_z0": 0.0})

    for it in range(1, opt.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, detail, paths_m = _loss(model, schedule, risk_model, cfg, start_n, goal_n, grid,
                                      z, z0, wait, pause_p, obstacles, time_steps, dt_s, opt)
        (grad,) = torch.autograd.grad(loss, z)
        z.grad = grad
        optimizer.step()
        z = project_trust_region(z, z0, opt.trust_radius)
        optimizer.state.clear() if False else None  # 保留 Adam 状态（信赖域只投影参数）
        with torch.no_grad():
            _, detail, paths_m = _loss(model, schedule, risk_model, cfg, start_n, goal_n, grid,
                                       z.detach(), z0, wait, pause_p, obstacles, time_steps,
                                       dt_s, opt)
            disp = float((paths_m.detach() - paths0.detach()).pow(2).mean().sqrt())
        row = {"iter": it, "loss": float(sum(detail.values())), **detail,
               "noise_rms": noise_rms(z.detach(), z0), "path_rms_vs_z0": disp}
        history.append(row)

    with torch.no_grad():
        _, _, paths_final = _loss(model, schedule, risk_model, cfg, start_n, goal_n, grid,
                                  z.detach(), z0, wait, pause_p, obstacles, time_steps,
                                  dt_s, opt)
    return NoiseOptResult(z=z.detach(), paths_m=paths_final.detach(), history=history,
                          elapsed_s=time.time() - started)


def random_perturbation_control(
    z0: torch.Tensor, radius: float, generator: torch.Generator
) -> torch.Tensor:
    """同一信赖域内的随机扰动对照：判断梯度优化是否真的优于随机搜索。"""
    noise = torch.randn(z0.shape, generator=generator, dtype=z0.dtype, device=z0.device)
    rms = noise.pow(2).mean().sqrt()
    return z0 + noise * (radius / (rms + 1e-12))
