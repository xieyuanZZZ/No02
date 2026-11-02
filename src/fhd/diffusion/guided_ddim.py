"""推理期风险引导的 DDIM 采样（文档 §7，S4 的核心接口）。

流程（每个 DDIM 采样步）：
  1. 冻结去噪网络，前向得到 ε 与 x̂0（归一化空间）；
  2. 反归一化到米制，建立**独立的局部计算图**（不反传整条采样链）；
  3. 用 FirstHitRisk 计算 ΣR_soft 与 J_finish 等项，得到对 Q 与 w 的梯度；
  4. 用带裁剪的投影梯度修正 x̂0 与 w；
  5. 用修正后的 x̂0 重算 ε_ref，再做一步 DDIM。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from fhd.diffusion.ddim import project_endpoints
from fhd.diffusion.schedule import DDPMSchedule
from fhd.geometry.primitives import segment_point_distance
from fhd.risk.objectives import guidance_loss, relative_proportions


@dataclass
class GuidanceConfig:
    enabled: bool = True
    start_fraction: float = 0.25      # 前 25% 采样步不引导（批复 §20）
    step_size_x: float = 0.05         # 归一化坐标下的步长
    step_size_w: float = 0.5          # 等待（时隙）步长
    grad_clip_norm: float = 1.0
    max_wait_per_point: float = 4.0
    max_total_wait: float = 8.0
    # all | free_space | strict | manual
    wait_mask_mode: str = "free_space"
    wait_mask: torch.Tensor | None = None
    hard_eval_every: int = 5
    mc_samples: int = 0
    optimize_path: bool = True        # A1/A3 打开，A2 关闭
    optimize_wait: bool = True        # A1 关闭，A2/A3 打开


@dataclass
class GuidanceResult:
    paths_m: torch.Tensor
    wait: torch.Tensor
    history: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)


def project_wait_local(wait: torch.Tensor, mask: torch.Tensor, max_per_point: float,
                       max_total: float) -> torch.Tensor:
    w = wait * mask.to(wait.dtype)
    w = torch.clamp(w, 0.0, max_per_point)
    totals = w.sum(dim=-1, keepdim=True)
    scale = torch.clamp(max_total / torch.clamp(totals, min=1e-12), max=1.0)
    return torch.clamp(w * scale, 0.0, max_per_point)


def auto_wait_mask(
    paths_m: torch.Tensor,
    d_safe: float,
    clearance_ok: torch.Tensor,
    mode: str = "free_space",
    n_candidates: int = 3,
) -> torch.Tensor:
    """等待点检测（批复 §22）。

    三种判据，按“越严格越保守”排列：

    * ``strict``     ：静态自由空间 **且** 到其他机器人整条路径的最小距离 ≥ d_safe。
      最安全，但在交叉/门口场景会把唯一有用的等待点全部排除（因为这些点就在冲突区旁边），
      实测会让梯度引导无法产生任何等待。
    * ``free_space``（默认）：静态自由空间内的所有路点，即文档 §3.3 所说
      “放开所有处于安全自由空间内的等待点”。是否真的安全由风险项本身惩罚，
      并在规划结束后用 `optim.evaluate.check_wait_safety` 独立复核。
    * ``all``        ：不限制（消融用）。
    """
    n, h, _ = paths_m.shape
    if mode == "all":
        return torch.ones(n, h - 1, dtype=torch.bool, device=paths_m.device)

    mask = torch.zeros(n, h - 1, dtype=torch.bool, device=paths_m.device)
    static_free = clearance_ok if clearance_ok is not None else torch.ones(
        n, h, dtype=torch.bool, device=paths_m.device)
    for i in range(n):
        for m in range(1, h - 2):
            if not bool(static_free[i, m]):
                continue
            if mode == "free_space":
                mask[i, m] = True
                continue
            q = paths_m[i, m]
            best = float("inf")
            for j in range(n):
                if j == i:
                    continue
                a = paths_m[j, :-1]
                b = paths_m[j, 1:]
                d = segment_point_distance(a, b, q.unsqueeze(0).expand(a.shape[0], 2))
                best = min(best, float(d.min()))
            if best >= d_safe:
                mask[i, m] = True
    if not mask.any():
        # 兜底：至少有可等待的路点，否则等待变量永远学不到东西
        for i in range(n):
            for m in range(1, h - 2):
                if bool(static_free[i, m]):
                    mask[i, m] = True
    return mask


def guided_ddim_sample(
    model,
    schedule: DDPMSchedule,
    risk_model,
    cfg,
    start_m: torch.Tensor,
    goal_m: torch.Tensor,
    grid: torch.Tensor,
    pause_p: torch.Tensor,
    obstacles=None,
    num_steps: int = 40,
    generator: torch.Generator | None = None,
    guide: GuidanceConfig | None = None,
    time_steps: int = 96,
    dt_s: float = 0.4,
    d_safe: float = 0.35,
    clearance_ok: torch.Tensor | None = None,
    x_init: torch.Tensor | None = None,
    objective_fn=None,
) -> GuidanceResult:
    """多机器人联合引导采样。

    start_m/goal_m : [N,2]（米制）
    grid           : [N,1,G,G]（同一地图时可直接广播）
    返回米制轨迹与等待计划。
    """
    guide = guide or GuidanceConfig()
    n = start_m.shape[0]
    h = int(cfg["trajectory"]["horizon"])
    w_m = float(cfg["world"]["width_m"])
    h_m = float(cfg["world"]["height_m"])
    scale = torch.tensor([w_m / 2.0, h_m / 2.0], dtype=torch.float32, device=start_m.device)

    start_n = start_m / scale - 1.0
    goal_n = goal_m / scale - 1.0
    start_n = start_n.to(torch.float32)
    goal_n = goal_n.to(torch.float32)
    grid = grid.to(torch.float32)

    if x_init is None:
        x = torch.randn(n, h, 2, generator=generator, device=start_n.device, dtype=torch.float32)
    else:
        x = x_init.clone().to(torch.float32)

    if guide.wait_mask is not None:
        wait_mask = guide.wait_mask.to(device=start_m.device, dtype=torch.bool)
    elif guide.wait_mask_mode == "all":
        wait_mask = torch.ones(n, h - 1, dtype=torch.bool, device=start_m.device)
    else:
        wait_mask = None  # auto：在第一个引导步根据当前 x0 现场检测

    wait = torch.zeros(n, h - 1, dtype=torch.float32, device=start_m.device)
    history: list[dict] = []
    ts = schedule.ddim_timesteps(num_steps)
    n_guided = 0

    for i, t in enumerate(ts.tolist()):
        tt = torch.full((n,), int(t), device=start_n.device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, tt, start_n, goal_n, grid)
            x0 = schedule.predict_x0(x, eps, tt)
            x0 = project_endpoints(x0, start_n, goal_n)

        fraction = i / max(1, len(ts) - 1)
        # 前 start_fraction 比例的采样步不做引导（批复 §20）
        do_guide = guide.enabled and fraction >= guide.start_fraction

        guided_this_step = False
        if do_guide:
            n_guided += 1
            guided_this_step = True
            # 单独建立局部计算图
            x0_m = (x0.detach() + 1.0) * scale
            x0_local = x0_m.detach().clone().requires_grad_(True)
            wait_local = wait.detach().clone().requires_grad_(True)
            if objective_fn is None:
                risk = risk_model(x0_local, wait_local, pause_p, time_steps,
                                  collision_mode="soft", gate_mode="soft", return_diagnostics=False)
                loss, detail = guidance_loss(x0_local, wait_local, risk, cfg, obstacles=obstacles)
                soft_risk_value = float(risk["pair_sum_risk"].detach())
                finish_value = float(risk["finish_cost"].detach())
                weights = detail["weighted"]
            else:
                loss, detail = objective_fn(x0_local, wait_local)
                soft_risk_value = float(detail.get("soft_risk", torch.zeros(())).detach())
                finish_value = float(detail.get("finish_cost", torch.zeros(())).detach())
                weights = detail.get("weighted", {})
            g_x, g_w = torch.autograd.grad(loss, (x0_local, wait_local), allow_unused=True)
            g_x = torch.zeros_like(x0_local) if g_x is None else g_x
            g_w = torch.zeros_like(wait_local) if g_w is None else g_w
            if not guide.optimize_path:
                g_x = torch.zeros_like(g_x)
            if not guide.optimize_wait:
                g_w = torch.zeros_like(g_w)

            # 端点梯度屏蔽 + 梯度裁剪
            g_x[:, 0, :] = 0.0
            g_x[:, -1, :] = 0.0
            norm_x = g_x.reshape(n, -1).norm(dim=-1).clamp(min=1e-12)
            scale_x = (guide.grad_clip_norm / norm_x).clamp(max=1.0)
            g_x = g_x * scale_x.view(n, 1, 1)
            norm_w = g_w.reshape(n, -1).norm(dim=-1).clamp(min=1e-12)
            scale_w = (guide.grad_clip_norm / norm_w).clamp(max=1.0)
            g_w = g_w * scale_w.view(n, 1)

            # 更新（米制 -> 归一化）
            x0_m_new = x0_m - guide.step_size_x * g_x
            if wait_mask is None:
                # 自动等待点：以当前 x0 的静态净空与其他机器人路径占用为判据
                if obstacles is not None and getattr(obstacles, "count", 0) > 0:
                    from fhd.geometry.sdf import analytic_sdf

                    sdf = analytic_sdf(x0_m.detach().reshape(-1, 2), obstacles,
                                       float(cfg["world"]["robot_radius_m"]),
                                       float(cfg["world"]["clearance_margin_m"]))
                    clearance_ok = (sdf >= 0).reshape(n, h)
                else:
                    clearance_ok = torch.ones(n, h, dtype=torch.bool)
                wait_mask = auto_wait_mask(x0_m.detach(), d_safe, clearance_ok,
                                           mode=guide.wait_mask_mode)
            wait = project_wait_local(wait_local.detach() - guide.step_size_w * g_w,
                                      wait_mask, guide.max_wait_per_point, guide.max_total_wait)
            x0 = x0_m_new / scale - 1.0
            x0 = project_endpoints(x0, start_n, goal_n)

            record = {
                "ddim_step": i, "t": int(t), "guided": True,
                "soft_risk": soft_risk_value,
                "finish_cost_s": finish_value,
                "loss": float(loss.detach()),
                "grad_norm_x": float(g_x.reshape(n, -1).norm()),
                "grad_norm_w": float(g_w.norm()),
                "wait_total": float(wait.sum()),
            }
            record.update({f"term/{k}": float(v.detach()) for k, v in weights.items()})
        else:
            x0_m_new = (x0.detach() + 1.0) * scale
            record = {"ddim_step": i, "t": int(t), "guided": False,
                      "wait_total": float(wait.sum())}

        if guide.hard_eval_every and (i % guide.hard_eval_every == 0 or i == len(ts) - 1):
            with torch.no_grad():
                hard = risk_model(x0_m_new.detach(), wait.detach(), pause_p, time_steps,
                                  collision_mode="hard", gate_mode="hard",
                                  return_diagnostics=False)
                record["hard_risk"] = float(hard["pair_sum_risk"])
                record["soft_risk_eval"] = float(
                    risk_model(x0_m_new.detach(), wait.detach(), pause_p, time_steps,
                               collision_mode="soft", gate_mode="soft",
                               return_diagnostics=False)["pair_sum_risk"]
                )
        history.append(record)

        # 一致性重算 epsilon 并做 DDIM 步
        t_next = ts[i + 1].item() if i + 1 < len(ts) else -1
        with torch.no_grad():
            x = schedule.ddim_step(x, x0, int(t), int(t_next))

    final_paths = (x + 1.0) * scale
    with torch.no_grad():
        hard = risk_model(final_paths, wait, pause_p, time_steps, collision_mode="hard",
                          gate_mode="hard", return_diagnostics=False)
        soft = risk_model(final_paths, wait, pause_p, time_steps, collision_mode="soft",
                          gate_mode="soft", return_diagnostics=False)
        final = {
            "hard_risk_sum": float(hard["pair_sum_risk"]),
            "hard_risk_max": float(hard["pair_risks"].max()) if hard["pair_risks"].numel() else 0.0,
            "soft_risk_sum": float(soft["pair_sum_risk"]),
            "finish_cost_s": float(hard["finish_cost"]),
            "wait_total": float(wait.sum()),
            "guided_steps": n_guided,
            "total_steps": len(ts),
        }
    return GuidanceResult(paths_m=final_paths, wait=wait, history=history, final=final)
