"""几何路径 Q 与主动等待 w 的联合梯度优化（文档 §7.1，S2 的验证工具）。

关键约束（批复 §21）：
  * 单点约束 0 <= w[i,m] <= max_wait_per_point；
  * 总预算   sum_m w[i,m] <= max_total_wait；
  * 等待只能加在被 wait_mask 允许的路点上（S1/S2 手动指定，S4 自动检测）；
  * 端点坐标固定：每步结束后把起终点写回条件值，并屏蔽端点梯度。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from fhd.execution.simulator import simulate_execution
from fhd.risk.multi_pair import FirstHitRisk
from fhd.risk.objectives import guidance_loss, relative_proportions


def project_wait(
    wait: torch.Tensor,
    mask: torch.Tensor,
    max_per_point: float,
    max_total: float,
) -> torch.Tensor:
    """把等待投影回可行域：掩码 -> 单点上界 -> 总量按比例缩放。"""
    with torch.no_grad():
        wait.mul_(mask.to(wait.dtype))
        wait.clamp_(min=0.0, max=max_per_point)
        totals = wait.sum(dim=-1, keepdim=True)
        scale = torch.clamp(max_total / torch.clamp(totals, min=1e-12), max=1.0)
        wait.mul_(scale)
        wait.clamp_(min=0.0, max=max_per_point)
    return wait


def mask_endpoint_grad(paths: torch.Tensor) -> None:
    """端点梯度置零（端点由条件固定，不参与优化）。"""
    if paths.grad is None:
        return
    paths.grad[:, 0, :] = 0.0
    paths.grad[:, -1, :] = 0.0


@dataclass
class OptimConfig:
    mode: str = "joint"          # path | wait | joint | none
    steps: int = 300
    lr_path: float = 0.02
    lr_wait: float = 0.1
    max_wait_per_point: float = 4.0
    max_total_wait: float = 8.0
    grad_clip_norm: float = 1.0
    eval_every: int = 10
    mc_samples: int = 2000
    mc_every: int = 50
    temporal_weight: float = 1.0   # lambda_T 缩放到 J_finish 的量纲（秒）
    verbose: bool = False


@dataclass
class OptimResult:
    paths: torch.Tensor
    wait: torch.Tensor
    history: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    wall_time_s: float = 0.0

    def metric_series(self, key: str) -> list[float]:
        return [row[key] for row in self.history if key in row]


def evaluate_policy(
    paths: torch.Tensor,
    wait: torch.Tensor,
    pause_p: torch.Tensor,
    model: FirstHitRisk,
    d_safe: float,
    time_steps: int,
    dt_s: float,
    mc_samples: int = 0,
    mc_seed: int = 0,
) -> dict:
    """用硬 DP（精确概率）与（可选）Monte Carlo 独立评估当前策略。"""
    with torch.no_grad():
        hard = model(paths, wait, pause_p, time_steps, collision_mode="hard", gate_mode="hard",
                     return_diagnostics=False)
        soft = model(paths, wait, pause_p, time_steps, collision_mode="soft", gate_mode="soft",
                     return_diagnostics=False)
        out = {
            "hard_risk_sum": float(hard["pair_sum_risk"]),
            "hard_risk_max": float(hard["pair_risks"].max()) if hard["pair_risks"].numel() else 0.0,
            "soft_risk_sum": float(soft["pair_sum_risk"]),
            "finish_cost_s": float(hard["finish_cost"]),
            "unfinished_prob": float(hard["unfinished_prob"].mean()),
            "wait_total": float(wait.sum()),
        }
        if mc_samples > 0:
            mc = simulate_execution(
                paths, wait, pause_p, d_safe, time_steps, dt_s,
                n_samples=mc_samples, chunk_size=min(mc_samples, 5000),
                generator=torch.Generator().manual_seed(mc_seed),
            )
            out.update({
                "mc_collision_rate": mc.collision_rate(),
                "mc_makespan_s": float(mc.makespan_steps.to(torch.float64).mean()) * dt_s,
                "mc_completion_rate": mc.completion_rate(),
            })
    return out


def optimize_path_wait(
    paths0: torch.Tensor,
    wait0: torch.Tensor,
    wait_mask: torch.Tensor,
    pause_p: torch.Tensor,
    model: FirstHitRisk,
    cfg,
    obstacles=None,
    opt: OptimConfig | None = None,
    d_safe: float = 0.35,
    time_steps: int = 96,
    dt_s: float = 0.4,
    callback: Callable[[int, dict], None] | None = None,
) -> OptimResult:
    """对 (Q, w) 做投影梯度优化，返回历史曲线与最终策略。"""
    opt = opt or OptimConfig()
    device, dtype = paths0.device, paths0.dtype
    paths = paths0.detach().clone().to(dtype=dtype).requires_grad_(True)
    wait = wait0.detach().clone().to(dtype=dtype).requires_grad_(True)
    endpoints = paths0.detach().clone()[:, [0, -1], :]
    mask = wait_mask.to(device=device)

    opt_mode = opt.mode
    params = []
    if opt_mode in ("path", "joint"):
        params.append(paths)
    if opt_mode in ("wait", "joint"):
        params.append(wait)
    if not params:
        raise ValueError(f"优化模式 {opt_mode} 没有可优化变量")

    optimizer = torch.optim.Adam(params, lr=opt.lr_path if opt_mode == "path" else opt.lr_wait)
    if opt_mode == "joint":
        optimizer = torch.optim.Adam(
            [
                {"params": [paths], "lr": opt.lr_path},
                {"params": [wait], "lr": opt.lr_wait},
            ]
        )

    history: list[dict] = []
    started = time.time()
    for step in range(opt.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        risk_result = model(paths, wait, pause_p, time_steps,
                            collision_mode="soft", gate_mode="soft", return_diagnostics=False)
        loss, detail = guidance_loss(paths, wait, risk_result, cfg, obstacles=obstacles)

        if step % opt.eval_every == 0 or step == opt.steps:
            with torch.no_grad():
                row = {
                    "step": step,
                    "loss": float(loss),
                    "soft_risk_sum": float(risk_result["pair_sum_risk"]),
                    "finish_cost_s": float(risk_result["finish_cost"]),
                    "wait_total": float(wait.sum()),
                    "path_length_mean": float(
                        torch.linalg.vector_norm(paths[:, 1:] - paths[:, :-1], dim=-1).sum(-1).mean()
                    ),
                }
                row.update({f"term/{k}": float(v) for k, v in detail["weighted"].items()})
                row.update({f"raw/{k}": float(v) for k, v in detail["raw"].items() if v.numel() == 1})
                hard = model(paths, wait, pause_p, time_steps, collision_mode="hard",
                             gate_mode="hard", return_diagnostics=False)
                row["hard_risk_sum"] = float(hard["pair_sum_risk"])
                if opt.mc_samples > 0 and (step % opt.mc_every == 0 or step == opt.steps):
                    ev = evaluate_policy(paths, wait, pause_p, model, d_safe, time_steps, dt_s,
                                         mc_samples=opt.mc_samples, mc_seed=step)
                    row["mc_collision_rate"] = ev["mc_collision_rate"]
                    row["mc_makespan_s"] = ev["mc_makespan_s"]
                row["proportions"] = relative_proportions(detail)
            history.append(row)
            if callback is not None:
                callback(step, row)
            if opt.verbose:
                print(f"step {step:4d} loss={row['loss']:.4f} soft={row['soft_risk_sum']:.4f} "
                      f"hard={row['hard_risk_sum']:.4f} wait={row['wait_total']:.2f} "
                      f"finish={row['finish_cost_s']:.2f}s")

        if step == opt.steps:
            break

        opt_params = []
        if opt_mode in ("path", "joint"):
            opt_params.append(paths)
        if opt_mode in ("wait", "joint"):
            opt_params.append(wait)
        grads = torch.autograd.grad(loss, opt_params, allow_unused=True)
        for p, grad in zip(opt_params, grads):
            p.grad = grad
        if opt_mode in ("path", "joint"):
            mask_endpoint_grad(paths)
            if opt.grad_clip_norm > 0 and paths.grad is not None:
                torch.nn.utils.clip_grad_norm_([paths], opt.grad_clip_norm)
        if opt_mode in ("wait", "joint") and opt.grad_clip_norm > 0 and wait.grad is not None:
            torch.nn.utils.clip_grad_norm_([wait], opt.grad_clip_norm)
        optimizer.step()

        # 端点固定 + 等待投影
        with torch.no_grad():
            paths[:, 0, :] = endpoints[:, 0, :]
            paths[:, -1, :] = endpoints[:, 1, :]
        project_wait(wait, mask, opt.max_wait_per_point, opt.max_total_wait)

    final = evaluate_policy(paths.detach(), wait.detach(), pause_p, model, d_safe, time_steps,
                            dt_s, mc_samples=max(opt.mc_samples, 2000), mc_seed=999)
    return OptimResult(paths=paths.detach(), wait=wait.detach(), history=history, final=final,
                       wall_time_s=time.time() - started)


def check_wait_safety(
    paths: torch.Tensor,
    wait: torch.Tensor,
    d_safe: float,
    threshold: float = 1e-6,
) -> list[dict]:
    """检查“主动等待是否发生在危险位置”（文档 §9 S2 通过条件）。

    对每个 w>0 的等待点，计算该点到其他机器人路径（折线）的最小距离；
    若小于 d_safe，说明机器人会停在一个别人可能经过的位置。
    """
    from fhd.geometry.primitives import segment_point_distance

    n, h, _ = paths.shape
    rows = []
    for i in range(n):
        idxs = torch.nonzero(wait[i] > threshold, as_tuple=False).flatten()
        for m in idxs:
            q = paths[i, m]
            best = float("inf")
            for j in range(n):
                if j == i:
                    continue
                a = paths[j, :-1]  # [H-1,2]
                b = paths[j, 1:]
                d = segment_point_distance(a, b, q.unsqueeze(0).expand(h - 1, 2))
                best = min(best, float(d.min()))
            rows.append({
                "agent": i, "progress_index": int(m), "wait_slots": float(wait[i, m]),
                "min_distance_other_path_m": best,
                "dangerous": bool(best < d_safe),
            })
    return rows


__all__ = [
    "OptimConfig",
    "OptimResult",
    "optimize_path_wait",
    "project_wait",
    "mask_endpoint_grad",
    "evaluate_policy",
    "check_wait_safety",
]
