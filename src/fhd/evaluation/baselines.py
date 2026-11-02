"""基线与消融方法（文档 §8）：B0 / B1 / B2 / B3 / A1 / A2 / A3。

所有方法共享：
  * 同一个单机器人 Diffusion 权重；
  * 同一组地图、起终点、候选噪声、DDIM 步数与引导步数；
  * **完全相同的后四项**（J_finish、静态 SDF、运动学、几何正则、||w||₁）；
只有“机器人间风险项”被替换（B0 直接去掉）。

因此方法之间的差异只来自风险项本身。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from fhd.diffusion.guided_ddim import GuidanceConfig, guided_ddim_sample
from fhd.execution.progress import move_probabilities, propagate_progress, release_times
from fhd.risk.multi_pair import FirstHitRisk
from fhd.risk.objectives import (
    geometry_cost,
    kinematic_cost,
    per_time_collision_risk,
    relative_proportions,
    sync_distance_risk,
    wait_l1,
    window_risk,
)
from fhd.geometry.sdf import sampled_static_penalty


@dataclass
class MethodSpec:
    key: str
    label: str
    description: str
    optimize_path: bool = True
    optimize_wait: bool = False
    risk_kind: str = "first_hit"   # first_hit | none | sync | window | per_time
    window_k: int = 3


METHODS = {
    "B0": MethodSpec("B0", "独立 Diffusion", "各自采样，只含静态/运动学约束引导，不含机器人间风险",
                     optimize_path=True, optimize_wait=False, risk_kind="none"),
    "B1": MethodSpec("B1", "同步距离引导", "只惩罚同一名义步的两机器人距离",
                     optimize_path=True, optimize_wait=False, risk_kind="sync"),
    "B2": MethodSpec("B2", "固定时间窗引导", "对 |m-n|<=k 的危险状态施加约束（k=3）",
                     optimize_path=True, optimize_wait=False, risk_kind="window", window_k=3),
    "B3": MethodSpec("B3", "逐时刻期望碰撞风险", "单时刻联合进度概率的碰撞量求和，不做首次吸收",
                     optimize_path=True, optimize_wait=True, risk_kind="per_time"),
    "A1": MethodSpec("A1", "首次碰撞 + 只改路径", "w=0 固定，仅 ∇_Q R_soft",
                     optimize_path=True, optimize_wait=False, risk_kind="first_hit"),
    "A2": MethodSpec("A2", "首次碰撞 + 只改等待", "Q 固定，仅 ∇_w R_soft",
                     optimize_path=False, optimize_wait=True, risk_kind="first_hit"),
    "A3": MethodSpec("A3", "首次碰撞 + 联合", "空间与等待梯度共同作用（完整方法）",
                     optimize_path=True, optimize_wait=True, risk_kind="first_hit"),
}


def build_objective(
    method: MethodSpec,
    cfg,
    risk_model: FirstHitRisk,
    pause_p: torch.Tensor,
    time_steps: int,
    dt_s: float,
    obstacles=None,
    d_safe: float = 0.35,
) -> Callable:
    """构造方法对应的引导目标 L(Q,w)。

    返回值签名：(paths_m, wait) -> (loss, detail)
    """
    g = cfg["guidance"]
    world = cfg["world"]
    traj = cfg["trajectory"]
    beta = float(cfg["risk"]["gate_beta"])
    offset = float(cfg["risk"]["gate_offset_c"])
    eps = float(cfg["risk"]["soft_collision_temperature_m"])
    kernel = str(cfg["risk"]["soft_collision_kernel"])
    support = float(cfg["risk"]["smoothstep_support_m"])

    weights = {
        "risk": float(g["lambda_R"]),
        "finish": float(g["lambda_T"]),
        "static": float(g["lambda_S"]),
        "kinematic": float(g["lambda_K"]),
        "geometry": float(g["lambda_G"]),
        "wait_l1": float(g["lambda_W"]),
    }

    def risk_term(paths_m: torch.Tensor, wait_m: torch.Tensor) -> tuple[torch.Tensor, dict]:
        kind = method.risk_kind
        if kind == "none":
            return paths_m.new_zeros(()), {}
        if kind == "first_hit":
            out = risk_model(paths_m, wait_m, pause_p, time_steps, collision_mode="soft",
                             gate_mode="soft", return_diagnostics=False)
            return out["pair_sum_risk"], {"finish_cost": out["finish_cost"]}
        if kind == "sync":
            return sync_distance_risk(paths_m, d_safe, eps, kernel, support), {}
        if kind == "window":
            return window_risk(paths_m, d_safe, eps, window=method.window_k, kernel=kernel,
                               support_m=support), {}
        if kind == "per_time":
            tau = release_times(wait_m)
            a = move_probabilities(tau, pause_p, time_steps, beta=beta, offset=offset,
                                   gate_mode="soft")
            pi = propagate_progress(a)
            ii, jj, _, collision = risk_model.pair_collision_tensor(paths_m, "soft")
            total = paths_m.new_zeros(())
            for p in range(ii.numel()):
                total = total + per_time_collision_risk(
                    pi[ii[p]], pi[jj[p]], collision[p], a[ii[p]], a[jj[p]], time_steps
                )
            return total, {}
        raise ValueError(f"未知风险类型: {kind}")

    def objective(paths_m: torch.Tensor, wait_m: torch.Tensor) -> tuple[torch.Tensor, dict]:
        term, extra = risk_term(paths_m, wait_m)
        # J_finish 必须与主方法同源（软门槛单机进度传播）
        from fhd.execution.progress import finish_cost

        tau = release_times(wait_m)
        a = move_probabilities(tau, pause_p, time_steps, beta=beta, offset=offset, gate_mode="soft")
        pi = propagate_progress(a)
        j_finish, _ = finish_cost(pi, dt_s=dt_s, lambda_tail_s=float(g["lambda_tail_s"]))

        if obstacles is not None and getattr(obstacles, "count", 0) > 0:
            static = sampled_static_penalty(
                paths_m, obstacles,
                robot_radius=float(world["robot_radius_m"]),
                clearance_margin=float(world["clearance_margin_m"]),
            )
        else:
            static = paths_m.new_zeros(())
        kin, kin_parts = kinematic_cost(paths_m, float(traj["max_speed_mps"]), float(traj["dt_s"]))
        geom = geometry_cost(paths_m, float(traj["max_speed_mps"]), float(traj["dt_s"]))
        w1 = wait_l1(wait_m)

        raw = {"risk": term, "finish": j_finish, "static": static, "kinematic": kin,
               "geometry": geom, "wait_l1": w1}
        weighted = {k: weights[k] * v for k, v in raw.items()}
        total = sum(weighted.values())
        detail = {"total": total, "weighted": weighted, "raw": raw,
                  "soft_risk": term, "finish_cost": j_finish,
                  "proportions": relative_proportions({"total": total, "weighted": weighted})}
        return total, detail

    return objective


@dataclass
class PlanResult:
    method: str
    paths_m: torch.Tensor
    wait: torch.Tensor
    final: dict
    history: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0


def plan_with_method(
    method_key: str,
    model,
    schedule,
    risk_model,
    cfg,
    start_m: torch.Tensor,
    goal_m: torch.Tensor,
    grid: torch.Tensor,
    pause_p: torch.Tensor,
    obstacles,
    num_steps: int = 40,
    generator: torch.Generator | None = None,
    x_init: torch.Tensor | None = None,
    d_safe: float = 0.35,
    time_steps: int = 96,
    dt_s: float = 0.4,
    step_size_x: float | None = None,
    step_size_w: float | None = None,
    wait_mask: torch.Tensor | None = None,
    clearance_ok: torch.Tensor | None = None,
) -> PlanResult:
    """用指定方法完成一次多机器人规划。"""
    import time

    spec = METHODS[method_key]
    objective = build_objective(spec, cfg, risk_model, pause_p, time_steps, dt_s,
                                obstacles=obstacles, d_safe=d_safe)
    guide = GuidanceConfig(
        enabled=spec.optimize_path or spec.optimize_wait,
        start_fraction=float(cfg["diffusion"]["guidance_start_fraction"]),
        step_size_x=float(step_size_x if step_size_x is not None
                          else cfg["guidance"].get("guidance_step_x", 0.05)),
        step_size_w=float(step_size_w if step_size_w is not None
                          else cfg["guidance"].get("guidance_step_w", 0.1)),
        grad_clip_norm=float(cfg["guidance"]["grad_clip_norm"]),
        max_wait_per_point=float(cfg["wait"]["max_wait_per_point"]),
        max_total_wait=float(cfg["wait"]["max_total_wait"]),
        wait_mask=wait_mask,
        wait_mask_mode="manual" if wait_mask is not None else "auto",
        optimize_path=spec.optimize_path,
        optimize_wait=spec.optimize_wait,
        hard_eval_every=5,
    )
    t0 = time.time()
    result = guided_ddim_sample(
        model, schedule, risk_model, cfg,
        start_m=start_m, goal_m=goal_m, grid=grid, pause_p=pause_p,
        obstacles=obstacles, num_steps=num_steps, generator=generator,
        guide=guide, time_steps=time_steps, dt_s=dt_s, d_safe=d_safe,
        clearance_ok=clearance_ok, x_init=x_init, objective_fn=objective,
    )
    return PlanResult(method=method_key, paths_m=result.paths_m, wait=result.wait,
                      final=result.final, history=result.history,
                      elapsed_s=time.time() - t0)
