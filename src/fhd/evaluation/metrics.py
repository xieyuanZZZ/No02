"""评测指标（文档 §10.2 的严格口径）。

“全尝试安全完成”要求同时满足：
    规划按时完成 AND 起终点正确 AND 静态几何连续无碰撞 AND 运动约束满足
    AND 执行在时间上限内全部到达 AND 执行全程无碰撞
规划失败的任务，其每一次执行复现都记为失败（不允许只统计成功规划）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fhd.geometry.collisions import check_speed_limit, path_segment_lengths, trajectory_is_static_safe
from fhd.geometry.paths import path_length


@dataclass
class PlanMetrics:
    static_safe: bool
    min_clearance_m: float
    speed_ok: bool
    max_segment_m: float
    endpoint_err_m: float
    path_length_mean_m: float
    smoothness: float
    valid: bool
    wait_total: float
    wait_points: int


@dataclass
class ExecutionMetrics:
    n_repeats: int
    collision_rate: float
    safe_completion_rate: float
    completion_rate: float
    makespan_mean_s: float
    makespan_p50_s: float
    makespan_p90_s: float
    unfinished_agent_rate: float


def plan_metrics(
    paths_m: torch.Tensor,
    wait: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    world,
    max_speed_mps: float,
    dt_s: float,
    endpoint_tol: float = 1e-3,
) -> PlanMetrics:
    safe, clearance = trajectory_is_static_safe(
        paths_m, world.obstacles(), world.robot_radius_m, world.clearance_margin_m)
    speed_ok, max_seg = check_speed_limit(paths_m, max_speed_mps, dt_s)
    err = max(
        float(torch.linalg.vector_norm(paths_m[:, 0] - starts, dim=-1).max()),
        float(torch.linalg.vector_norm(paths_m[:, -1] - goals, dim=-1).max()),
    )
    accel = paths_m[:, 2:] - 2 * paths_m[:, 1:-1] + paths_m[:, :-2]
    smooth = float((accel ** 2).sum(-1).mean())
    lengths = path_length(paths_m, )
    return PlanMetrics(
        static_safe=bool(safe.all()),
        min_clearance_m=float(clearance.min()),
        speed_ok=bool(speed_ok.all()),
        max_segment_m=float(max_seg.max()),
        endpoint_err_m=err,
        path_length_mean_m=float(torch.as_tensor(lengths).mean()),
        smoothness=smooth,
        valid=bool(safe.all() and speed_ok.all() and err <= endpoint_tol),
        wait_total=float(wait.sum()),
        wait_points=int((wait > 1e-6).sum()),
    )


def execution_metrics(batch, dt_s: float) -> ExecutionMetrics:
    ms = batch.makespan_steps.to(torch.float64) * dt_s
    return ExecutionMetrics(
        n_repeats=batch.n_samples,
        collision_rate=batch.collision_rate(),
        safe_completion_rate=float((batch.all_arrived & (~batch.collision)).to(torch.float64).mean()),
        completion_rate=batch.completion_rate(),
        makespan_mean_s=float(ms.mean()),
        makespan_p50_s=float(ms.median()),
        makespan_p90_s=float(torch.quantile(ms, 0.9)),
        unfinished_agent_rate=float(1.0 - (batch.completion_steps >= 0).to(torch.float64).mean()),
    )


FAILURE_CATEGORIES = (
    "ok", "static_collision", "speed_violation", "endpoint_error",
    "inter_robot_collision", "unfinished", "infeasible_task", "numeric_error",
)


def attribute_failure(plan: PlanMetrics, exec_metrics: ExecutionMetrics | None) -> str:
    if not plan.static_safe:
        return "static_collision"
    if not plan.speed_ok:
        return "speed_violation"
    if plan.endpoint_err_m > 1e-3:
        return "endpoint_error"
    if exec_metrics is None:
        return "ok"
    if exec_metrics.unfinished_agent_rate > 0:
        return "unfinished"
    if exec_metrics.collision_rate > 0:
        return "inter_robot_collision"
    return "ok"
