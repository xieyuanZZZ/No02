"""引导目标 L(Q,w)（文档 §7.1）与其中的各项代价。

    L = λ_R ΣR_soft + λ_T J_finish + λ_S L_static + λ_K L_kinematic
        + λ_G L_geometry + λ_W ||w||_1

每一项都单独返回并可单独记录（批复 §23 要求分项数值/梯度/占比全部记录）。
"""

from __future__ import annotations

import torch

from fhd.geometry.collisions import path_segment_lengths
from fhd.geometry.sdf import sampled_static_penalty


def kinematic_cost(
    paths: torch.Tensor,
    max_speed_mps: float,
    dt_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """速度约束违反的平方惩罚 + 二阶差分平滑度惩罚。

    返回 (总代价, 逐项字典)。
    """
    lens = path_segment_lengths(paths)  # [N,H-1]
    limit = max_speed_mps * dt_s
    viol = torch.clamp(lens / limit - 1.0, min=0.0)
    speed_term = (viol * viol).mean()
    if paths.shape[1] >= 3:
        accel = paths[:, 2:, :] - 2.0 * paths[:, 1:-1, :] + paths[:, :-2, :]
        smooth_term = (accel * accel).sum(-1).mean() / (limit * limit)
    else:
        smooth_term = paths.new_zeros(())
    return speed_term + 0.1 * smooth_term, {"speed_violation": speed_term, "smoothness": smooth_term}


def geometry_cost(
    paths: torch.Tensor,
    max_speed_mps: float,
    dt_s: float,
    target_ratio: float = 0.6,
) -> torch.Tensor:
    """几何正则：把平均段长保持在速度上限的 target_ratio 附近。

    用于避免风险引导把轨迹拉得过长（段长逼近上限）或整体塌缩。
    """
    lens = path_segment_lengths(paths)
    limit = max_speed_mps * dt_s
    target = max(target_ratio * limit, 1e-6)
    ratio = lens.mean() / target - 1.0
    return ratio * ratio


def wait_l1(wait: torch.Tensor) -> torch.Tensor:
    """主动等待的 L1 稀疏项（鼓励少量局部等待，而非处处小停顿）。"""
    return wait.sum() if wait.numel() else wait.new_zeros(())


def per_time_collision_risk(
    pi_i: torch.Tensor,
    pi_j: torch.Tensor,
    collision: torch.Tensor,
    a_i: torch.Tensor,
    a_j: torch.Tensor,
    time_steps: int,
) -> torch.Tensor:
    """基线 B3：逐时刻期望碰撞量（**不做首次碰撞吸收**，文档 §8.1）。

    Σ_t Σ_{m,n} P_t(m,n) Σ_u P_u^t(m,n) C̃_{mn,u}
    该量随时间线性累积，没有概率上界，**不是**碰撞概率。

    参数
    ----
    pi_i, pi_j : [T+1,H] 单机进度分布（不含机器人间碰撞）
    collision  : [4,H,H] 与主方法完全相同的连续运动段碰撞核
    a_i, a_j   : [T,H] 前进概率
    """
    total = pi_i.new_zeros(())
    for t in range(time_steps):
        joint = pi_i[t].unsqueeze(-1) * pi_j[t].unsqueeze(-2)  # [H,H]
        ai = a_i[t].unsqueeze(-1)
        aj = a_j[t].unsqueeze(-2)
        p = torch.stack(
            [
                (1.0 - ai) * (1.0 - aj),
                ai * (1.0 - aj),
                (1.0 - ai) * aj,
                ai * aj,
            ],
            dim=0,
        )  # [4,H,H]
        total = total + (joint.unsqueeze(0) * p * collision).sum()
    return total


def sync_distance_risk(
    paths: torch.Tensor,
    d_safe: float,
    temperature_m: float,
    kernel: str = "logistic",
    support_m: float = 0.05,
) -> torch.Tensor:
    """基线 B1：只惩罚同一名义步的两机器人距离（同步距离引导）。"""
    from fhd.geometry.collisions import soft_collision

    n, h, _ = paths.shape
    total = paths.new_zeros(())
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.sqrt(((paths[i] - paths[j]) ** 2).sum(-1) + 1e-16)  # [H]
            total = total + soft_collision(d, d_safe, temperature_m, kernel, support_m).sum()
    return total


def window_risk(
    paths: torch.Tensor,
    d_safe: float,
    temperature_m: float,
    window: int = 3,
    kernel: str = "logistic",
    support_m: float = 0.05,
) -> torch.Tensor:
    """基线 B2：固定时间窗引导，对 |m-n| <= k 的联合状态取窗口内最大软碰撞（批复 §23：max 聚合）。"""
    from fhd.geometry.collisions import pair_action_min_distances, soft_collision

    n, h, _ = paths.shape
    idx = torch.arange(h)
    band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= window  # [H,H]
    total = paths.new_zeros(())
    for i in range(n):
        for j in range(i + 1, n):
            d = pair_action_min_distances(paths[i : i + 1], paths[j : j + 1]).d_min[0]  # [4,H,H]
            c = soft_collision(d, d_safe, temperature_m, kernel, support_m).amax(dim=0)  # [H,H]
            total = total + (c * band).sum()
    return total


def guidance_loss(
    paths: torch.Tensor,
    wait: torch.Tensor,
    risk_result: dict,
    cfg,
    obstacles=None,
    use_risk: bool = True,
) -> tuple[torch.Tensor, dict]:
    """总引导目标与分项明细（全部保留计算图）。"""
    g = cfg["guidance"]
    world = cfg["world"]
    traj = cfg["trajectory"]

    comps: dict[str, torch.Tensor] = {}
    comps["risk"] = risk_result["pair_sum_risk"] if use_risk else paths.new_zeros(())
    comps["finish"] = risk_result["finish_cost"]

    if obstacles is not None and getattr(obstacles, "count", 0) > 0:
        comps["static"] = sampled_static_penalty(
            paths,
            obstacles,
            robot_radius=float(world["robot_radius_m"]),
            clearance_margin=float(world["clearance_margin_m"]),
        )
    else:
        comps["static"] = paths.new_zeros(())

    kin_total, kin_parts = kinematic_cost(
        paths, float(traj["max_speed_mps"]), float(traj["dt_s"])
    )
    comps["kinematic"] = kin_total
    comps.update({f"kinematic/{k}": v for k, v in kin_parts.items()})
    comps["geometry"] = geometry_cost(paths, float(traj["max_speed_mps"]), float(traj["dt_s"]))
    comps["wait_l1"] = wait_l1(wait)

    weights = {
        "risk": float(g["lambda_R"]),
        "finish": float(g["lambda_T"]),
        "static": float(g["lambda_S"]),
        "kinematic": float(g["lambda_K"]),
        "geometry": float(g["lambda_G"]),
        "wait_l1": float(g["lambda_W"]),
    }
    total = paths.new_zeros(())
    weighted: dict[str, torch.Tensor] = {}
    for name, weight in weights.items():
        term = weight * comps[name]
        weighted[name] = term
        total = total + term

    detail = {
        "total": total,
        "weighted": weighted,
        "raw": comps,
    }
    return total, detail


def relative_proportions(detail: dict) -> dict[str, float]:
    """各加权项相对于总损失的占比（绝对值），用于日志与图。"""
    with torch.no_grad():
        total = abs(float(detail["total"]))
        if total <= 1e-12:
            return {k: 0.0 for k in detail["weighted"]}
        return {k: abs(float(v)) / total for k, v in detail["weighted"].items()}


__all__ = [
    "kinematic_cost",
    "geometry_cost",
    "wait_l1",
    "per_time_collision_risk",
    "sync_distance_risk",
    "window_risk",
    "guidance_loss",
    "relative_proportions",
]
