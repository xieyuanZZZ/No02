"""连续碰撞检测：机器人之间的相对运动最小距离、软碰撞代理、静态障碍与速度检查。

核心约定（文档 §4.1）：
    一个执行时隙内，机器人 i 从 q_{i,m} 线性移动到 q_{i,m+a}，a∈{0,1}；
    机器人 j 同理。相对位置 r(u) = r0 + u v, u∈[0,1]，
    连续最小距离 d_min = ||r0 + u* v||，u* = clip(-r0·v/(||v||²+eps), 0, 1)。
    必须检查“互换位置时中途相撞”，因此不能只比较端点。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fhd.geometry.primitives import (
    EPS_NUM,
    Obstacles,
    relative_motion_min_distance,
    safe_sqrt,
    segment_obstacle_distance,
)

# 四种动作的索引：00 双方停、10 只有 i 前进、01 只有 j 前进、11 双方都前进
ACTION_STOP_STOP = 0
ACTION_MOVE_I = 1
ACTION_MOVE_J = 2
ACTION_MOVE_BOTH = 3
ACTION_NAMES = ("00", "10", "01", "11")


@dataclass
class PairActionDistances:
    """一对机器人在所有联合进度状态上、四种动作下的连续最小距离。

    d[action, m, n] 的形状为 [4, H, H]（批量版本为 [P, 4, H, H]）。
    """

    d_min: torch.Tensor

    def action(self, index: int) -> torch.Tensor:
        return self.d_min[..., index, :, :]


def pair_action_min_distances(
    paths_i: torch.Tensor, paths_j: torch.Tensor
) -> PairActionDistances:
    """计算四种动作下的连续最小距离矩阵。

    参数
    ----
    paths_i, paths_j : [P, H, 2] 同一批机器人对的几何路径（批维为机器人对数）。

    返回
    ----
    PairActionDistances，其 d_min 形状为 [P, 4, H, H]。
    对 m = H-1（终点）的前进动作，位移被置零（终点不能再前进），
    对应概率质量由 a_{i,H-1}=0 保证为零。
    """
    if paths_i.dim() != 3 or paths_j.dim() != 3:
        raise ValueError("paths 必须是 [P, H, 2]")
    h = paths_i.shape[1]
    # 前进动作的终点：最后一点复制自身 -> 位移为零，避免越界索引
    end_i = torch.cat([paths_i[:, 1:, :], paths_i[:, -1:, :]], dim=1)  # [P,H,2]
    end_j = torch.cat([paths_j[:, 1:, :], paths_j[:, -1:, :]], dim=1)

    r0 = paths_i.unsqueeze(2) - paths_j.unsqueeze(1)  # [P,H,H,2]
    v_i = (end_i - paths_i).unsqueeze(2)  # [P,H,1,2]
    v_j = (end_j - paths_j).unsqueeze(1)  # [P,1,H,2]

    # 相对速度 v = d_i - d_j（r(u) = r_i(u) - r_j(u)），因此只有 j 前进时 v = -v_j
    d00 = safe_sqrt((r0 * r0).sum(-1))  # [P,H,H]
    d10, _ = relative_motion_min_distance(r0, v_i.expand_as(r0))
    d01, _ = relative_motion_min_distance(r0, (-v_j).expand_as(r0))
    d11, _ = relative_motion_min_distance(r0, (v_i - v_j).expand_as(r0))

    stacked = torch.stack([d00, d10, d01, d11], dim=1)  # [P,4,H,H]
    assert stacked.shape[-2] == h and stacked.shape[-1] == h
    return PairActionDistances(d_min=stacked)


def soft_collision(
    d_min: torch.Tensor,
    d_safe: float,
    temperature_m: float,
    kernel: str = "logistic",
    support_m: float = 0.05,
) -> torch.Tensor:
    """软碰撞代理 tC ∈ (0,1)。**不是**真实碰撞概率（文档 §4.2、§7.5）。

    logistic  : sigma((d_safe - d)/temperature)，处处非零 -> 远距离存在虚假累积风险；
    smoothstep: 有限作用距离 [d_safe, d_safe+support] 的 C1 紧支撑平滑阶跃，
                d >= d_safe+support 时严格为 0 且梯度为 0。
    """
    if kernel == "logistic":
        return torch.sigmoid((d_safe - d_min) / temperature_m)
    if kernel == "smoothstep":
        x = torch.clamp((d_safe + support_m - d_min) / support_m, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    raise ValueError(f"未知的软碰撞核: {kernel}")


def hard_collision(d_min: torch.Tensor, d_safe: float) -> torch.Tensor:
    """硬碰撞指示 d_min < d_safe（返回 0/1 张量，不可微）。"""
    return (d_min < d_safe).to(d_min.dtype)


# --------------------------------------------------------------------------- #
# 轨迹层面的静态检查（连续几何，独立于栅格与 SDF）
# --------------------------------------------------------------------------- #
def path_segment_lengths(paths: torch.Tensor) -> torch.Tensor:
    """每条路径的逐段长度。paths [N,H,2] -> [N,H-1]。"""
    return safe_sqrt(((paths[:, 1:] - paths[:, :-1]) ** 2).sum(-1))


def check_speed_limit(
    paths: torch.Tensor, max_speed_mps: float, dt_s: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐段速度约束检查（文档 §2.3：段长 <= v_max * dt）。

    返回 (ok [N] bool, 每台机器人的最大段长 [N])。
    """
    lengths = path_segment_lengths(paths)
    limit = max_speed_mps * dt_s
    ok = (lengths <= limit + 1e-9).all(dim=-1)
    return ok, lengths.max(dim=-1).values


def trajectory_obstacle_clearance(
    paths: torch.Tensor,
    obstacles: Obstacles,
    robot_radius: float,
    clearance_margin: float,
    rect_iterations: int = 25,
) -> torch.Tensor:
    """轨迹逐段到障碍物的最小净空（已扣除 机器人半径+净空）。

    paths [N,H,2] -> [N] 的最小净空（>=0 表示全程安全）。
    使用连续线段几何，不对栅格做近似。
    """
    if paths.dim() != 3:
        raise ValueError("paths 必须是 [N, H, 2]")
    n, h, _ = paths.shape
    inflate = robot_radius + clearance_margin
    a = paths[:, :-1, :].reshape(-1, 2)
    b = paths[:, 1:, :].reshape(-1, 2)
    dist = segment_obstacle_distance(a, b, obstacles, rect_iterations=rect_iterations)
    clearance = dist - inflate
    # 同时检查端点（线段已包含端点，但零长度段也要返回有限值）
    return clearance.reshape(n, h - 1).min(dim=-1).values


def trajectory_is_static_safe(
    paths: torch.Tensor,
    obstacles: Obstacles,
    robot_radius: float,
    clearance_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """轨迹静态安全布尔与最小净空。返回 (safe [N] bool, clearance [N])。"""
    clearance = trajectory_obstacle_clearance(paths, obstacles, robot_radius, clearance_margin)
    return clearance >= 0.0, clearance


def point_in_free_space(
    points: torch.Tensor,
    obstacles: Obstacles,
    robot_radius: float,
    clearance_margin: float,
) -> torch.Tensor:
    """点是否处于（按机器人半径与净空膨胀后的）自由空间。返回 [...,] bool。"""
    from fhd.geometry.primitives import point_obstacle_distance

    dist = point_obstacle_distance(points, obstacles)
    return dist >= robot_radius + clearance_margin


def d_safe_from_config(world_cfg) -> float:
    """d_safe = 2r + margin。"""
    return 2.0 * float(world_cfg["robot_radius_m"]) + float(world_cfg["safety_margin_m"])
