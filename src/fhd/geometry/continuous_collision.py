"""连续时间碰撞检测（阶段一 v2）：相对运动的最小距离与**首次越界时刻**。

对同一动力学子步区间 [t_j, t_{j+1}]：
    r(α) = r0 + α dr,  α∈[0,1]
    r0 = p_i(t_j) - p_j(t_j)
    dr = [p_i(t_{j+1}) - p_i(t_j)] - [p_j(t_{j+1}) - p_j(t_j)]

  * 最小距离：α* = clip(-r0·dr / ||dr||², 0, 1)，d_min = ||r0 + α* dr||；
  * 首次越界：解 ||r0 + α dr||² = d_safe²，取 [0,1] 内最小根 α_hit，
    首次碰撞时刻 = t_j + α_hit (t_{j+1}-t_j)。

旧实现用整个 dt_ctrl 区间的 4 个离散采样位置，可能漏掉采样点之间的
最小间距，且时间戳 `t - dt_ctrl + sub*...` 会偏差一个控制周期甚至为负。
本模块只处理"同一子步区间内两侧都做线性相对运动"的情形，正好对应
执行器按 dt_dyn 输出的子步位置序列。`dr≈0` 时按静止相对运动处理。
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def relative_min_distance(r0: np.ndarray, dr: np.ndarray) -> tuple[float, float]:
    """返回 (最小距离, 取到最小值的 α∈[0,1])。"""
    r0 = np.asarray(r0, dtype=np.float64)
    dr = np.asarray(dr, dtype=np.float64)
    a = float(dr @ dr)
    if a < _EPS:
        return float(np.linalg.norm(r0)), 0.0
    alpha = float(np.clip(-(r0 @ dr) / a, 0.0, 1.0))
    return float(np.linalg.norm(r0 + alpha * dr)), alpha


def first_breach_alpha(r0: np.ndarray, dr: np.ndarray, d_safe: float) -> float | None:
    """返回区间内首次满足 ||r0+αdr|| < d_safe 的 α；无越界返回 None。

    * 区间起点已越界 -> α=0（初始重叠必须能检测）；
    * 恰好切阈值（判别式为 0 / 最小距离==d_safe）不算碰撞；
    * dr≈0 时退化为静止点判定。
    """
    r0 = np.asarray(r0, dtype=np.float64)
    dr = np.asarray(dr, dtype=np.float64)
    d2 = float(d_safe) ** 2
    c0 = float(r0 @ r0) - d2
    if c0 < 0.0:
        return 0.0
    a = float(dr @ dr)
    if a < _EPS:
        return None
    b = 2.0 * float(r0 @ dr)
    disc = b * b - 4.0 * a * c0
    if disc <= 0.0:
        return None
    root = (-b - float(np.sqrt(disc))) / (2.0 * a)
    if root < 0.0:
        # 起点在界外、根在负半轴：极小值在区间外，不可能在区间内越界
        return None
    if root > 1.0:
        return None
    # 用距离复核（分数根可能因浮点误差落在边界外）
    if float(np.linalg.norm(r0 + root * dr)) <= float(d_safe) + 1e-9:
        return float(root)
    return None


def pair_continuous_collision(pos_i: np.ndarray, pos_j: np.ndarray,
                              times: np.ndarray, d_safe: float) -> dict:
    """一对机器人在一段时间内的首次碰撞与最小距离。

    pos_i/pos_j: [K,2] 同一时间栅格上的位置；times: [K] 严格递增。
    返回 {"collision": bool, "first_time": float|None, "min_distance": float,
          "min_alpha_time": float|None}
    """
    pi = np.asarray(pos_i, dtype=np.float64)
    pj = np.asarray(pos_j, dtype=np.float64)
    ts = np.asarray(times, dtype=np.float64)
    if pi.shape != pj.shape or pi.ndim != 2 or ts.shape != (pi.shape[0],):
        raise ValueError("pos 形状应为 [K,2]，times 形状应为 [K]")
    d_min, first_time, arg_time = np.inf, None, None
    for j in range(len(ts) - 1):
        r0 = pi[j] - pj[j]
        dr = (pi[j + 1] - pi[j]) - (pj[j + 1] - pj[j])
        d, alpha = relative_min_distance(r0, dr)
        if d < d_min:
            d_min = d
            arg_time = float(ts[j] + alpha * (ts[j + 1] - ts[j]))
        if first_time is None:
            a_hit = first_breach_alpha(r0, dr, d_safe)
            if a_hit is not None:
                first_time = float(ts[j] + a_hit * (ts[j + 1] - ts[j]))
    return {"collision": first_time is not None, "first_time": first_time,
            "min_distance": float(d_min), "min_alpha_time": arg_time}


def all_pair_events(positions: np.ndarray, times: np.ndarray, pairs: list,
                    d_safe: float) -> list[dict]:
    """positions [K,N,2] -> 每对的事件列表（顺序同 pairs）。"""
    out = []
    for (i, j) in pairs:
        ev = pair_continuous_collision(positions[:, i], positions[:, j], times, d_safe)
        ev["pair"] = (int(i), int(j))
        out.append(ev)
    return out
