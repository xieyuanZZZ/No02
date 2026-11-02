"""独立暴力参考实现（**故意不复用主代码**）。

仅用于单元测试中交叉验证 DP 与 Monte Carlo：状态推进、碰撞检测、概率传播
全部用最朴素的 Python 循环 + 加密采样实现，确保主代码的错误不会在这里被复制。
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _seg_min_dist(a, b, c, d, n_samples: int = 4001) -> float:
    """两条线性运动段之间的最小距离，用加密采样（不做解析推导）。"""
    best = float("inf")
    for k in range(n_samples):
        u = k / (n_samples - 1)
        pi_x = a[0] + u * (b[0] - a[0])
        pi_y = a[1] + u * (b[1] - a[1])
        pj_x = c[0] + u * (d[0] - c[0])
        pj_y = c[1] + u * (d[1] - c[1])
        dx, dy = pi_x - pj_x, pi_y - pj_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < best:
            best = dist
    return best


def brute_force_first_hit(
    qi: Iterable[Iterable[float]],
    qj: Iterable[Iterable[float]],
    wait_i: Iterable[float],
    wait_j: Iterable[float],
    p_i: float,
    p_j: float,
    time_steps: int,
    d_safe: float,
    gate_mode: str = "hard",
    beta: float = 0.25,
    offset: float = 0.5,
    n_samples: int = 2001,
) -> float:
    """显式枚举联合状态与四种动作的首次碰撞概率。"""
    qi = [tuple(map(float, q)) for q in qi]
    qj = [tuple(map(float, q)) for q in qj]
    h = len(qi)
    assert len(qj) == h

    tau_i = []
    acc = 0.0
    for m, w in enumerate(wait_i):
        acc += float(w)
        tau_i.append(m + acc)
    tau_j = []
    acc = 0.0
    for m, w in enumerate(wait_j):
        acc += float(w)
        tau_j.append(m + acc)

    def move_prob(tau: list[float], p: float, t: int, m: int) -> float:
        if m >= h - 1:
            return 0.0
        if gate_mode == "hard":
            gate = 1.0 if t >= tau[m] else 0.0
        else:
            z = (t + 0.5 + offset - tau[m]) / beta
            gate = 1.0 / (1.0 + math.exp(-z))
        return (1.0 - p) * gate

    # 预计算四种动作在联合状态 (m,n) 上的碰撞指示
    collide: dict[tuple[int, int, int], bool] = {}
    for m in range(h):
        for n in range(h):
            a = qi[m]
            ai = qi[m + 1] if m + 1 < h else qi[m]
            c = qj[n]
            cj = qj[n + 1] if n + 1 < h else qj[n]
            for action in range(4):
                if action == 0:
                    bb, dd = a, c
                elif action == 1:
                    bb, dd = ai, c
                elif action == 2:
                    bb, dd = a, cj
                else:
                    bb, dd = ai, cj
                d = _seg_min_dist(a, bb, c, dd, n_samples=n_samples)
                collide[(m, n, action)] = d < d_safe

    f: dict[tuple[int, int], float] = {(0, 0): 1.0}
    for t in range(time_steps):
        nxt: dict[tuple[int, int], float] = {}
        for (m, n), prob in f.items():
            if prob == 0.0:
                continue
            ai = move_prob(tau_i, p_i, t, m)
            aj = move_prob(tau_j, p_j, t, n)
            moves = (
                (0, 0, 0, (1 - ai) * (1 - aj)),
                (1, 0, 1, ai * (1 - aj)),
                (0, 1, 2, (1 - ai) * aj),
                (1, 1, 3, ai * aj),
            )
            for dm, dn, action, pu in moves:
                if pu == 0.0:
                    continue
                if collide[(m, n, action)]:
                    continue
                key = (min(m + dm, h - 1), min(n + dn, h - 1))
                nxt[key] = nxt.get(key, 0.0) + prob * pu
        f = nxt
        if not f:
            return 1.0
    return 1.0 - sum(f.values())


def brute_force_progress_pmf(
    wait: Iterable[float],
    p: float,
    time_steps: int,
    h: int,
    gate_mode: str = "hard",
    beta: float = 0.25,
    offset: float = 0.5,
) -> list[list[float]]:
    """单机器人进度分布 [T+1][H]，显式枚举。"""
    tau = []
    acc = 0.0
    for m, w in enumerate(wait):
        acc += float(w)
        tau.append(m + acc)

    def move_prob(t: int, m: int) -> float:
        if m >= h - 1:
            return 0.0
        if gate_mode == "hard":
            gate = 1.0 if t >= tau[m] else 0.0
        else:
            z = (t + 0.5 + offset - tau[m]) / beta
            gate = 1.0 / (1.0 + math.exp(-z))
        return (1.0 - p) * gate

    dist = [0.0] * h
    dist[0] = 1.0
    out = [list(dist)]
    for t in range(time_steps):
        nxt = [0.0] * h
        for m in range(h):
            if dist[m] == 0.0:
                continue
            a = move_prob(t, m)
            nxt[m] += dist[m] * (1 - a)
            nxt[min(m + 1, h - 1)] += dist[m] * a
        dist = nxt
        out.append(list(dist))
    return out


def brute_force_collision_frequency(
    qi, qj, wait_i, wait_j, p_i, p_j, time_steps, d_safe, n_samples: int, d_min_table=None
) -> float:
    """朴素 Monte Carlo：逐条轨迹推进（Python 循环），返回碰撞频率。"""
    import random

    qi = [tuple(map(float, q)) for q in qi]
    qj = [tuple(map(float, q)) for q in qj]
    h = len(qi)
    tau_i = []
    acc = 0.0
    for m, w in enumerate(wait_i):
        acc += float(w)
        tau_i.append(m + acc)
    tau_j = []
    acc = 0.0
    for m, w in enumerate(wait_j):
        acc += float(w)
        tau_j.append(m + acc)

    hits = 0
    for _ in range(n_samples):
        zi = zj = 0
        hit = False
        for t in range(time_steps):
            if zi < h - 1 and t >= tau_i[zi] and random.random() >= p_i:
                zi_new = zi + 1
            else:
                zi_new = zi
            if zj < h - 1 and t >= tau_j[zj] and random.random() >= p_j:
                zj_new = zj + 1
            else:
                zj_new = zj
            if d_min_table is not None:
                d = d_min_table[(zi, zj, zi_new != zi, zj_new != zj)]
            else:
                d = _seg_min_dist(qi[zi], qi[zi_new], qj[zj], qj[zj_new], n_samples=n_samples)
            if d < d_safe:
                hit = True
                break
            zi, zj = zi_new, zj_new
        hits += int(hit)
    return hits / n_samples
