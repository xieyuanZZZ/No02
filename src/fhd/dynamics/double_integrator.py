"""二维点质量双积分器（本轮唯一动力学模型）。

    x = [px, py, vx, vy]^T,   u = [ax, ay]^T
    x_{k+1} = A x_k + B u_k

建模边界：**只做平面平移**，不含航向角、差速运动学与转向约束。
因此本文件的结果不能称为"完整移动机器人动力学验证"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DoubleIntegratorParams:
    dt: float = 0.05            # 动力学仿真周期
    dt_ctrl: float = 0.1        # MPC 控制周期
    dt_slot: float = 0.4        # 上层时序决策周期（执行时隙）
    v_max: float = 1.5          # m/s
    a_max: float = 2.0          # m/s^2
    a_brake: float = 2.0        # m/s^2（制动减速度上限）


class DoubleIntegrator:
    """离散双积分器，含范数约束的**保守多边形近似**。"""

    def __init__(self, p: DoubleIntegratorParams):
        self.p = p
        dt = p.dt
        self.A = np.array([[1.0, 0.0, dt, 0.0],
                           [0.0, 1.0, 0.0, dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.B = np.array([[0.5 * dt ** 2, 0.0],
                           [0.0, 0.5 * dt ** 2],
                           [dt, 0.0],
                           [0.0, dt]])
        # 控制周期上的等价离散化（MPC 用）
        dc = p.dt_ctrl
        self.Ac = np.array([[1.0, 0.0, dc, 0.0],
                            [0.0, 1.0, 0.0, dc],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
        self.Bc = np.array([[0.5 * dc ** 2, 0.0],
                            [0.0, 0.5 * dc ** 2],
                            [dc, 0.0],
                            [0.0, dc]])

    # ------------------------------------------------------------------ #
    @staticmethod
    def brake_distance(v: float, a_brake: float) -> float:
        """制动距离 d = v^2 / (2 a_brake)。"""
        return float(v * v / (2.0 * max(a_brake, 1e-9)))

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """U: [Np,2] -> 状态序列 [Np+1,4]。"""
        Np = U.shape[0]
        X = np.zeros((Np + 1, 4))
        X[0] = x0
        for k in range(Np):
            X[k + 1] = self.Ac @ X[k] + self.Bc @ U[k]
        return X

    # ------------------------------------------------------------------ #
    @staticmethod
    def l1_disk_rows(dim: int = 2) -> np.ndarray:
        """L1 单位球的四个顶点方向：用于把 ||u||<=a 近似成 4 条线性约束。

        |a_x| + |a_y| <= a_max  ⇒  ||a||_2 <= ||a||_1 <= a_max（保守，内接多边形）。
        """
        return np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])


def l1_polynomial_violation(a: np.ndarray, a_max: float) -> float:
    """返回 ||a||_1 / a_max（>1 表示违反；由于内接多边形，1 一定满足 ||a||<=a_max）。"""
    return float(np.abs(a).sum() / max(a_max, 1e-9))
