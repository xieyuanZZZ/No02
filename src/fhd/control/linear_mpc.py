"""线性 MPC（OSQP）：跟踪参考位置/速度，满足速度与加速度约束。

决策变量 U = [u_0 .. u_{Np-1}] ∈ R^{2Np}。

预测（控制周期离散）：X_{k+1} = Ac X_k + Bc u_k，X_0 为当前状态。
因 X_k 对 U 线性，整问题可写成标准 QP：

    min  0.5 U^T H U + f^T U
    s.t. l ≤ A_c U ≤ u

约束（**保守内接多边形**近似二维范数球）：
  |u_x| + |u_y| ≤ a_max        （4 条线性约束/步）
  |v_x| + |v_y| ≤ v_max        （4 条线性约束/步）
因为 ||x||_2 ≤ ||x||_1，所以满足 L1 约束必然满足欧氏范数约束。
这是**保守**的：真实可行域比欧氏球小，但绝不会越界。

本控制器**不做**多机器人避碰，也不修改其他机器人的时序——
安全收益必须归因于 First-Hit 调度，而不是 MPC 本身。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import osqp
    import scipy.sparse as spa
    _HAS_OSQP = True
except Exception:  # pragma: no cover
    _HAS_OSQP = False


@dataclass
class MPCParams:
    horizon: int = 18
    q_pos: float = 20.0
    q_vel: float = 2.0
    r_u: float = 0.05
    s_du: float = 0.5
    p_terminal: float = 40.0
    v_max: float = 1.5
    a_max: float = 2.0
    osqp_eps: float = 1e-4
    osqp_max_iter: int = 4000


class LinearMPC:
    def __init__(self, di, params: MPCParams):
        self.di = di
        self.p = params
        self.Np = int(params.horizon)
        self._build_prediction()
        self._last_u = np.zeros(2)
        self.solve_count = 0
        self.fail_count = 0
        self.last_solve_time = 0.0
        self._solver = None

    # ------------------------------------------------------------------ #
    def _build_prediction(self):
        """构造 X = F x0 + G U 的 F、G。

        正确关系（离散双积分器，控制周期 dt_ctrl）：
            x_0 = x_current
            x_1 = Ac x_0 + Bc u_0
            x_2 = Ac^2 x_0 + Ac Bc u_0 + Bc u_1
            ...
            x_k = Ac^k x_0 + Σ_{j=0}^{k-1} Ac^{k-1-j} Bc u_j

        **注意**：u_j 的系数是 `Ac^{k-1-j} Bc`，不是 `Ac^{k-j} Bc`，
        且 j 只能取到 k-1（x_k 不依赖 u_k）。旧实现两者都写错了。
        """
        Np, Ac, Bc = self.Np, self.di.Ac, self.di.Bc
        F = np.zeros((4 * (Np + 1), 4))
        G = np.zeros((4 * (Np + 1), 2 * Np))
        # 预先算好 Ac^p
        powers = [np.eye(4)]
        for _ in range(Np):
            powers.append(Ac @ powers[-1])
        for k in range(Np + 1):
            F[4 * k:4 * k + 4, :] = powers[k]
            for j in range(k):
                G[4 * k:4 * k + 4, 2 * j:2 * j + 2] = powers[k - 1 - j] @ Bc
        self.F, self.G = F, G

    # ------------------------------------------------------------------ #
    def _weights(self):
        Np = self.Np
        P = self.p
        qd = np.tile([P.q_pos, P.q_pos, P.q_vel, P.q_vel], Np + 1).astype(np.float64)
        qd[-4:] = [P.p_terminal, P.p_terminal, P.p_terminal * 0.5, P.p_terminal * 0.5]
        Q = np.diag(qd)
        R = np.kron(np.eye(Np), np.diag([P.r_u, P.r_u]))
        D = np.zeros((2 * (Np - 1), 2 * Np))
        for k in range(Np - 1):
            D[2 * k:2 * k + 2, 2 * k:2 * k + 2] = -np.eye(2)
            D[2 * k:2 * k + 2, 2 * k + 2:2 * k + 4] = np.eye(2)
        S = P.s_du * np.eye(2 * (Np - 1))
        return Q, R, S, D

    # ------------------------------------------------------------------ #
    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> tuple[np.ndarray, dict]:
        """x0: [4]; x_ref: [Np+1,4]。返回 (u0 或全序列, info)。"""
        import time as _t
        t0 = _t.perf_counter()
        Np = self.Np
        Q, R, S, D = self._weights()
        Gm = self.G
        bias = self.F @ x0
        # J = (bias + G U - Xref)' Q (...) + U' R U + (D U - du_prev)' S (...)
        H = 2.0 * (Gm.T @ Q @ Gm + R + D.T @ S @ D)
        f = 2.0 * (Gm.T @ Q @ (bias - x_ref.reshape(-1)))
        # Δu 平滑项中的上一拍 u
        du_prev = np.tile(self._last_u, Np - 1)
        f = f - 2.0 * (D.T @ S @ du_prev)
        H = 0.5 * (H + H.T) + 1e-8 * np.eye(2 * Np)

        # 约束（保守内接 L1 多边形）
        L1 = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        Bv = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])   # 取 (vx,vy)
        rows, lo, hi = [], [], []
        for k in range(Np):                       # 控制约束 |ax|+|ay| <= a_max
            blk = np.zeros((4, 2 * Np))
            blk[:, 2 * k:2 * k + 2] = L1
            rows.append(blk)
            lo.extend([-np.inf] * 4)
            hi.extend([self.p.a_max] * 4)
        for k in range(Np + 1):                   # 速度约束 |vx|+|vy| <= v_max
            blk = (L1 @ Bv) @ Gm[4 * k:4 * k + 4, :]
            rows.append(blk)
            off = (L1 @ Bv) @ bias[4 * k:4 * k + 4]
            lo.extend([-np.inf] * 4 - off)
            hi.extend([self.p.v_max] * 4 - off)
        A_c = np.vstack(rows)
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)

        info = {"ok": False, "ub": None}
        try:
            if not _HAS_OSQP:
                raise RuntimeError("osqp 未安装")
            if self._solver is None:
                self._solver = osqp.OSQP()
                self._solver.setup(P=spa.csc_matrix(H), q=f, A=spa.csc_matrix(A_c),
                                   l=np.asarray(lo, dtype=np.float64),
                                   u=np.asarray(hi, dtype=np.float64),
                                   verbose=False, eps_abs=self.p.osqp_eps,
                                   eps_rel=self.p.osqp_eps, max_iter=self.p.osqp_max_iter,
                                   polish=False)
            else:
                self._solver.update(q=f, l=lo, u=hi)
            res = self._solver.solve()
            U = np.asarray(res.x).reshape(Np, 2)
            ok = res.info.status in ("solved", "solved inaccurate")
            info = {"ok": bool(ok), "status": res.info.status,
                    "iter": int(res.info.iter), "obj": float(res.info.obj_val)}
        except Exception as e:  # pragma: no cover
            U = np.zeros((Np, 2))
            info = {"ok": False, "status": f"error:{type(e).__name__}:{e}"}
        if not info["ok"]:
            self.fail_count += 1
            # 回退：零控制（保持当前速度，最保守的安全选择是跟随上一拍）
            U = np.tile(self._last_u, (Np, 1))
        u0 = U[0]
        u0 = self._project(u0)
        self._last_u = u0
        self.solve_count += 1
        self.last_solve_time = _t.perf_counter() - t0
        info["u"] = u0
        return U, info

    # ------------------------------------------------------------------ #
    def _project(self, u: np.ndarray) -> np.ndarray:
        """把控制投影到 L1 球（保守内接）内，数值兜底。"""
        s = np.abs(u).sum()
        if s > self.p.a_max:
            u = u * (self.p.a_max / s)
        return u

    def reset(self):
        self._last_u = np.zeros(2)
        self._solver = None
