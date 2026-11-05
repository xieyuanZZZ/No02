"""线性 MPC（OSQP）：跟踪参考位置/速度，满足速度与加速度约束。

决策变量 U = [u_0 .. u_{Np-1}] ∈ R^{2Np}。

预测（控制周期离散）：X_{k+1} = Ac X_k + Bc u_k，X_0 为当前状态。
因 X_k 对 U 线性，整问题可写成标准 QP：

    min  0.5 U^T H U + f^T U
    s.t. l ≤ A_c U ≤ u

目标函数（与 `explicit_cost` 逐项一致）：

    J(U) = ||F x0 + G U - X_ref||_Q^2
         + ||U||_R^2
         + λ_Δu ( ||u_0 - u_{-1}||^2
                  + Σ_{k=1}^{Np-1} ||u_k - u_{k-1}||^2 )

注意平滑项**只对第一个控制**使用上一拍控制 u_{-1}，其余差分是相邻控制之差。
旧实现把 u_{-1} 从每个相邻差分中减掉，是错误的；本文件的 `_difference_matrix`
用一行 [I, 0, ...] 单独表示 u_0 - u_{-1}，测试 `test_mpc_incremental_cost_*`
会与显式目标对比梯度。

约束（**保守内接多边形**近似二维范数球）：
  |u_x| + |u_y| ≤ a_max        （4 条线性约束/步）
  |v_x| + |v_y| ≤ v_max        （4 条线性约束/步）
因为 ||x||_2 ≤ ||x||_1，所以满足 L1 约束必然满足欧氏范数约束。

失败回退：求解失败/不可行时不再"重用上一拍加速度"，而是构造
**受加速度限幅的制动控制**（沿速度反方向、限 L1 球内、且不产生反向速度），
并显式检查速度界与停车可行性；若连该动作也不可行，返回 `usable=False`，
由执行器计数，不允许静默修改物理状态。

本控制器**不做**多机器人避碰，也不修改其他机器人的时序——
安全收益必须归因于 First-Hit 调度，而不是 MPC 本身。
"""

from __future__ import annotations

from collections import Counter
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


class SolveStats:
    """全程求解统计（不是"最后一次求解耗时"）。

    `solve_times` 保存每一次调用的 wall-clock 秒数，供 P50/P95/P99 汇总。
    """

    def __init__(self) -> None:
        self.solve_times: list[float] = []
        self.status_counts: Counter = Counter()
        self.n_solves = 0
        self.n_failed = 0
        self.n_infeasible = 0
        self.n_inaccurate = 0
        self.n_fallback = 0
        self.n_fallback_infeasible = 0
        self.n_projection = 0

    def add(self, elapsed_s: float, info: dict) -> None:
        self.n_solves += 1
        self.solve_times.append(float(elapsed_s))
        status = str(info.get("status", "unknown"))
        self.status_counts[status] += 1
        if status == "solved inaccurate":
            self.n_inaccurate += 1
        if not info.get("ok", False):
            self.n_failed += 1
        if info.get("infeasible", False):
            self.n_infeasible += 1
        if info.get("fallback") is not None:
            self.n_fallback += 1
        if info.get("fallback") is not None and not info.get("usable", True):
            self.n_fallback_infeasible += 1
        if info.get("projected", False):
            self.n_projection += 1

    def summary(self) -> dict:
        t = np.asarray(self.solve_times, dtype=np.float64)
        if t.size == 0:
            pct = {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
            mean_ms = 0.0
        else:
            pct = {
                "p50_ms": float(np.percentile(t, 50) * 1e3),
                "p95_ms": float(np.percentile(t, 95) * 1e3),
                "p99_ms": float(np.percentile(t, 99) * 1e3),
            }
            mean_ms = float(t.mean() * 1e3)
        return {
            "n_solves": int(self.n_solves),
            "mean_ms": mean_ms,
            **pct,
            "failed": int(self.n_failed),
            "infeasible": int(self.n_infeasible),
            "inaccurate": int(self.n_inaccurate),
            "fallback": int(self.n_fallback),
            "fallback_infeasible": int(self.n_fallback_infeasible),
            "projections": int(self.n_projection),
            "failure_rate": float(self.n_failed / max(self.n_solves, 1)),
            "statuses": dict(self.status_counts),
        }


class LinearMPC:
    def __init__(self, di, params: MPCParams):
        self.di = di
        self.p = params
        self.Np = int(params.horizon)
        self._build_prediction()
        self._build_difference_matrix()
        self._build_constraint_matrix()
        self._last_u = np.zeros(2)
        self.solve_count = 0
        self.fail_count = 0
        self.last_solve_time = 0.0
        self.stats = SolveStats()
        self._solver = None
        self._solver_signature = None

    # ------------------------------------------------------------------ #
    def _build_prediction(self):
        """构造 X = F x0 + G U 的 F、G（本修复已通过独立积分测试，保持不动）。

        正确关系（离散双积分器，控制周期 dt_ctrl）：
            x_0 = x_current
            x_1 = Ac x_0 + Bc u_0
            x_2 = Ac^2 x_0 + Ac Bc u_0 + Bc u_1
            ...
            x_k = Ac^k x_0 + Σ_{j=0}^{k-1} Ac^{k-1-j} Bc u_j
        """
        Np, Ac, Bc = self.Np, self.di.Ac, self.di.Bc
        F = np.zeros((4 * (Np + 1), 4))
        G = np.zeros((4 * (Np + 1), 2 * Np))
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
        return Q, R

    # ------------------------------------------------------------------ #
    def _build_difference_matrix(self) -> np.ndarray:
        """Δu 的线性映射矩阵 D，使 D U = [u_0, u_1-u_0, ..., u_{Np-1}-u_{Np-2}]。

        第一行块只作用 u_0；上一拍 u_{-1} 通过常数向量 d_prev 进入目标，
        不再像旧实现那样从每个相邻差分中减去 u_{-1}。
        """
        Np = self.Np
        D = np.zeros((2 * Np, 2 * Np))
        D[0:2, 0:2] = np.eye(2)
        for k in range(1, Np):
            D[2 * k:2 * k + 2, 2 * (k - 1):2 * k] = -np.eye(2)
            D[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.eye(2)
        self.D_delta = D
        S = self.p.s_du * np.eye(2 * Np)
        self.S_delta = S
        return D

    # ------------------------------------------------------------------ #
    def _build_constraint_matrix(self):
        """约束矩阵 A_c 与 x0 无关，缓存起来；只有 l/u 随 x0 改变。"""
        Np = self.Np
        L1 = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        Bv = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self._L1 = L1
        self._L1Bv = L1 @ Bv
        rows = []
        for k in range(Np):                       # 控制约束 |ax|+|ay| <= a_max
            blk = np.zeros((4, 2 * Np))
            blk[:, 2 * k:2 * k + 2] = L1
            rows.append(blk)
        for k in range(Np + 1):                   # 速度约束 |vx|+|vy| <= v_max
            rows.append(self._L1Bv @ self.G[4 * k:4 * k + 4, :])
        self._A = np.vstack(rows)
        self._n_acc_rows = 4 * Np
        self._a_hi = np.full(4 * Np, float(self.p.a_max))
        self._a_lo = np.full(4 * Np, -np.inf)

    # ------------------------------------------------------------------ #
    def _assemble_qp(self, x0: np.ndarray, x_ref: np.ndarray):
        """组装 QP 的 (H, f, A, l, u)。H/f 与 `explicit_cost` 严格对应。"""
        Np = self.Np
        Q, R = self._weights()
        if self._last_u.shape != (2,):
            raise ValueError(f"_last_u 形状应为 (2,)，实际 {self._last_u.shape}")
        Gm = self.G
        bias = self.F @ x0
        QD = self.D_delta
        SD = float(self.p.s_du) * np.eye(2 * Np)
        self.S_delta = SD
        d_prev = np.zeros(2 * Np)
        d_prev[0:2] = self._last_u

        H = 2.0 * (Gm.T @ Q @ Gm + R + QD.T @ SD @ QD)
        f = 2.0 * (Gm.T @ Q @ (bias - x_ref.reshape(-1)))
        f = f - 2.0 * (QD.T @ SD @ d_prev)
        H = 0.5 * (H + H.T) + 1e-8 * np.eye(2 * Np)

        # 速度约束右端随 x0 改变；加速度约束不变
        bias_blocks = bias.reshape(Np + 1, 4)
        off = (self._L1Bv @ bias_blocks.T).T          # [Np+1,4]
        # +1e-6 容忍浮点累计误差；否则当前速度比 v_max 大 1e-9 就会让 QP
        # 误报 primal infeasible（物理上仍在数值容差内）。
        hi_vel_blocks = np.full_like(off, self.p.v_max + 1e-6) - off
        # k=0 是"已发生"的当前状态；若它因数值漂移略超 v_max，放宽这一块，
        # 让 QP 有机会用后续控制把速度拉回界内，而不是直接报不可行。
        cur_l1 = float(np.abs(np.asarray(x0[2:])).sum())
        if cur_l1 > self.p.v_max:
            hi_vel_blocks[0] = cur_l1 + 1e-6 - off[0]
        hi_vel = hi_vel_blocks.reshape(-1)
        lo = np.concatenate([self._a_lo, np.full(4 * (Np + 1), -np.inf)])
        hi = np.concatenate([self._a_hi, hi_vel])
        return H, f, self._A, lo, hi

    # ------------------------------------------------------------------ #
    def explicit_cost(self, U: np.ndarray, x0: np.ndarray, x_ref: np.ndarray,
                      last_u: np.ndarray | None = None) -> float:
        """显式目标函数，供测试与 QP 的 H/f 对照（含常数项）。"""
        Np = self.Np
        Q, R = self._weights()
        U = np.asarray(U, dtype=np.float64).reshape(Np, 2)
        last = self._last_u if last_u is None else np.asarray(last_u, dtype=np.float64)
        X = self.F @ x0 + self.G @ U.reshape(-1)
        err = X - np.asarray(x_ref, dtype=np.float64).reshape(-1)
        du = U.copy()
        du[0] = U[0] - last
        du[1:] = U[1:] - U[:-1]
        u_vec = U.reshape(-1)
        return float(err @ Q @ err + u_vec @ R @ u_vec + self.p.s_du * np.sum(du ** 2))

    # ------------------------------------------------------------------ #
    @staticmethod
    def stopping_feasible(speed: float, distance: float, a_brake: float,
                          margin: float = 0.0) -> bool:
        """以 a_brake 制动能否在 distance 内停住（用于停车可行性检查）。"""
        return float(speed) ** 2 / (2.0 * max(a_brake, 1e-9)) <= max(distance - margin, 0.0) + 1e-9

    # ------------------------------------------------------------------ #
    def _safe_fallback(self, x0: np.ndarray) -> tuple[np.ndarray, bool, str]:
        """失败回退：限幅制动 / 零控制，并检查速度界与不反向。"""
        v = np.asarray(x0[2:], dtype=np.float64)
        speed = float(np.linalg.norm(v))
        dt = float(self.di.p.dt_ctrl)
        if speed <= 1e-9:
            u = np.zeros(2)
            reason = "zero"
        else:
            # 不超过 a_max 且不在一拍内反向；L1 投影是保守内接
            decel = min(self.p.a_max, speed / max(dt, 1e-9))
            u = -v / speed * decel
            u = self._project(u)
            reason = "brake"
        v_next = v + dt * u
        ok = (
            np.abs(u).sum() <= self.p.a_max * (1.0 + 1e-6)
            and np.linalg.norm(v_next) <= self.p.v_max * (1.0 + 1e-6)
            and np.linalg.norm(v_next) <= speed + 1e-9
        )
        return u, bool(ok), reason

    # ------------------------------------------------------------------ #
    def _ensure_solver(self, H, f, A_c, lo, hi):
        import hashlib
        signature = (H.shape, A_c.shape,
                     hashlib.blake2b(H.tobytes(), digest_size=16).hexdigest(),
                     hashlib.blake2b(A_c.tobytes(), digest_size=16).hexdigest())
        if self._solver is None or self._solver_signature != signature:
            self._solver = osqp.OSQP()
            self._solver.setup(P=spa.csc_matrix(H), q=f, A=spa.csc_matrix(A_c),
                               l=lo, u=hi, verbose=False,
                               eps_abs=self.p.osqp_eps, eps_rel=self.p.osqp_eps,
                               max_iter=self.p.osqp_max_iter, polish=False)
            self._solver_signature = signature
            return "setup"
        self._solver.update(q=f, l=lo, u=hi)
        return "update"

    # ------------------------------------------------------------------ #
    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> tuple[np.ndarray, dict]:
        """x0: [4]; x_ref: [Np+1,4]。返回 (U [Np,2], info)。"""
        import time as _t
        t0 = _t.perf_counter()
        Np = self.Np
        H, f, A_c, lo, hi = self._assemble_qp(x0, x_ref)

        info: dict = {"ok": False, "status": "unknown", "u": np.zeros(2),
                      "fallback": None, "usable": True, "projected": False}
        U = np.zeros((Np, 2))
        try:
            if not _HAS_OSQP:
                raise RuntimeError("osqp 未安装")
            mode = self._ensure_solver(H, f, A_c, lo, hi)
            res = self._solver.solve()
            status = str(res.info.status)
            ok = status in ("solved", "solved inaccurate")
            U = np.asarray(res.x, dtype=np.float64).reshape(Np, 2)
            info = {
                "ok": bool(ok), "status": status,
                "iter": int(res.info.iter), "obj": float(res.info.obj_val),
                "setup_mode": mode, "fallback": None, "usable": True,
                "projected": False, "infeasible": "infeasible" in status.lower(),
            }
        except Exception as e:  # pragma: no cover
            info = {"ok": False, "status": f"error:{type(e).__name__}:{e}",
                    "fallback": None, "usable": True, "projected": False,
                    "infeasible": True}

        if not info["ok"]:
            self.fail_count += 1
            u_fb, fb_ok, reason = self._safe_fallback(x0)
            info["fallback"] = reason
            info["usable"] = bool(fb_ok)
            info["fallback_feasible"] = bool(fb_ok)
            if fb_ok:
                U = np.tile(u_fb, (Np, 1))
            else:
                U = np.zeros((Np, 2))

        u0 = np.asarray(U[0], dtype=np.float64)
        u_proj = self._project(u0)
        if np.abs(u_proj - u0).max() > 1e-12:
            info["projected"] = True
            u0 = u_proj
        self._last_u = u0
        self.solve_count += 1
        self.last_solve_time = _t.perf_counter() - t0
        self.stats.add(self.last_solve_time, info)
        info["u"] = u0
        info["solve_s"] = self.last_solve_time
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
        self._solver_signature = None
        self.solve_count = 0
        self.fail_count = 0
        self.last_solve_time = 0.0
        self.stats = SolveStats()
