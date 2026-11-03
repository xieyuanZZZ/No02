"""把 No02 的**首次碰撞进度风险**接进官方 MPD 的 Guide 接口。

官方契约（已从源码核对，见 docs/MPD_复现与引导融合报告.md §3）：

  guide_gradient_steps():
      grad_guide = guide(x_opt, context_d=context_d)   # x_opt 是**归一化**控制点
      x_opt.grad = -grad_guide; opt.step()             # x ← x + lr * grad_guide
  => guide 必须返回 **-∂L/∂cp_norm**（"要最大化的方向"）

本模块只实现"代价 + 梯度"，不改官方网络、不改官方 Guide 类；
使用时把它和官方 SDF guide 相加即可（方案 A）。

坐标与单位：
  * 入参/出参都是**归一化**控制点 [B*N, K, D]；
  * 反归一化用官方 dataset 的 `unnormalize_control_points`（保持计算图）；
  * 官方 2D 世界是米制、约 [-1.25, 1.25]^2（见 EnvSimple2D.limits），
    风险模型直接在米制上算，不需要再缩放。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FirstHitGuideConfig:
    lam_R: float = 1.0          # 软首次碰撞风险权重
    lam_T: float = 0.05         # 完成时间代价权重
    n_agents: int = 1           # N：一组里的机器人数
    batch_candidates: int = 1   # B：候选数
    clip_value: float | None = None   # 对最终梯度做逐元素裁剪（None 表示不裁）
    detach_risk_params: bool = True   # 风险模型的内部参数不参与求导（默认已冻结）
    # 端点在 B-spline 里是硬约束，必须与扩散条件用同一组值
    start: torch.Tensor | None = None   # [B*N, D]
    goal: torch.Tensor | None = None    # [B*N, D]


class FirstHitRiskGuide:
    """官方 Guide 契约的实现：返回 -∂(λ_R·R + λ_T·J_finish)/∂cp_norm。"""

    def __init__(self, risk_model, bspline, dataset, cfg: FirstHitGuideConfig,
                 wait: torch.Tensor | None = None, pause_p: torch.Tensor | None = None,
                 time_steps: int = 96, d_safe: float | None = None):
        self.risk_model = risk_model
        self.bspline = bspline
        self.dataset = dataset
        self.cfg = cfg
        self.wait = wait
        self.pause_p = pause_p
        self.time_steps = int(time_steps)
        self.d_safe = d_safe
        self.last_cost: float | None = None
        self.n_calls = 0

    # ------------------------------------------------------------------ #
    def __call__(self, cp_norm: torch.Tensor, context_d=None, **kwargs) -> torch.Tensor:
        """cp_norm: [B*N, K, D]（归一化）。返回同形状的 -∂L/∂cp_norm。"""
        cfg = self.cfg
        # 关键：不要 detach，保持 cp_norm -> cp -> q -> risk 的完整计算图
        cp = self.dataset.unnormalize_control_points(cp_norm)
        if not cp.requires_grad:
            cp.requires_grad_(True)

        q_d = self.bspline.get_q_trajectory(
            cp, cfg.start, cfg.goal, get_type=("pos",),
            get_time_representation=False)
        q = q_d["pos"]                      # [B*N, n_T_pts, D]
        b_n, t, d = q.shape
        n = cfg.n_agents
        if b_n % n != 0:
            raise ValueError(f"batch {b_n} 不是机器人数 {n} 的整数倍；请用 B*N 展平")
        cap = int(self.time_steps)
        if t > cap:
            # 官方 n_T_pts(128) 可能大于执行模型的时隙数(96)：按进度重采样到 cap
            idx = torch.linspace(0, t - 1, cap).round().long().to(q.device)
            q = q.index_select(1, idx)
            t = cap
        q_group = q.reshape(b_n // n, n, t, d)   # [B, N, T, D]

        # No02 的风险模型按「一组 N 个机器人」计算，B 个候选逐个累加后取平均
        n_b = q_group.shape[0]
        risk_sum, finish_sum = None, None
        for b in range(n_b):
            risk_b = self.risk_model(
                q_group[b], self._wait_like(q_group, b), self._pause_like(q_group, b),
                self.time_steps, collision_mode="soft", gate_mode="soft",
                return_diagnostics=False,
            )
            risk_sum = risk_b["pair_sum_risk"] if risk_sum is None \
                else risk_sum + risk_b["pair_sum_risk"]
            finish_sum = risk_b["finish_cost"] if finish_sum is None \
                else finish_sum + risk_b["finish_cost"]
        loss = cfg.lam_R * (risk_sum / n_b) + cfg.lam_T * (finish_sum / n_b)
        self.last_cost = float(loss.detach())

        (g,) = torch.autograd.grad(loss, cp_norm, retain_graph=False, create_graph=False)
        g = -g                                    # ← 官方约定的负号
        if cfg.clip_value is not None:
            g = torch.clamp(g, -cfg.clip_value, cfg.clip_value)
        self.n_calls += 1
        return g

    # ------------------------------------------------------------------ #
    def _wait_like(self, q_group: torch.Tensor, b: int = 0) -> torch.Tensor:
        """单个候选（第 b 组）的等待张量 [N, T-1]。"""
        if self.wait is not None:
            return self.wait
        _, n, t, _ = q_group.shape
        return torch.zeros(n, t - 1, dtype=q_group.dtype, device=q_group.device)

    def _pause_like(self, q_group: torch.Tensor, b: int = 0) -> torch.Tensor:
        """单个候选（第 b 组）的暂停概率 [N]。"""
        if self.pause_p is not None:
            return self.pause_p
        _, n, _, _ = q_group.shape
        return torch.full((n,), 0.1, dtype=q_group.dtype, device=q_group.device)


class SumGuide:
    """把多个 guide 相加（例如 官方 SDF guide + 我们的进度风险 guide）。"""

    def __init__(self, guides, weights=None):
        self.guides = list(guides)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.guides)

    def __call__(self, cp_norm, context_d=None, **kwargs):
        total = None
        for g, w in zip(self.guides, self.weights):
            val = g(cp_norm, context_d=context_d, **kwargs)
            total = w * val if total is None else total + w * val
        return total
