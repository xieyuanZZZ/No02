"""First-Hit Risk 的**几何引导**（真正作用在 DDIM 去噪过程中的控制点上）。

与早期 `first_hit_guide.py` 的区别（都是必须修的）：
  1. **单位**：官方 B-spline 输出是 [-1,1] 归一化坐标，必须先乘 5 换成米制，
     再送进按米制计算的 First-Hit DP（`d_safe=0.35 m`）；
  2. **重采样**：128 个稠密采样点 != 96 执行时隙。这里用**可微等弧长重采样**
     到 H=32 个执行航点，与执行模型/等待优化使用同一套航点；
  3. **多机器人批**：`[B*N, K, 2]`，按 `b*N+i` 排布，reshape 成 `[B,N,T,2]` 后
     联合计算风险（N 台一起，不是各自独立）。

官方 Guide 契约（已在 `docs/MPD_复现与引导融合报告.md` §3 核对）：
`guide(x)` 必须返回 **-∂L/∂x**，DDIM 内部按 `x ← x + lr·guide` 上升。
"""

from __future__ import annotations

import torch

from multi_robot_mpd_sampler import official_to_meters, resample_arc_length_diff


class FirstHitGeometryGuide:
    """把首碰风险梯度接进官方 DDIM 的 guide 接口。"""

    def __init__(self, risk_model, sampler, cfg, n_agents: int,
                 lam_R: float = 1.0, lam_T: float = 0.05,
                 pause_p: float = 0.1, time_steps: int = 96,
                 lam_anchor: float = 0.0, lam_length: float = 0.0,
                 clip_value: float | None = None, h_waypoints: int = 32,
                 start_guide: bool = True, stagger_slots: int = 3,
                 lam_D: float = 1.0, d_safe_margin: float = 0.05,
                 hinge_temperature: float = 0.10):
        self.risk = risk_model
        self.sampler = sampler
        self.cfg = cfg
        self.n = int(n_agents)
        self.lam_R = float(lam_R)
        self.lam_D = float(lam_D)
        self.d_safe_margin = float(d_safe_margin)
        self.hinge_temperature = float(hinge_temperature)
        self.lam_T = float(lam_T)
        self.pause_p = float(pause_p)
        self.T = int(time_steps)
        self.lam_anchor = float(lam_anchor)
        self.lam_length = float(lam_length)
        self.clip_value = clip_value
        self.h = int(h_waypoints)
        self.enabled = start_guide
        # 基准发车间隔：与冻结轨迹库一致（第 i 台在起点等 stagger*i 个时隙）。
        # 若不设间隔，交叉轨迹在 p=0 就必然重叠，软风险饱和在 1、梯度为 0。
        self.stagger = int(stagger_slots)
        # 诊断
        self.n_calls = 0
        self.last = {}
        self.trace: list[dict] = []

    # ------------------------------------------------------------------ #
    def __call__(self, cp_norm: torch.Tensor, context_d=None, **kwargs):
        # 与官方 cost_guides 的做法一致：Guide 内部自己打开梯度
        # （ddim_sample_loop 传进来的 x 不带 requires_grad）
        with torch.enable_grad():
            return self._compute(cp_norm, context_d)

    def _compute(self, cp_norm: torch.Tensor, context_d=None, **kwargs):
        n = self.n
        bn = cp_norm.shape[0]
        b = bn // n
        starts = context_d["qs"][:, :2] if context_d is not None else None
        goals = context_d["qs"][:, 2:] if context_d is not None else None
        if starts is None:
            raise ValueError("FirstHitGeometryGuide 需要 context_d['qs'] 提供起终点")

        cp = cp_norm.detach().clone().requires_grad_(True)
        cp_m = official_to_meters(self.sampler.dataset.unnormalize_control_points(cp))
        # context_d["qs"] 已是批次 B*N 且按 b*N+i 排布，无需再 repeat
        s_b = starts                                                  # [B*N,2]
        g_b = goals
        q = self.sampler.bspline.get_q_trajectory(
            self.sampler.dataset.unnormalize_control_points(cp), s_b, g_b,
            get_type=("pos",), get_time_representation=False)["pos"]
        dense_m = official_to_meters(q)
        paths = resample_arc_length_diff(dense_m, self.h)          # [B*N,H,2]
        paths_g = paths.reshape(b, n, self.h, 2)

        wait = torch.zeros(b, n, self.h - 1, dtype=paths.dtype, device=paths.device)
        if self.stagger > 0:
            ar = torch.arange(n, dtype=paths.dtype, device=paths.device) * float(self.stagger)
            wait[:, :, 0] = ar.unsqueeze(0)
        pause = torch.full((b, n), self.pause_p, dtype=paths.dtype, device=paths.device)

        risk_sum = None
        finish_sum = None
        hard_sum = 0.0
        for bi in range(b):
            r = self.risk(paths_g[bi], wait[bi], pause[bi], self.T,
                          collision_mode="soft", gate_mode="soft",
                          return_diagnostics=False)
            risk_sum = r["pair_sum_risk"] if risk_sum is None else risk_sum + r["pair_sum_risk"]
            finish_sum = r["finish_cost"] if finish_sum is None else finish_sum + r["finish_cost"]
        risk_sum = risk_sum / b
        finish_sum = finish_sum / b

        # ---- 同刻最小间距铰链（非饱和几何代理）----
        # 首次碰撞概率在"路径重叠"时恒等于 1，梯度为 0，无法驱动几何。
        # 这里用与 First-Hit 同一时间映射（含基准发车间隔）下的同刻间距铰链，
        # 它对深度重叠仍有非零梯度；First-Hit DP 的定义不做任何改动。
        idx = torch.arange(self.h, device=paths.device).view(1, 1, self.h).expand(b, n, self.h)
        shift = wait[:, :, 0].round().long().clamp(min=0).unsqueeze(-1)   # [B,n,1]
        ti = (idx - shift).clamp(0, self.h - 1)                           # [B,n,H]
        aligned = torch.gather(paths_g, 2, ti.unsqueeze(-1).expand(b, n, self.h, 2))

        margin = getattr(self, "d_safe_margin", 0.05)
        thr = float(self.risk.d_safe) + margin
        tau_d = getattr(self, "hinge_temperature", 0.10)
        d_ij = torch.linalg.vector_norm(
            aligned[:, :, None, :, :] - aligned[:, None, :, :, :], dim=-1)   # [B,N,N,H]
        gap = torch.clamp(thr - d_ij, min=0.0)
        iu = torch.triu_indices(n, n, offset=1)
        gap_pairs = gap[:, iu[0], iu[1]]                                    # [B,P,H]
        # 对时间取均值：深度重叠时每个时刻都有正梯度，不会像 softmin/概率那样饱和
        hinge = gap_pairs.mean()

        loss = self.lam_R * risk_sum + self.lam_T * finish_sum
        if self.lam_D > 0:
            loss = loss + self.lam_D * hinge
        if self.lam_length > 0:
            seg = torch.linalg.vector_norm(paths[:, 1:] - paths[:, :-1], dim=-1)
            loss = loss + self.lam_length * seg.sum(dim=-1).mean()
        if self.lam_anchor > 0 and getattr(self, "cp_ref", None) is not None:
            loss = loss + self.lam_anchor * (cp - self.cp_ref).pow(2).sum(dim=(-1, -2)).mean()

        (g,) = torch.autograd.grad(loss, cp, retain_graph=False, create_graph=False)
        g = -g
        if self.clip_value is not None:
            g = torch.clamp(g, -self.clip_value, self.clip_value)

        # 记录（不参与求导）
        with torch.no_grad():
            for bi in range(b):
                hr = self.risk(paths_g[bi], wait[bi], pause[bi], self.T,
                               collision_mode="hard", gate_mode="hard",
                               return_diagnostics=False)
                hard_sum = hard_sum + float(hr["pair_sum_risk"])
            self.n_calls += 1
            rec = {
                "call": self.n_calls,
                "soft_risk": float(risk_sum),
                "hard_risk": hard_sum / b,
                "hinge_gap": float(hinge),
                "min_pair_dist": float(d_ij[:, iu[0], iu[1]].min()),
                "finish_cost": float(finish_sum),
                "loss": float(loss),
                "grad_norm": float(g.norm()),
                "grad_absmax": float(g.abs().max()),
                "cp_disp_absmax": float((cp - (self.cp_ref if getattr(self, "cp_ref", None)
                                               is not None else cp)).abs().max()),
                "path_length": float(torch.linalg.vector_norm(
                    paths[:, 1:] - paths[:, :-1], dim=-1).sum(dim=-1).mean()),
            }
            self.trace.append(rec)
            self.last = rec
        return g


class SumGuide:
    """把多个 guide 相加（官方 SDF + First-Hit），按权重。"""

    def __init__(self, guides, weights=None, names=None):
        self.guides = list(guides)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.guides)
        self.names = names or [f"g{i}" for i in range(len(self.guides))]

    def __call__(self, cp_norm, context_d=None, **kwargs):
        total = None
        for g, w, name in zip(self.guides, self.weights, self.names):
            if g is None or w == 0.0:
                continue
            try:
                val = g(cp_norm, context_d=context_d, **kwargs)
            except TypeError:
                val = g(cp_norm)
            total = w * val if total is None else total + w * val
        if total is None:
            return torch.zeros_like(cp_norm)
        return total

    # 供官方代码调用
    def warmup(self, *a, **k):
        for g in self.guides:
            if g is not None and hasattr(g, "warmup"):
                g.warmup(*a, **k)

    def __getattr__(self, item):
        for g in self.guides:
            if g is not None and hasattr(g, item):
                return getattr(g, item)
        raise AttributeError(item)
