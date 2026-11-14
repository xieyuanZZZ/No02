"""阶段三：等待策略（W0–W3）与风险梯度诊断。

约定（与 `fhd.execution` 完全一致）：
  * `w[i,k]` = 机器人在第 k 个航点处的**额外等待时隙**，非负；
  * `tau_{i,m} = m + Σ_{h≤m} w[i,h]`（`release_times`），DP 与执行模拟器共用；
  * 完成时隙 = (H-1) + Σ_k w[i,k]；
  * 冻结任务里已含固定 3 时隙发车间隔（写在 `baseline_wait[i,0]`），
    所有策略都以 `baseline_wait` 为下界，绝不删除它。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
@dataclass
class WaitBudget:
    """等待约束（时隙单位）。"""

    max_added_per_point: float = 6.0
    max_added_total_per_robot: float = 12.0
    time_steps: int = 96
    horizon: int = 32

    def max_added_total_allowed(self) -> float:
        """由执行窗口推出的硬上限：完成时隙必须 < time_steps。"""
        return float(self.time_steps - 1 - (self.horizon - 1))


@dataclass
class WaitResult:
    wait: torch.Tensor
    added: torch.Tensor
    soft_risk: float
    hard_risk: float
    finish_s: float
    feasible: bool
    fallback: bool
    history: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    dp_evals: int = 0          # 调用（硬/软）First-Hit DP 前向的次数
    grad_steps: int = 0        # 反向传播步数（无梯度方法恒为 0）
    p0_safe: bool = True       # 最终等待计划在 p=0 下是否无碰撞


# --------------------------------------------------------------------------- #
def allowed_positions(horizon: int, stride: int = 5) -> list[int]:
    """允许增加等待的航点位置：起点 + 每 stride 个内部航点（最后一列对应终点前）。"""
    pos = {0}
    for k in range(stride, horizon - 1, stride):
        pos.add(k)
    return sorted(pos)


def make_mask(horizon: int, positions: list[int], n_agents: int) -> torch.Tensor:
    m = torch.zeros(n_agents, horizon - 1, dtype=torch.float64)
    for k in positions:
        if 0 <= k < horizon - 1:
            m[:, k] = 1.0
    return m


def project_wait(w: torch.Tensor, w0: torch.Tensor, mask: torch.Tensor,
                 budget: WaitBudget) -> torch.Tensor:
    """投影回可行域：w ≥ w0、只允许的位置、单点与总量上限、执行窗口。"""
    w = torch.maximum(w, w0)                      # 只能增加
    w = w0 + (w - w0) * mask                       # 只允许的位置
    w = w0 + (w - w0).clamp(max=budget.max_added_per_point)

    added = w - w0
    total = added.sum(dim=-1, keepdim=True)
    cap = min(budget.max_added_total_per_robot, budget.max_added_total_allowed())
    scale = torch.clamp(cap / total.clamp(min=1e-9), max=1.0)
    added = added * scale
    return w0 + added


def discretize(w: torch.Tensor, w0: torch.Tensor, mask: torch.Tensor,
               budget: WaitBudget) -> torch.Tensor:
    """连续松弛 -> 整数时隙，并做约束修复。"""
    added = (w - w0) * mask
    added = torch.round(added)                     # 执行器只接受整数时隙
    added = added.clamp(min=0.0, max=budget.max_added_per_point)

    cap = min(budget.max_added_total_per_robot, budget.max_added_total_allowed())
    total = added.sum(dim=-1, keepdim=True)
    # 超预算时按比例削减，再取整，直到满足
    for _ in range(8):
        if bool((total <= cap).all()):
            break
        scale = (cap / total.clamp(min=1e-9)).clamp(max=1.0)
        added = torch.floor(added * scale)
        total = added.sum(dim=-1, keepdim=True)
    return w0 + added


# --------------------------------------------------------------------------- #
def _eval(risk_model, paths, wait, pause_p, time_steps):
    with torch.no_grad():
        soft = risk_model(paths, wait, pause_p, time_steps, collision_mode="soft",
                          gate_mode="soft", return_diagnostics=False)
        hard = risk_model(paths, wait, pause_p, time_steps, collision_mode="hard",
                          gate_mode="hard", return_diagnostics=False)
    return (float(soft["pair_sum_risk"]), float(hard["pair_sum_risk"]),
            float(soft["finish_cost"]), float(hard["finish_cost"]))


def _eval_hard(risk_model, paths, wait, pause_p, time_steps):
    """只跑 Hard First-Hit DP（W2+ 候选评价用，省掉软核那一半开销）。"""
    with torch.no_grad():
        hard = risk_model(paths, wait, pause_p, time_steps, collision_mode="hard",
                          gate_mode="hard", return_diagnostics=False)
    return float(hard["pair_sum_risk"]), float(hard["finish_cost"])


def policy_w0(w0, mask, budget, risk_model, paths, pause_p, time_steps) -> WaitResult:
    s, h, f, _ = _eval(risk_model, paths, w0, pause_p, time_steps)
    return WaitResult(wait=w0.clone(), added=torch.zeros_like(w0), soft_risk=s,
                      hard_risk=h, finish_s=f, feasible=True, fallback=False)


def apply_start_wait(w0, delta: float) -> torch.Tensor:
    """固定发车间隔扩增：第 i 台机器人在起点额外等待 delta*i 个时隙。

    N=2 时退化为"只给第 2 台加 delta"，与阶段三的实现一致；
    N>2 时相当于把固定时间裕度按顺序摊开，是所有方法共用的同族基线。
    """
    n = w0.shape[0]
    add = torch.zeros_like(w0)
    idx = torch.arange(n, dtype=w0.dtype, device=w0.device) * float(delta)
    add[:, 0] = idx
    return w0 + add


def policy_w1(w0, mask, budget, risk_model, paths, pause_p, time_steps,
              delta: int, robot: int | None = None) -> WaitResult:
    """固定额外等待：在起点给第 i 台机器人加 delta*i 个时隙。"""
    w = apply_start_wait(w0, delta)
    w = project_wait(w, w0, mask, budget)
    w = torch.round(w)
    s, h, f, _ = _eval(risk_model, paths, w, pause_p, time_steps)
    return WaitResult(wait=w, added=w - w0, soft_risk=s, hard_risk=h, finish_s=f,
                      feasible=True, fallback=False)


def policy_w2_enum(w0, mask, budget, risk_model, paths, pause_p, time_steps,
                   lam_T: float, deltas=tuple(range(0, 9)),
                   robot: int | None = None) -> WaitResult:
    """枚举起点等待：用 DP 的硬风险 + 完成时间增量选最优。"""
    base_finish = _eval(risk_model, paths, w0, pause_p, time_steps)[2]
    zero = torch.zeros_like(pause_p)
    best = None
    for d in deltas:
        w = apply_start_wait(w0, d)
        w = torch.round(project_wait(w, w0, mask, budget))
        # 硬约束：等待不得让 p=0 变得不安全（Δ=0 一定满足，故候选集非空）
        _, h0, _, _ = _eval(risk_model, paths, w, zero, time_steps)
        if h0 > 1e-9:
            continue
        _, h, f, _ = _eval(risk_model, paths, w, pause_p, time_steps)
        j = h + lam_T * (f - base_finish)
        rec = {"delta": int(d), "hard_risk": h, "finish_s": f, "J": j}
        if best is None or j < best[0]:
            best = (j, w, rec)
    _, w, rec = best
    s, h, f, _ = _eval(risk_model, paths, w, pause_p, time_steps)
    return WaitResult(wait=w, added=w - w0, soft_risk=s, hard_risk=h, finish_s=f,
                      feasible=True, fallback=False, history=[rec])


def policy_w3_optimize(w0, mask, budget, risk_model, paths, pause_p, time_steps,
                       lam_R=1.0, lam_T=0.05, lam_W=0.01, steps=120, lr=0.15,
                       gd_check: bool = False) -> WaitResult:
    """First-Hit 风险优化等待（连续松弛 + 离散化 + 硬风险复核 + 安全回退）。"""
    t0 = time.perf_counter()
    u = torch.zeros_like(w0, requires_grad=True)     # 用 softplus 保证 w ≥ w0
    opt = torch.optim.Adam([u], lr=lr)
    base_finish = _eval(risk_model, paths, w0, pause_p, time_steps)[2]
    history: list[dict] = []
    soft0, hard0, _, _ = _eval(risk_model, paths, w0, pause_p, time_steps)

    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        w = project_wait(w0 + F.softplus(u) * mask, w0, mask, budget)
        risk = risk_model(paths, w, pause_p, time_steps, collision_mode="soft",
                          gate_mode="soft", return_diagnostics=False)
        loss = (lam_R * risk["pair_sum_risk"]
                + lam_T * (risk["finish_cost"] - base_finish)
                + lam_W * (w - w0).abs().sum())
        loss.backward()
        gnorm = float(u.grad.norm()) if u.grad is not None else 0.0
        nz = int((u.grad.abs() > 1e-8).sum()) if u.grad is not None else 0
        opt.step()
        if it % 20 == 0 or it == steps - 1:
            with torch.no_grad():
                soft_v = float(risk["pair_sum_risk"])
                added_v = float((w - w0).sum())
            history.append({"iter": it, "loss": float(loss.detach()),
                            "soft_risk": soft_v, "grad_norm": gnorm,
                            "n_nonzero_grad": nz, "added_slots": added_v})

    with torch.no_grad():
        w_cont = project_wait(w0 + F.softplus(u) * mask, w0, mask, budget)
        w_disc = discretize(w_cont, w0, mask, budget)
        s_d, h_d, f_d, _ = _eval(risk_model, paths, w_disc, pause_p, time_steps)
        completion = (budget.horizon - 1) + w_disc.sum(dim=-1)   # 每台机器人的完成时隙
        over_window = bool((completion >= budget.time_steps).any())
        if (not bool(torch.isfinite(w_disc).all())) or h_d > hard0 + 1e-9 or over_window:
            # 回退：优化结果不优于原计划（硬风险）或越过执行窗口
            return WaitResult(wait=w0.clone(), added=torch.zeros_like(w0),
                              soft_risk=soft0, hard_risk=hard0, finish_s=base_finish,
                              feasible=True, fallback=True, history=history,
                              elapsed_s=time.perf_counter() - t0,
                              dp_evals=steps + 2, grad_steps=steps, p0_safe=True)
    with torch.no_grad():
        zero = torch.zeros_like(pause_p)
        p0_risk = risk_model(paths, w_disc, zero, time_steps, collision_mode="hard",
                             gate_mode="hard", return_diagnostics=False)
        p0_safe = float(p0_risk["pair_sum_risk"]) <= 1e-9
    return WaitResult(wait=w_disc, added=w_disc - w0, soft_risk=s_d, hard_risk=h_d,
                      finish_s=f_d, feasible=True, fallback=False, history=history,
                      elapsed_s=time.perf_counter() - t0,
                      dp_evals=steps + 3, grad_steps=steps, p0_safe=bool(p0_safe))


# --------------------------------------------------------------------------- #
def policy_w2plus(w0, mask, budget, risk_model, paths, pause_p, time_steps,
                  lam_T: float = 0.05, max_evals: int = 800, restarts: int = 3,
                  allow_two_move: bool = True, seed: int = 0,
                  require_p0: bool = True, p0_weight: float = 1.0,
                  initial_stagger: bool = False) -> WaitResult:
    """W2+：与 W3 **完全相同的等待变量与约束**，但用整数坐标下降（无梯度）。

    搜索：
      * 变量 = 允许航点上的整数新增等待（与 W3 同一 mask）；
      * 每次尝试把某个变量 ±1 时隙，用 **Hard First-Hit DP** 评价；
      * 目标 J = R_hard + λ_T·(T_finish − T_finish(w0))；
      * 硬约束：w ≥ w0、单点 ≤6、每机器人总量 ≤12、完成时隙 < T、**p=0 必须无碰撞**；
      * 多起点 + 随机重启，可选一次调整两个变量的邻域。
    """
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(int(seed))
    zero = torch.zeros_like(pause_p)
    n, h1 = w0.shape
    positions = [(i, k) for i in range(n) for k in range(h1) if float(mask[i, k]) > 0]

    base = _eval(risk_model, paths, w0, pause_p, time_steps)
    base_finish = base[2]
    dp_evals = 0

    def evaluate(w):
        nonlocal dp_evals
        dp_evals += 1
        hard_p, hard_finish = _eval_hard(risk_model, paths, w, pause_p, time_steps)
        # Optional relaxation for scenes whose w0 already violates p=0 hard
        # feasibility: keep optimizing but penalize the p=0 residual instead of
        # rejecting every start (the frozen default remains require_p0=True).
        penalty = 0.0
        if not require_p0:
            h0, _ = _eval_hard(risk_model, paths, w, zero, time_steps)
            penalty = p0_weight * h0
        return (hard_p + penalty + lam_T * (hard_finish - base_finish),
                hard_p, hard_finish)

    def p0_ok(w):
        nonlocal dp_evals
        dp_evals += 1
        h0, _ = _eval_hard(risk_model, paths, w, zero, time_steps)
        return h0 <= 1e-9

    def repair(w):
        w = torch.maximum(w, w0)
        w = torch.round(w)
        return project_wait(w, w0, mask, budget).round()

    def legal(w):
        added = w - w0
        if float(added.min()) < -1e-9:
            return False
        if float(added.max()) > budget.max_added_per_point + 1e-9:
            return False
        cap = min(budget.max_added_total_per_robot, budget.max_added_total_allowed())
        if float(added.sum(dim=-1).max()) > cap + 1e-9:
            return False
        completion = (budget.horizon - 1) + w.sum(dim=-1)
        return bool((completion < budget.time_steps).all())

    # ---- 多起点 ----
    starts = [w0.clone()]
    if restarts >= 2:
        w1 = w0.clone()
        for i, k in positions:
            w1[i, k] = w1[i, k] + 1.0
        starts.append(repair(w1))
    if restarts >= 3:
        w2 = w0.clone()
        noise = torch.randint(0, 3, (n, h1), generator=gen).to(w0.dtype)
        w2 = w2 + noise * mask
        starts.append(repair(w2))
    if initial_stagger:
        ws = w0.clone()
        for i in range(n):
            if mask[i, 0] > 0:
                ws[i, 0] = max(ws[i, 0], float(i))
        starts.append(repair(ws))

    best = None
    history: list[dict] = []
    for si, start in enumerate(starts[:max(1, restarts)]):
        if not legal(start):
            continue
        cur = start.clone()
        cur_j, cur_hard, cur_fin = evaluate(cur)
        if require_p0 and not p0_ok(cur):
            continue
        evals_this = 0
        improved = True
        sweep = 0
        while improved and evals_this < max_evals:
            improved = False
            sweep += 1
            # 单变量邻域
            for (i, k) in positions:
                for delta in (+1.0, -1.0):
                    if evals_this >= max_evals:
                        break
                    cand = cur.clone()
                    cand[i, k] = cand[i, k] + delta
                    cand = repair(cand)
                    if not legal(cand):
                        continue
                    if torch.allclose(cand, cur):
                        continue
                    evals_this += 1
                    j, hp, fin = evaluate(cand)
                    if j < cur_j - 1e-9 and (not require_p0 or p0_ok(cand)):
                        cur, cur_j, cur_hard, cur_fin = cand, j, hp, fin
                        improved = True
            # 双变量邻域（一次 +1 两个位置），缓解坐标下降的局部最优
            if allow_two_move and evals_this < max_evals and not improved:
                for a in range(len(positions)):
                    if evals_this >= max_evals or improved:
                        break
                    for b in range(a + 1, len(positions)):
                        if evals_this >= max_evals:
                            break
                        cand = cur.clone()
                        for idx in (a, b):
                            i, k = positions[idx]
                            cand[i, k] = cand[i, k] + 1.0
                        cand = repair(cand)
                        if not legal(cand) or torch.allclose(cand, cur):
                            continue
                        evals_this += 1
                        j, hp, fin = evaluate(cand)
                        if j < cur_j - 1e-9 and (not require_p0 or p0_ok(cand)):
                            cur, cur_j, cur_hard, cur_fin = cand, j, hp, fin
                            improved = True
            history.append({"restart": si, "sweep": sweep, "J": cur_j,
                            "hard_risk": cur_hard, "finish_s": cur_fin,
                            "added": float((cur - w0).sum()), "evals": evals_this})
        if best is None or cur_j < best[0]:
            best = (cur_j, cur, cur_hard, cur_fin)

    if best is None:
        s0, h0v, f0, _ = _eval(risk_model, paths, w0, pause_p, time_steps)
        return WaitResult(wait=w0.clone(), added=torch.zeros_like(w0), soft_risk=s0,
                          hard_risk=h0v, finish_s=f0, feasible=True, fallback=True,
                          history=history, elapsed_s=time.perf_counter() - t0,
                          dp_evals=dp_evals, grad_steps=0, p0_safe=True)

    _, w_best, hard_best, fin_best = best
    soft_best = _eval(risk_model, paths, w_best, pause_p, time_steps)[0]
    p0 = p0_ok(w_best)
    return WaitResult(wait=w_best, added=w_best - w0, soft_risk=soft_best,
                      hard_risk=hard_best, finish_s=fin_best, feasible=True,
                      fallback=False, history=history,
                      elapsed_s=time.perf_counter() - t0, dp_evals=dp_evals,
                      grad_steps=0, p0_safe=bool(p0))


# --------------------------------------------------------------------------- #
def gradient_diagnostics(w0, mask, budget, risk_model, paths, pause_p, time_steps,
                         lam_T=0.05, lam_W=0.01) -> dict:
    """§4 风险梯度诊断 + 有限差分方向校验。"""
    base_finish = _eval(risk_model, paths, w0, pause_p, time_steps)[2]
    u = torch.zeros_like(w0, requires_grad=True)
    w = project_wait(w0 + F.softplus(u) * mask, w0, mask, budget)
    risk = risk_model(paths, w, pause_p, time_steps, collision_mode="soft",
                      gate_mode="soft", return_diagnostics=False)
    loss = (risk["pair_sum_risk"] + lam_T * (risk["finish_cost"] - base_finish)
            + lam_W * (w - w0).abs().sum())
    (g,) = torch.autograd.grad(loss, u)
    soft0, hard0, _, _ = _eval(risk_model, paths, w0, pause_p, time_steps)

    # 有限差分：沿 -g 方向走一小步应使 loss 下降
    eps = 1e-3
    with torch.no_grad():
        w1 = project_wait(w0 + F.softplus(u - eps * g / (g.norm() + 1e-12)) * mask,
                          w0, mask, budget)
        r1 = risk_model(paths, w1, pause_p, time_steps, collision_mode="soft",
                        gate_mode="soft", return_diagnostics=False)
        l1 = (r1["pair_sum_risk"] + lam_T * (r1["finish_cost"] - base_finish)
              + lam_W * (w1 - w0).abs().sum())
    fd_ok = bool(l1 < loss.detach())

    return {
        "soft_risk_init": soft0,
        "hard_risk_init": hard0,
        "loss_init": float(loss.detach()),
        "grad_norm": float(g.norm()),
        "n_nonzero_grad": int((g.abs() > 1e-8).sum()),
        "n_vars": int(g.numel()),
        "fd_descent_ok": fd_ok,
        "fd_loss_after": float(l1),
    }
