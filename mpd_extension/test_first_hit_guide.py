"""对 `FirstHitRiskGuide` 做接口与符号自检，并量化"软风险饱和"这一实测现象。

三项检查：
  1. 形状/类型：返回与输入同形状的梯度；
  2. 符号：沿 +guide 走一小步，目标函数必须**下降**（官方契约的核心）；
  3. 反号会变坏：沿 -guide 走，目标必须上升。

另外单独测量：首碰软风险在不同最小间距下的梯度量级，
用于说明"深重叠时软核饱和、梯度消失"（写进报告 §7）。

跑法（WSL）：bash /mnt/d/mpd_official/wsl_setup/run_guide_test.sh
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/mnt/d/paper2/src")
sys.path.insert(0, "/mnt/d/mpd_official/mpd_extension")

from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline  # noqa: E402

from first_hit_guide import FirstHitGuideConfig, FirstHitRiskGuide  # noqa: E402


class IdentityDataset:
    """归一化当成恒等（只验证接口与符号，不涉及统计量）。"""

    def unnormalize_control_points(self, cp_norm):
        return cp_norm


class MinDistanceSurrogate:
    """把"最小间距小于 d_safe"写成光滑代价，用作符号自检的目标函数。

    它与首碰风险同号（越近越差），但不会像 DP 那样在深重叠时饱和，
    因此可以干净地验证 -∂L/∂cp 的方向约定。
    """

    def __init__(self, d_safe: float = 0.35, margin: float = 0.35):
        self.d_safe = d_safe
        self.margin = margin

    def __call__(self, paths: torch.Tensor, wait, pause_p, time_steps, **kwargs):
        d = torch.cdist(paths[0], paths[1]).min()
        gap = torch.relu(self.d_safe - d)
        risk = (gap / self.margin).pow(2)
        return {"pair_sum_risk": risk, "finish_cost": paths.new_zeros(())}


def main() -> int:
    torch.manual_seed(0)
    dtype = torch.float64
    bspline = ParametricTrajectoryBspline(
        n_control_points=18, degree=5, num_T_pts=128, trajectory_duration=10.0,
        zero_vel_at_start_and_goal=True, zero_acc_at_start_and_goal=True,
        tensor_args={"device": "cpu", "dtype": dtype},
    )
    B, N, K, D = 1, 2, 18, 2
    T = 96

    def straight(s, g):
        lin = torch.linspace(0, 1, K, dtype=dtype)
        return torch.stack([s + (g - s) * u for u in lin])

    def cp_from(starts, goals, noise=0.01):
        cp = torch.stack([straight(starts[n], goals[n]) for n in range(N)])
        cp = cp + torch.randn_like(cp) * noise
        cp[:, 0, :] = starts
        cp[:, -1, :] = goals
        return cp

    # ---------------- 1) 符号自检（用不饱和的替代目标） ----------------
    off = 0.25
    starts = torch.tensor([[-1.0, 0.0], [-1.0, off]], dtype=dtype)
    goals = torch.tensor([[1.0, 0.0], [1.0, off]], dtype=dtype)
    surrogate = MinDistanceSurrogate()
    cfg = FirstHitGuideConfig(lam_R=1.0, lam_T=0.0, n_agents=N, batch_candidates=B,
                              start=starts, goal=goals)
    guide = FirstHitRiskGuide(surrogate, bspline, IdentityDataset(), cfg, time_steps=T)

    cp = cp_from(starts, goals).requires_grad_(True)
    g = guide(cp)
    print(f"[1] 形状 {tuple(g.shape)}（期望 {(N, K, D)}），|g|={float(g.norm()):.4e}")
    assert g.shape == cp.shape and float(g.norm()) > 1e-8

    def loss_of(x):
        q = bspline.get_q_trajectory(x, starts, goals, get_type=("pos",),
                                     get_time_representation=False)["pos"]
        if q.shape[1] > T:
            q = q.index_select(1, torch.linspace(0, q.shape[1] - 1, T).round().long())
        return float(surrogate(q.reshape(N, T, D), None, None, T)["pair_sum_risk"])

    base = loss_of(cp.detach())
    eps = 2e-3
    up = loss_of(cp.detach() + eps * g.detach())
    down = loss_of(cp.detach() - eps * g.detach())
    print(f"[2] L(cp)={base:.6e}  L(+eps*guide)={up:.6e}  L(-eps*guide)={down:.6e}")
    ok = up < base < down
    print(f"[3] 符号自检：{'通过（+guide 降风险，-guide 升风险）' if ok else '失败'}")
    assert ok, "guide 符号反了：沿 +guide 反而更危险"

    # ---------------- 2) 真实首碰风险的梯度 vs 间距（饱和现象） ----------------
    print("")
    print("[4] 真实首碰软风险：不同最小间距下的损失与梯度量级")
    print("    间距[m]   pair_sum_risk        |dR/dcp|")
    risk_model = FirstHitRisk.from_config(load_config("/mnt/d/paper2/configs/base.yaml"))
    for sep in (0.0, 0.15, 0.30, 0.34, 0.38, 0.50, 0.80):
        st = torch.tensor([[-1.0, 0.0], [-1.0, sep]], dtype=dtype)
        gl = torch.tensor([[1.0, 0.0], [1.0, sep]], dtype=dtype)
        c = cp_from(st, gl, noise=0.0).requires_grad_(True)
        q = bspline.get_q_trajectory(c, st, gl, get_type=("pos",),
                                     get_time_representation=False)["pos"]
        q = q.reshape(N, q.shape[1], D)
        w = torch.zeros(N, q.shape[1] - 1, dtype=dtype)
        p = torch.full((N,), 0.1, dtype=dtype)
        r = risk_model(q, w, p, q.shape[1], collision_mode="soft", gate_mode="soft",
                       return_diagnostics=False)
        (gg,) = torch.autograd.grad(r["pair_sum_risk"], c, retain_graph=False)
        print(f"    {sep:5.2f}     {float(r['pair_sum_risk']):.6f}       "
              f"{float(gg.norm()):.3e}")

    print("")
    print("[5] 官方约定核对：CostGuideManager 返回 -dC/dcp_norm；本实现一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
