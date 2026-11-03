"""调试：为什么风险对控制点没有梯度。"""

import sys

import torch

sys.path.insert(0, "/mnt/d/paper2/src")
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402

cfg = load_config("/mnt/d/paper2/configs/base.yaml")
risk = FirstHitRisk.from_config(cfg)
print("d_safe =", risk.d_safe)

T = 96
u = torch.linspace(0, 1, T, dtype=torch.float64)
# 两条交叉直线（官方 2D 世界尺度）
a = torch.stack([-1 + 2 * u, torch.zeros_like(u)], dim=-1)
b = torch.stack([torch.zeros_like(u), -1 + 2 * u], dim=-1)
paths = torch.stack([a, b])
print("paths", paths.shape, "min pair dist =",
      float(torch.linalg.vector_norm(paths[0] - paths[1], dim=-1).min()))

paths.requires_grad_(True)
w = torch.zeros(2, T - 1, dtype=torch.float64)
p = torch.full((2,), 0.1, dtype=torch.float64)
out = risk(paths, w, p, T, collision_mode="soft", gate_mode="soft",
           return_diagnostics=True)
print("pair_sum_risk =", float(out["pair_sum_risk"]))
print("finish_cost   =", float(out["finish_cost"]))
for k, v in out.items():
    if torch.is_tensor(v) and v.numel() <= 4:
        print("  ", k, v.detach().tolist() if v.numel() > 1 else float(v))
g = torch.autograd.grad(out["pair_sum_risk"], paths, allow_unused=True)[0]
print("grad norm wrt paths:", None if g is None else float(g.norm()))
g2 = torch.autograd.grad(out["finish_cost"], paths, allow_unused=True)[0]
print("grad norm finish:", None if g2 is None else float(g2.norm()))
