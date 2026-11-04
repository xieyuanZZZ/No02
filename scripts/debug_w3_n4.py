"""诊断 N=4 下 W3 为何完全回退到 W0。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget,
    allowed_positions,
    make_mask,
    policy_w2_enum,
    project_wait,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402

cfg = load_config(ROOT / "configs" / "base.yaml")
dt_s = float(cfg["trajectory"]["dt_s"])
T = int(cfg["execution"]["horizon_steps"])
H = int(cfg["trajectory"]["horizon"])
risk = FirstHitRisk.from_config(cfg)
d_safe = risk.d_safe

bank = json.loads((ROOT / "data" / "mpd_delay_plan_bank" / "plan_bank_n4.json")
                  .read_text(encoding="utf-8"))
trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / "plan_bank_n4_trajs.npz")
jt = bank["tasks"][0]
paths = torch.tensor(trajs[jt["task_id"]], dtype=torch.float64)
w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
n = paths.shape[0]
pause = torch.full((n,), 0.1, dtype=torch.float64)
budget = WaitBudget(time_steps=T, horizon=H)
mask = make_mask(H, allowed_positions(H, 5), n)

with torch.no_grad():
    soft0 = risk(paths, w0, pause, T, collision_mode="soft", gate_mode="soft",
                 return_diagnostics=False)
    hard0 = risk(paths, w0, pause, T, collision_mode="hard", gate_mode="hard",
                 return_diagnostics=False)
print("task:", jt["task_id"], " N =", n)
print(f"初始 soft={float(soft0['pair_sum_risk']):.4f} hard={float(hard0['pair_sum_risk']):.4f} "
      f"finish={float(soft0['finish_cost']):.2f}s")

import torch.nn.functional as F

u = torch.zeros_like(w0, requires_grad=True)
opt = torch.optim.Adam([u], lr=0.15)
base_finish = float(soft0["finish_cost"])
print("\niter  loss     soft    hard     added  finish")
for it in range(121):
    opt.zero_grad(set_to_none=True)
    w = project_wait(w0 + F.softplus(u) * mask, w0, mask, budget)
    r = risk(paths, w, pause, T, collision_mode="soft", gate_mode="soft",
             return_diagnostics=False)
    loss = r["pair_sum_risk"] + 0.05 * (r["finish_cost"] - base_finish) + 0.01 * (w - w0).abs().sum()
    loss.backward()
    opt.step()
    if it % 15 == 0:
        with torch.no_grad():
            h = risk(paths, w, pause, T, collision_mode="hard", gate_mode="hard",
                     return_diagnostics=False)
            print(f"{it:4d}  {float(loss):7.3f}  {float(r['pair_sum_risk']):6.3f}  "
                  f"{float(h['pair_sum_risk']):6.3f}  {float((w-w0).sum()):6.2f}  "
                  f"{float(r['finish_cost']):6.2f}")

with torch.no_grad():
    w_cont = project_wait(w0 + F.softplus(u) * mask, w0, mask, budget)
    w_disc = torch.round(w_cont - w0) + w0
    hc = risk(paths, w_disc, pause, T, collision_mode="hard", gate_mode="hard",
              return_diagnostics=False)
    sc = risk(paths, w_disc, pause, T, collision_mode="soft", gate_mode="soft",
              return_diagnostics=False)
print(f"\n连续解: added={float((w_cont-w0).sum()):.2f}  soft={float(sc['pair_sum_risk']):.3f}")
print(f"离散解: added={float((w_disc-w0).sum()):.2f}  hard={float(hc['pair_sum_risk']):.4f} "
      f"(初始 {float(hard0['pair_sum_risk']):.4f})")

r2 = policy_w2_enum(w0, mask, budget, risk, paths, pause, T, lam_T=0.05)
print(f"W2: added={float(r2.added.sum()):.2f}  hard={r2.hard_risk:.4f}  finish={r2.finish_s:.2f}")
