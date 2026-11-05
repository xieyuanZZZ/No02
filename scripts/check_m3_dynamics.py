"""阶段二 §2.4-4：M3 预测运动分布 vs 独立测试 MPC 实测（动力学解释）。

只读取冻结预测里的等待计划与 M3 参数，用**评测真值种子表之外**的种子
单独跑少量 record=True 的 MPC 轨迹，比较：
  * s(t) 的 RMSE（每台机器人、共同时间栅格）
  * 完成时间（M3 分布 vs 实测）
  * 平均/最大速度、HOLD 次数与总保持时长
输出 dynamics_check.csv / dynamics_summary.json / figures。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc  # noqa: E402
from fhd.risk.conditional_execution_model import (  # noqa: E402
    ConditionalExecutionModel, ExecutionModelParams,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402

_setup_fonts()


def _m3_completion(s_hist, v_hist, lengths, times, tol=0.05):
    arrived = s_hist >= (lengths[None, None, :] - tol)
    all_arr = arrived.all(axis=2)              # [T+1,K]
    has = all_arr.any(axis=0)
    idx = np.argmax(all_arr, axis=0)
    comp = np.where(has, times[idx], np.nan)
    return comp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="results/Online_FirstHit_v2/03_risk_baselines")
    ap.add_argument("--m3", default="results/Online_FirstHit_v2/04_dynamic_risk_m3")
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--n-d2", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=20280977)
    args = ap.parse_args()

    baseline_dir = Path(args.baselines)
    m3_dir = Path(args.m3)
    cfg = load_config(ROOT / "configs" / "base.yaml")
    T = int(cfg["execution"]["horizon_steps"])
    dt = 0.1
    horizon_s = T * 0.4
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe
    frozen = json.loads((baseline_dir / "predictions_m0m1m2.json")
                        .read_text(encoding="utf-8"))["predictions"]
    split = json.loads((baseline_dir / "split.json").read_text(encoding="utf-8"))
    test_ids = set(split["2"]["test"]) | set(split["4"]["test"])
    params = {n: ExecutionModelParams.load(m3_dir / f"m3_params_n{n}.json")
              for n in (2, 4)}

    # 每个 N 取最多 3 个任务（p 固定）以控制成本
    keys = []
    seen = {"N2": 0, "N4": 0}
    for key, p in sorted(frozen.items()):
        if p["task_id"] not in test_ids or abs(float(p["p"]) - args.p) > 1e-12:
            continue
        if p["plan"] != "orig":
            continue
        g = p["group"]
        if seen[g] >= 3:
            continue
        seen[g] += 1
        keys.append(key)

    rows = []
    t0 = time.perf_counter()
    trace_example = None
    for key in keys:
        base = frozen[key]
        n = 2 if base["group"] == "N2" else 4
        # 从 plan bank 取几何
        bank_json = "plan_bank.json" if n == 2 else "plan_bank_n4.json"
        trajs_npz = "plan_bank_trajs.npz" if n == 2 else "plan_bank_n4_trajs.npz"
        trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / trajs_npz)
        paths = np.asarray(trajs[base["task_id"]], dtype=np.float64)
        wait = np.asarray(base["wait"], dtype=np.float64)

        kh = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        d2_runs = [simulate_mpc(paths, wait, args.p,
                                MPCSimConfig(d_safe=d_safe, horizon_steps=T),
                                seed=args.seed0 + kh % 1000 + j, record=True)
                   for j in range(args.n_d2)]
        d2 = d2_runs[0]
        model = ConditionalExecutionModel(params[n])
        out = model.predict(paths, wait, args.p, horizon_s, d_safe,
                            n_samples=args.n_samples,
                            seed=909090 + kh % 10000)
        tr = out["trace_arrays"]
        comp_m3 = _m3_completion(tr["s_hist"], tr["v_hist"],
                                 np.full(n, 0.0), tr["times"])
        # 用真实路径长度补上
        from fhd.risk.conditional_execution_model import PathGeometry
        geo = PathGeometry(paths, wait)
        comp_m3 = _m3_completion(tr["s_hist"], tr["v_hist"], geo.length, tr["times"])
        t_common = min(min(r.s_trace.shape[0] for r in d2_runs),
                       tr["s_hist"].shape[0])
        s_pred_mean = tr["s_hist"][:t_common].mean(axis=1)       # [T,N]
        s_obs = np.mean([r.s_trace[:t_common] for r in d2_runs], axis=0)
        rmse = np.sqrt(np.mean((s_pred_mean - s_obs) ** 2, axis=0))
        v_pred_mean = tr["v_hist"][:t_common].mean(axis=1)
        v_obs = np.mean([r.v_trace[:t_common] for r in d2_runs], axis=0)
        hold_steps = (tr["mode_hist"][:t_common] == 2).sum(axis=0).mean(axis=1)
        rows.append({
            "group": base["group"], "task_id": base["task_id"], "plan": base["plan"],
            "p": args.p, "n_agents": n,
            "completion_obs_s": float(np.mean([r.completion_time for r in d2_runs])),
            "completion_m3_mean_s": float(np.nanmean(comp_m3)),
            "completion_m3_p10_s": float(np.nanpercentile(comp_m3, 10)),
            "completion_m3_p90_s": float(np.nanpercentile(comp_m3, 90)),
            "s_rmse_mean_m": float(np.mean(rmse)),
            "s_rmse_max_m": float(np.max(rmse)),
            "v_mean_obs": float(v_obs.mean()),
            "v_mean_m3": float(v_pred_mean.mean()),
            "v_max_obs": float(np.mean([np.max(r.v_trace) for r in d2_runs])),
            "v_max_m3": float(np.max(tr["v_hist"])),
            "hold_count_obs": float(np.mean([
                len(r.pause_events) + int(round(r.actual_wait_time / 0.4))
                for r in d2_runs])),
            "hold_steps_m3": float(np.mean(hold_steps)),
            "obs_timeout": bool(any(r.timeout for r in d2_runs)),
            "obs_infeasible": int(sum(r.infeasible_stops for r in d2_runs)),
        })
        if trace_example is None:
            trace_example = (key, np.asarray(s_obs).copy(), s_pred_mean.copy(),
                             np.arange(t_common) * dt)

    total = time.perf_counter() - t0
    out_dir = m3_dir
    with (out_dir / "dynamics_check.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "p": args.p, "n_keys": len(rows), "n_samples": args.n_samples,
        "elapsed_s": total,
        "mean_abs_completion_diff_s": float(np.mean(
            [abs(r["completion_obs_s"] - r["completion_m3_mean_s"]) for r in rows])),
        "mean_s_rmse_m": float(np.mean([r["s_rmse_mean_m"] for r in rows])),
        "mean_speed_diff": float(np.mean([r["v_mean_m3"] - r["v_mean_obs"] for r in rows])),
        "note": "仅用于动力学解释，不参与风险评分；种子独立于真值 MC",
    }
    (out_dir / "dynamics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if trace_example is not None:
        key, s_obs, s_pred, times = trace_example
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for i in range(min(2, s_obs.shape[1])):
            ax.plot(times, s_obs[:, i], "o", ms=2, alpha=0.5, label=f"实测 robot{i}")
            ax.plot(times, s_pred[:, i], "-", lw=1.3, label=f"M3 mean robot{i}")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("弧长 s (m)")
        ax.set_title(f"s(t) 对比：{key}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / "m3_vs_mpc_strace.png", dpi=130)
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
