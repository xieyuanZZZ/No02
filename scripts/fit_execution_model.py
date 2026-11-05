"""阶段二 §2.2/§2.3：从校准轨迹拟合 M3 执行模型，并在**验证集**选择采样数。

输入：`collect_execution_risk_data.py` 的 traces_n*.npz / events.jsonl / split.json
输出：
  m3_params_n2.json / m3_params_n4.json   执行模型参数 + 选定采样数
  m3_fit_summary.json                     拟合样本量、参数、验证 MAE、耗时
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.evaluation.risk_dataset import load_bank, pairwise_stats  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget, allowed_positions, make_mask, policy_w0, policy_w2plus,
)
from fhd.risk.conditional_execution_model import (  # noqa: E402
    ConditionalExecutionModel, ExecutionModelParams, fit_execution_model,
)
from fhd.risk.dynamic_first_hit import predict_pair_risks  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402


def _load_traces(data_dir: Path, n: int):
    z = np.load(data_dir / f"traces_n{n}.npz")
    idx = json.loads((data_dir / "run_index.json").read_text(encoding="utf-8"))
    events = {}
    for line in (data_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        events[int(d["run_id"])] = d["robot_events"]
    runs = [r for r in idx if r["n_agents"] == n]
    traces, evs, labels, valid_idx = [], [], [], []
    # run_id 是全局编号；每个 traces_n*.npz 只含该 N 的本地行，必须用本地下标
    for local_idx, r in enumerate(runs):
        rid = int(r["run_id"])
        tv = int(r["T_valid"])
        traces.append({
            "s": z["s"][local_idx, :tv].astype(np.float64),
            "v": z["v"][local_idx, :tv].astype(np.float64),
            "v_plan": z["v_plan"][local_idx, :tv].astype(np.float64),
            "modes": z["modes"][local_idx, :tv].astype(np.int64),
            "T_valid": tv,
        })
        evs.append(events.get(rid, []))
        labels.append(z["labels"][local_idx].astype(bool))
        valid_idx.append(rid)
    return runs, traces, evs, np.asarray(labels, dtype=bool)


def _plans_for_val(n: int, split: dict, risk, cfg) -> dict:
    """重建验证任务的等待计划；ti 与采集脚本中的 calib+val 顺序一致。"""
    tasks, trajs = load_bank(n, ROOT)
    by_id = {t["task_id"]: t for t in tasks}
    ordered = [by_id[t] for t in split[str(n)]["calib"] + split[str(n)]["val"]]
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    budget = WaitBudget(time_steps=T, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), n)
    out = {}
    for ti, jt in enumerate(ordered):
        if jt["task_id"] not in set(split[str(n)]["val"]):
            continue
        paths = torch.tensor(np.asarray(jt["paths_m"]), dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        pause01 = torch.full((n,), 0.1, dtype=torch.float64)
        r0 = policy_w0(w0, mask, budget, risk, paths, pause01, T)
        r2 = policy_w2plus(w0, mask, budget, risk, paths, pause01, T,
                           lam_T=0.05, max_evals=250, restarts=2, seed=ti)
        out[jt["task_id"]] = {"orig": r0.wait.numpy(), "w2plus": r2.wait.numpy(),
                              "paths": np.asarray(trajs[jt["task_id"]], dtype=np.float64)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/Online_FirstHit_v2/04_dynamic_risk_m3/data")
    ap.add_argument("--out", default="results/Online_FirstHit_v2/04_dynamic_risk_m3")
    ap.add_argument("--samples", default="100,200,400")
    args = ap.parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "base.yaml")
    set_seed(int(cfg["seed"]))
    risk = FirstHitRisk.from_config(cfg)
    T = int(cfg["execution"]["horizon_steps"])
    d_safe = risk.d_safe
    split = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    candidates = [int(x) for x in args.samples.split(",")]

    summary = {"candidates": candidates, "by_n": {}}
    t_start = time.perf_counter()
    for n in (2, 4):
        runs, traces, events, labels = _load_traces(data_dir, n)
        params = fit_execution_model(traces, events, dt=0.1, dt_slot=0.4, v_max=1.5)
        model = ConditionalExecutionModel(params)
        val_plans = _plans_for_val(n, split, risk, cfg)

        # 验证集观测：run_index/traces 里的 labels 只来自 calib/val
        val_runs = [k for k, r in enumerate(runs) if r["split"] == "val"]
        obs_by_key = {}
        for k in val_runs:
            r = runs[k]
            key = (r["task_id"], r["plan"], float(r["p"]))
            obs_by_key.setdefault(key, []).append(labels[k])
        obs_mean = {key: np.mean(np.stack(v), axis=0) for key, v in obs_by_key.items()}

        cand_results = {}
        for n_samples in candidates:
            preds, obss = [], []
            t0 = time.perf_counter()
            for (tid, plan, p), obs in sorted(obs_mean.items()):
                plan_data = val_plans[tid]
                out = predict_pair_risks(
                    plan_data["paths"], plan_data[plan], p, T * 0.4, params,
                    d_safe=d_safe, n_samples=n_samples, seed=424242)
                preds.append(np.asarray(out["pair_risks"]))
                obss.append(np.asarray(obs))
            pred = np.concatenate(preds) if preds else np.zeros(0)
            obs = np.concatenate(obss) if obss else np.zeros(0)
            st = pairwise_stats(pred, obs)
            cand_results[n_samples] = {**st, "wall_s": time.perf_counter() - t0}
        best = min(cand_results, key=lambda s: cand_results[s]["mae"]
                   if np.isfinite(cand_results[s]["mae"]) else 1e9)
        params.calib_notes["validation"] = {"n_samples_selected": best,
                                            "by_n_samples": cand_results}
        params.save(out_dir / f"m3_params_n{n}.json")
        summary["by_n"][str(n)] = {
            "params": params.to_dict(),
            "n_runs": len(runs), "n_val_runs": len(val_runs),
            "val_pairs": len(obs_mean),
            "samples_selected": best,
            "validation": cand_results,
        }
        print(f"[N{n}] a_acc={params.a_acc:.3f} a_brake={params.a_brake:.3f} "
              f"v_scale={params.v_scale:.3f} restart={params.restart_time:.2f} "
              f"pause_n={len(params.pause_durations)} n_samples={best} "
              f"val_mae={cand_results[best]['mae']:.4f}")

    summary["elapsed_s"] = time.perf_counter() - t_start
    (out_dir / "m3_fit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出：{out_dir}  耗时 {summary['elapsed_s']/60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
