"""阶段二 §2.4/§2.5：M0/M1/M2/M3 冻结盲测、消融、分组不确定性与开销。

严格顺序：
  1. 读取已冻结的 M0/M1/M2 预测（来自 03_risk_baselines）；
  2. 用**校准/验证集**拟合的 M3 参数生成 M3（含 3 个消融）预测并落盘 SHA256；
  3. 预测冻结之后，才用**独立真值种子表**跑测试任务的 MPC MC；
  4. 评分 + 任务级 bootstrap + 可靠性 + Top-K + 动力学一致性 + 耗时。
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fhd.evaluation.risk_dataset import (  # noqa: E402
    bootstrap_task_ci, pairwise_stats, path_pair_min_distance,
    reliability_table,
)
from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc  # noqa: E402
from fhd.geometry.static_maps import load_obstacles_for_map  # noqa: E402
from fhd.risk.conditional_execution_model import (  # noqa: E402
    ConditionalExecutionModel, ExecutionModelParams,
)
from fhd.risk.dynamic_first_hit import predict_pair_risks  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

_setup_fonts()
ABLATIONS = {
    "m3": {},
    "m3_no_braking": {"use_braking": False},
    "m3_no_modes": {"use_modes": False},
    "m3_no_speed": {"use_speed": False},
}


def _sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True)
                          .encode()).hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _model_samples(m3_dir: Path, n: int, default: int) -> int:
    p = m3_dir / "m3_params_n{n}.json".format(n=n)
    if not p.exists():
        return default
    d = _load_json(p)
    return int(d.get("calib_notes", {}).get("validation", {})
               .get("n_samples_selected", default))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="results/Online_FirstHit_v2/03_risk_baselines")
    ap.add_argument("--m3", default="results/Online_FirstHit_v2/04_dynamic_risk_m3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-mc", type=int, default=30)
    ap.add_argument("--ablation-samples", type=int, default=120)
    ap.add_argument("--truth-seed0", type=int, default=20280915)
    ap.add_argument("--max-keys", type=int, default=0, help="调试用；0=全部测试任务")
    args = ap.parse_args()

    baseline_dir = Path(args.baselines)
    m3_dir = Path(args.m3)
    out_dir = Path(args.out) if args.out else m3_dir
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.m3_bench", out_dir / "m3_bench.log")
    log_section(logger, "M3 冻结盲测")

    cfg = load_config(ROOT / "configs" / "base.yaml")
    set_seed(int(cfg["seed"]))
    T = int(cfg["execution"]["horizon_steps"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    horizon_s = T * dt_s
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe

    frozen = _load_json(baseline_dir / "predictions_m0m1m2.json")
    preds_m012 = frozen["predictions"]
    split = _load_json(baseline_dir / "split.json")
    test_ids = set()
    for n in ("2", "4"):
        test_ids |= set(split[n]["test"])

    params = {n: ExecutionModelParams.load(m3_dir / f"m3_params_n{n}.json")
              for n in (2, 4)}
    n_samples = {n: _model_samples(m3_dir, n, 200) for n in (2, 4)}
    logger.info("M3 采样数：N2=%d N4=%d", n_samples[2], n_samples[4])

    # 测试任务元数据（只取测试 split）
    task_meta = {}
    test_keys = []
    for key, p in preds_m012.items():
        if p["task_id"] not in test_ids:
            continue
        n = 2 if p["group"] == "N2" else 4
        if p["task_id"] not in task_meta:
            task_meta[p["task_id"]] = {
                "group": p["group"], "map_id": p["map_id"], "n_agents": n,
                "paths": None,          # 下面从 plan bank 填入真实几何
            }
        test_keys.append(key)
    # 从 plan bank 取真实几何
    bank_cache = {}
    for n in (2, 4):
        bank_json = "plan_bank.json" if n == 2 else "plan_bank_n4.json"
        trajs_npz = "plan_bank_trajs.npz" if n == 2 else "plan_bank_n4_trajs.npz"
        bank = _load_json(ROOT / "data" / "mpd_delay_plan_bank" / bank_json)
        trajs = np.load(ROOT / "data" / "mpd_delay_plan_bank" / trajs_npz)
        for t in bank["tasks"]:
            bank_cache[t["task_id"]] = (n, np.asarray(trajs[t["task_id"]], dtype=np.float64),
                                        str(t.get("map_id", "")))
    geom = {}
    for tid in task_meta:
        n, paths, mid = bank_cache[tid]
        task_meta[tid]["paths"] = paths
        task_meta[tid]["map_id"] = mid
        geom[tid] = path_pair_min_distance(paths)

    if args.max_keys:
        test_keys = test_keys[:args.max_keys]
    logger.info("测试预测键数：%d", len(test_keys))

    # ---------------- 1) M3 预测 + 冻结 ----------------
    m3_pred = {}
    t_pred = time.perf_counter()
    for key in test_keys:
        base = preds_m012[key]
        tid, plan, p = base["task_id"], base["plan"], float(base["p"])
        n = task_meta[tid]["n_agents"]
        paths = task_meta[tid]["paths"]
        wait = np.asarray(base["wait"], dtype=np.float64)
        seed = 606000 + int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100000
        for name, kwargs in ABLATIONS.items():
            ns = n_samples[n] if name == "m3" else args.ablation_samples
            out = predict_pair_risks(paths, wait, p, horizon_s, params[n],
                                     d_safe=d_safe, n_samples=ns, seed=seed,
                                     **kwargs)
            m3_pred.setdefault(key, {})[name] = {
                "pair_risks": np.asarray(out["pair_risks"]).tolist(),
                "team_collision_estimate": float(out["team_collision_estimate"]),
                "runtime_ms": float(out["runtime_ms"]),
                "n_samples": int(ns),
                "model_info": out["model_info"],
            }
        logger.info("M3 冻结预测 %s", key)
    pred_elapsed = time.perf_counter() - t_pred
    frozen_m3 = {
        "predictions": m3_pred,
        "sha256": _sha256(m3_pred),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": pred_elapsed,
        "n_samples": n_samples,
        "ablation_samples": args.ablation_samples,
        "note": "生成于任何测试任务 MPC 真值之前；M0-M2 直接引用早先冻结文件",
        "m0m1m2_sha256": frozen.get("sha256"),
    }
    (out_dir / "predictions_m3.json").write_text(
        json.dumps(frozen_m3, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("M3 预测冻结 SHA256=%s（%.1fs）", frozen_m3["sha256"][:16], pred_elapsed)

    # ---------------- 2) 冻结后才跑真值 MC ----------------
    truth = {}
    truth_rows = []
    t_truth = time.perf_counter()
    for ti, key in enumerate(sorted(test_keys)):
        base = preds_m012[key]
        tid, plan, p = base["task_id"], base["plan"], float(base["p"])
        paths = task_meta[tid]["paths"]
        wait = np.asarray(base["wait"], dtype=np.float64)
        obs_obj, obs_meta = load_obstacles_for_map(task_meta[tid]["map_id"])
        flags, hits = [], []
        for k in range(args.n_mc):
            seed = args.truth_seed0 + ti * 1000 + k
            r = simulate_mpc(paths, wait, p,
                             MPCSimConfig(d_safe=d_safe, horizon_steps=T),
                             seed=int(seed), record=False, obstacles=obs_obj,
                             obstacle_meta=obs_meta)
            flags.append(np.asarray(r.pair_collision, dtype=bool))
            hits.append({
                "group": base["group"], "task_id": tid, "plan": plan, "p": p,
                "seed": seed, "collision": bool(r.collision),
                "first_hit_time": r.first_collision_time,
                "all_arrived": bool(r.all_arrived), "timeout": bool(r.timeout),
                "completion": float(r.completion_time),
                "min_pair_distance": float(np.nanmin(r.pair_min_distance)),
                "infeasible_stops": int(r.infeasible_stops),
                "mpc_failures": int(r.mpc_failures),
                "pair_flags": json.dumps(np.asarray(r.pair_collision, dtype=int).tolist()),
            })
        truth[key] = np.asarray(flags, dtype=bool)
        truth_rows.extend(hits)
        logger.info("真值 %s 完成", key)
    truth_elapsed = time.perf_counter() - t_truth

    with (out_dir / "truth_per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(truth_rows[0].keys()))
        w.writeheader()
        w.writerows(truth_rows)

    # ---------------- 3) 评分 ----------------
    rows = []
    team_rows = []
    for key in sorted(test_keys):
        base = preds_m012[key]
        tid, plan, p = base["task_id"], base["plan"], float(base["p"])
        group = base["group"]
        pairs = base["pairs"]
        obs = truth[key].mean(axis=0)
        team_obs = float(np.mean([np.any(f) for f in truth[key]]))
        m3 = m3_pred[key]
        team_rows.append({
            "group": group, "task_id": tid, "plan": plan, "p": p,
            "n_pairs": len(pairs), "n_mc": args.n_mc,
            "m3_team_estimate": m3["m3"]["team_collision_estimate"],
            "m0_union_bound": float(np.clip(np.sum(base["m0"]), 0, 1)),
            "observed_team_rate": team_obs,
            "observed_pairs_hit": int(np.sum(obs > 0)),
        })
        for k, (i, j) in enumerate(pairs):
            rows.append({
                "group": group, "task_id": tid, "map_id": task_meta[tid]["map_id"],
                "plan": plan, "p": p, "pair": f"{i}-{j}",
                "m0": float(base["m0"][k]), "m1": float(base["m1"][k]),
                "m2": float(base["m2"][k]),
                "m3": float(m3["m3"]["pair_risks"][k]),
                "m3_no_braking": float(m3["m3_no_braking"]["pair_risks"][k]),
                "m3_no_modes": float(m3["m3_no_modes"]["pair_risks"][k]),
                "m3_no_speed": float(m3["m3_no_speed"]["pair_risks"][k]),
                "obs": float(obs[k]), "n_mc": args.n_mc,
                "geometry_min_distance_m": float(geom[tid][i, j]),
                "crossing": bool(geom[tid][i, j] < 0.5),
            })
    with (out_dir / "risk_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    models = ["m0", "m1", "m2", "m3", "m3_no_braking", "m3_no_modes", "m3_no_speed"]

    def sel(**filters):
        out = rows
        for k, v in filters.items():
            out = [r for r in out if (abs(r[k] - v) < 1e-12 if isinstance(v, float)
                                      else r[k] == v)]
        return out

    def stats_for(sel_rows):
        return {m: pairwise_stats([r[m] for r in sel_rows],
                                  [r["obs"] for r in sel_rows]) for m in models}

    def task_mae_diff(sel_rows, model):
        by_task = {}
        for tid in sorted({r["task_id"] for r in sel_rows}):
            s = [r for r in sel_rows if r["task_id"] == tid]
            mae_m = np.mean([abs(r[model] - r["obs"]) for r in s])
            mae_0 = np.mean([abs(r["m0"] - r["obs"]) for r in s])
            by_task[tid] = float(mae_m - mae_0)
        return bootstrap_task_ci(by_task)

    def topk(sel_rows, model, k=1):
        hits = []
        for tid in sorted({r["task_id"] for r in sel_rows}):
            for plan in ("orig", "w2plus"):
                for p in sorted({r["p"] for r in sel_rows}):
                    s = [r for r in sel_rows if r["task_id"] == tid
                         and r["plan"] == plan and abs(r["p"] - p) < 1e-12]
                    if not s:
                        continue
                    order = sorted(range(len(s)), key=lambda i: -s[i][model])
                    top = order[:min(k, len(order))]
                    hits.append(any(s[i]["obs"] > 0 for i in top))
        return float(np.mean(hits)) if hits else float("nan")

    summary = {
        "stage": "phase2_m3_blind_test",
        "n_mc": args.n_mc, "n_pairs": len(rows), "n_tasks": len({r["task_id"] for r in rows}),
        "ps": sorted({r["p"] for r in rows}),
        "prediction_sha256": frozen_m3["sha256"],
        "m0m1m2_sha256": frozen.get("sha256"),
        "truth_seeds": {"seed0": args.truth_seed0, "n_mc": args.n_mc},
        "overall": stats_for(rows),
        "delta_mae_vs_m0": {m: task_mae_diff(rows, m) for m in models[1:]},
        "by_group": {g: stats_for(sel(group=g)) for g in ("N2", "N4")},
        "by_p": {str(p): stats_for(sel(p=p)) for p in sorted({r["p"] for r in rows})},
        "by_plan": {pl: stats_for(sel(plan=pl)) for pl in ("orig", "w2plus")},
        "by_geometry": {
            "crossing": stats_for([r for r in rows if r["crossing"]]),
            "non_crossing": stats_for([r for r in rows if not r["crossing"]]),
        },
        "reliability": {m: reliability_table([r[m] for r in rows],
                                             [r["obs"] for r in rows])
                        for m in ("m0", "m1", "m2", "m3")},
        "top1_hit_rate": {m: topk(rows, m, 1) for m in models},
        "top3_hit_rate": {m: topk(rows, m, 3) for m in models},
        "team": {
            "n_keys": len(team_rows),
            "m3_team_mae": float(np.mean([abs(r["m3_team_estimate"]
                                             - r["observed_team_rate"])
                                          for r in team_rows])),
            "m3_team_bias": float(np.mean([r["m3_team_estimate"]
                                           - r["observed_team_rate"] for r in team_rows])),
            "m0_union_bound_coverage": float(np.mean(
                [r["m0_union_bound"] + 1e-12 >= r["observed_team_rate"]
                 for r in team_rows])),
            "note": "两两风险之和只作为并集上界，不是全队碰撞概率",
        },
        "runtime": {
            "m3_prediction_total_s": pred_elapsed,
            "truth_mc_total_s": truth_elapsed,
            "m3_prediction_ms_mean": float(np.mean(
                [v["m3"]["runtime_ms"] for v in m3_pred.values()])),
            "m3_prediction_ms_p95": float(np.percentile(
                [v["m3"]["runtime_ms"] for v in m3_pred.values()], 95)),
            "m3_n_samples": n_samples,
            "m2_dp_calls": int(24 * len(test_keys)),
            "m2_dp_calls_note": "M2 每次预测跑 n_draws=24 次 DP（见 03_risk_baselines summary）",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    with (out_dir / "team_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(team_rows[0].keys()))
        w.writeheader()
        w.writerows(team_rows)

    # ---------------- 4) 图 ----------------
    obs = np.array([r["obs"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, m, name in zip(axes, ("m0", "m2", "m3"),
                           ("M0 原始 DP", "M2 时长分布", "M3 状态模型")):
        pred = np.array([r[m] for r in rows])
        ax.scatter(pred, obs, s=16, alpha=0.7)
        lim = [0, max(0.6, float(max(pred.max(), obs.max())) * 1.05)]
        ax.plot(lim, lim, "k--", lw=1)
        st = pairwise_stats(pred, obs)
        ax.set_title(f"{name}  MAE={st['mae']:.3f}  rho={st['spearman']:.3f}", fontsize=10)
        ax.set_xlabel("预测两两首次碰撞概率")
        ax.set_ylabel("MPC MC 两两碰撞频率")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "m0m2m3_scatter.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = ["M0", "M1", "M2", "M3", "M3-无制动", "M3-无模式", "M3-无速度"]
    vals = [summary["overall"][m]["mae"] for m in models]
    ax.bar(labels, vals, color=["#4c72b0", "#5d8bc4", "#7aa3d3", "#dd8452", "#c98b6b",
                                "#b57a5e", "#a06a50"])
    ax.set_ylabel("pair MAE")
    ax.set_title("冻结测试集：预测误差")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "model_mae.png", dpi=130)
    plt.close(fig)

    print(json.dumps({m: {k: round(v, 4) for k, v in summary["overall"][m].items()
                          if k != "n"} for m in ("m0", "m1", "m2", "m3")},
                     ensure_ascii=False, indent=1))
    print(f"输出：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
