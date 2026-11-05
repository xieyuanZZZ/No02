"""阶段二 §2.1：修复后的 M0/M1/M2 对照（无数据泄漏、M2 抽样索引正确）。

流程（严格顺序）：
  1. 按 map_id 切分 calib/val/test；
  2. **只用 calib** 的 p=0 MPC 运行拟合段时长模型（M1 均值 + M2 残差池）；
     校准不接触任何碰撞标签；
  3. **只输入测试任务的静态规划与 p**，生成并落盘 M0/M1/M2 预测，计算 SHA256；
  4. 预测冻结后，才用**独立的真值种子表**跑测试任务的 MPC MC；
  5. 评分 + 任务级 bootstrap 区间 + 分组统计。

关键修复：
  * 旧脚本先在测试任务上跑 simulate_mpc 估 dur_by_p，再在同一任务上评测；
  * 旧 M2 抽样 `samp[idx[:,None,None]][:,0,:]` 固定取 robot 0；
  * 旧 M2 把含随机暂停的实测时长送进 DP，又乘 (1-p)，重复计入等待。
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
    bootstrap_task_ci,
    build_splits,
    load_bank,
    pairwise_stats,
    reliability_table,
    save_split,
)
from fhd.evaluation.visualizations import _setup_fonts  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc  # noqa: E402
from fhd.geometry.static_maps import load_obstacles_for_map  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget, allowed_positions, make_mask, policy_w0, policy_w2plus,
)
from fhd.risk.baselines_m0m1m2 import pair_index, predict_m0m1m2  # noqa: E402
from fhd.risk.duration_model import DurationModel, segment_features  # noqa: E402
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

_setup_fonts()


def _collect_calib_runs(n_agents, tasks, trajs, risk, cfg, seeds, logger):
    """p=0 校准运行：只取段行驶时长，不读取碰撞标签。"""
    T, H = int(cfg["execution"]["horizon_steps"]), int(cfg["trajectory"]["horizon"])
    d_safe = risk.d_safe
    budget = WaitBudget(time_steps=T, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), n_agents)
    pause01 = torch.full((n_agents,), 0.1, dtype=torch.float64)
    feats, targets = [], []
    n_censored = n_valid = 0
    plan_cache = {}
    for ti, jt in enumerate(tasks):
        paths_np = np.asarray(trajs[jt["task_id"]], dtype=np.float64)
        paths_t = torch.tensor(paths_np, dtype=torch.float64)
        w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        r0 = policy_w0(w0, mask, budget, risk, paths_t, pause01, T)
        r2 = policy_w2plus(w0, mask, budget, risk, paths_t, pause01, T,
                           lam_T=0.05, max_evals=250, restarts=2, seed=ti)
        plan_cache[jt["task_id"]] = {"orig": r0.wait.numpy(), "w2plus": r2.wait.numpy()}
        for plan_name, w_np in plan_cache[jt["task_id"]].items():
            X = segment_features(paths_np, w_np)
            for si, seed in enumerate(seeds):
                r = simulate_mpc(paths_np, w_np, 0.0,
                                 MPCSimConfig(d_safe=d_safe, horizon_steps=T),
                                 seed=int(seed), record=True)
                for i in range(n_agents):
                    segs = r.segment_times[i]
                    y = np.array([s["dt"] / 0.4 if s["dt"] is not None else np.nan
                                  for s in segs], dtype=np.float64)
                    n_valid += int(np.isfinite(y).sum())
                    n_censored += int((~np.isfinite(y)).sum())
                    feats.append(X[i])
                    targets.append(y)
        logger.info("[N%d] 校准 %s 完成", n_agents, jt["task_id"])
    X = np.stack(feats)
    y = np.stack(targets)
    return X, y, {"n_valid_segments": n_valid, "n_censored_segments": n_censored,
                  "coverage": n_valid / max(n_valid + n_censored, 1),
                  "plan_cache": {k: {p: v.tolist() for p, v in d.items()}
                                 for k, d in plan_cache.items()}}


def _path_pair_min_distance(paths: np.ndarray) -> np.ndarray:
    """两两折线之间的最小距离（米），用于几何交叉分组。"""
    n = paths.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a0 = paths[i, :-1]
            b0 = paths[i, 1:]
            c0 = paths[j, :-1]
            d0 = paths[j, 1:]
            best = np.inf
            for k in range(a0.shape[0]):
                a, b = a0[k], b0[k]
                for m in range(c0.shape[0]):
                    c, d = c0[m], d0[m]
                    best = min(best, _segment_segment_distance(a, b, c, d))
            out[i, j] = out[j, i] = best
    return out


def _segment_segment_distance(a, b, c, d) -> float:
    def seg_point(p, q, x):
        pq = q - p
        L2 = float(pq @ pq)
        if L2 < 1e-12:
            return float(np.linalg.norm(x - p))
        t = float(np.clip((x - p) @ pq / L2, 0.0, 1.0))
        return float(np.linalg.norm(p + t * pq - x))

    # 端点投影 + 相交判定（二维叉积）
    r = b - a
    s = d - c
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) > 1e-12:
        t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / denom
        u = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(seg_point(a, b, c), seg_point(a, b, d),
               seg_point(c, d, a), seg_point(c, d, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-mc", type=int, default=30)
    ap.add_argument("--n-draws", type=int, default=24)
    ap.add_argument("--ps", default="0,0.05,0.1,0.2")
    ap.add_argument("--calib-seeds", default="91001,91002")
    ap.add_argument("--truth-seed0", type=int, default=20280901)
    ap.add_argument("--ratios", default="0.5,0.2,0.3")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "base.yaml")
    set_seed(int(cfg["seed"]))
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe
    ps = [float(x) for x in args.ps.split(",")]
    calib_seeds = [int(x) for x in args.calib_seeds.split(",")]
    ratios = tuple(float(x) for x in args.ratios.split(","))

    run_dir = Path(args.out) if args.out else new_run_dir("Online_FirstHit_v2", "03_m1m2")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.m1m2_v2", run_dir / "logs" / "m1m2.log")
    log_section(logger, "M0/M1/M2 无泄漏对照（阶段二）")

    tasks_by_n, trajs_by_n = {}, {}
    for n_agents in (2, 4):
        tasks, trajs = load_bank(n_agents, ROOT)
        tasks_by_n[n_agents] = tasks
        trajs_by_n[n_agents] = trajs
    splits = build_splits(tasks_by_n, ROOT, ratios=ratios, seed=20261105)
    split_manifest = {str(n): b.manifest for n, b in splits.items()}
    save_split(split_manifest, run_dir / "split.json")
    logger.info("切分：%s", {n: b.manifest["counts"] for n, b in splits.items()})

    # ---------------- 1) 校准（仅 calib，p=0） ----------------
    models: dict[int, DurationModel] = {}
    calib_meta = {}
    for n_agents in (2, 4):
        X, y, meta = _collect_calib_runs(n_agents, splits[n_agents].calib,
                                         trajs_by_n[n_agents], risk, cfg, calib_seeds, logger)
        # 验证集只用于选择正则强度（不接触测试任务）
        Xv, yv, _ = _collect_calib_runs(n_agents, splits[n_agents].val,
                                        trajs_by_n[n_agents], risk, cfg,
                                        [calib_seeds[-1] + 1], logger)
        vm = np.isfinite(yv)
        best = None
        for alpha in (1e-4, 1e-3, 1e-2):
            cand = DurationModel(alpha=alpha).fit(X, y)
            pred_v = cand.predict(Xv)
            rmse_v = float(np.sqrt(np.mean((pred_v[vm] - yv[vm]) ** 2))) if vm.any() else float("nan")
            if best is None or (np.isfinite(rmse_v) and rmse_v < best[1]):
                best = (cand, rmse_v, alpha)
        model, val_rmse, alpha = best
        models[n_agents] = model
        model.save(run_dir / f"duration_model_n{n_agents}.json")
        calib_meta[str(n_agents)] = {
            **{k: v for k, v in meta.items() if k != "plan_cache"},
            "alpha_selected": alpha,
            "val_rmse_slots": val_rmse,
            "r2": model.r2, "train_rmse_slots": model.train_rmse_slots,
            "n_train_segments": model.n_train_segments,
            "residual_pool": len(model.residual_pool),
        }
        logger.info("[N%d] 时长模型 alpha=%.0e R2=%.3f train RMSE=%.3f val RMSE=%.3f 覆盖=%.2f",
                    n_agents, alpha, model.r2, model.train_rmse_slots,
                    val_rmse, meta["coverage"])

    # ---------------- 2) 冻结测试预测（不读测试执行真值） ----------------
    predictions = {}
    geom_by_task = {}
    pred_start = time.perf_counter()
    for n_agents in (2, 4):
        budget = WaitBudget(time_steps=T, horizon=H)
        mask = make_mask(H, allowed_positions(H, 5), n_agents)
        pause01 = torch.full((n_agents,), 0.1, dtype=torch.float64)
        for ti, jt in enumerate(splits[n_agents].test):
            paths_np = np.asarray(trajs_by_n[n_agents][jt["task_id"]], dtype=np.float64)
            paths_t = torch.tensor(paths_np, dtype=torch.float64)
            geom_by_task[jt["task_id"]] = _path_pair_min_distance(paths_np)
            w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
            r0 = policy_w0(w0, mask, budget, risk, paths_t, pause01, T)
            r2 = policy_w2plus(w0, mask, budget, risk, paths_t, pause01, T,
                               lam_T=0.05, max_evals=250, restarts=2, seed=ti)
            plans = {"orig": r0.wait.numpy(), "w2plus": r2.wait.numpy()}
            for plan_name, w_np in plans.items():
                for p in ps:
                    pause = torch.full((n_agents,), p, dtype=torch.float64)
                    rng = np.random.default_rng(77000 + ti * 37 + int(p * 100))
                    out = predict_m0m1m2(
                        risk, paths_t, torch.tensor(w_np, dtype=torch.float64),
                        pause, T, models[n_agents], n_draws=args.n_draws,
                        rng=rng, dt_slot=0.4)
                    pairs = pair_index(n_agents)
                    predictions[f"{jt['task_id']}|{plan_name}|{p}"] = {
                        "group": f"N{n_agents}", "task_id": jt["task_id"],
                        "map_id": jt.get("map_id", ""), "plan": plan_name, "p": p,
                        "pairs": [[int(i), int(j)] for i, j in pairs],
                        "m0": out["m0"].tolist(), "m1": out["m1"].tolist(),
                        "m2": out["m2"].tolist(),
                        "durations": np.asarray(out["durations"]).tolist(),
                        "wait": np.asarray(w_np).tolist(),
                    }
            logger.info("[N%d] 冻结预测 %s", n_agents, jt["task_id"])
    pred_elapsed = time.perf_counter() - pred_start
    pred_blob = json.dumps(predictions, ensure_ascii=False, sort_keys=True)
    pred_hash = hashlib.sha256(pred_blob.encode()).hexdigest()
    (run_dir / "predictions_m0m1m2.json").write_text(
        json.dumps({"predictions": predictions, "sha256": pred_hash,
                    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "elapsed_s": pred_elapsed,
                    "note": "生成于任何测试任务 MPC 真值之前；只使用静态规划输入"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("预测冻结 SHA256=%s（%.1fs）", pred_hash[:16], pred_elapsed)

    # ---------------- 3) 冻结之后才跑测试真值 ----------------
    truth = {}
    t_truth = time.perf_counter()
    for n_agents in (2, 4):
        for ti, jt in enumerate(splits[n_agents].test):
            paths_np = np.asarray(trajs_by_n[n_agents][jt["task_id"]], dtype=np.float64)
            for plan_name in ("orig", "w2plus"):
                # 复用预测文件里冻结的等待计划，保证预测/真值同计划
                pred0 = predictions[f"{jt['task_id']}|{plan_name}|{ps[0]}"]
                w_np = np.asarray(pred0["wait"], dtype=np.float64)
                for p in ps:
                    flags = []
                    for k in range(args.n_mc):
                        seed = args.truth_seed0 + ti * 1000 + k
                        r = simulate_mpc(
                            paths_np, w_np, p,
                            MPCSimConfig(d_safe=d_safe, horizon_steps=T,
                                         pause_fixed_s=0.4, pause_mode="fixed"),
                            seed=int(seed), record=False)
                        flags.append(np.asarray(r.pair_collision, dtype=bool))
                    truth[f"{jt['task_id']}|{plan_name}|{p}"] = np.asarray(flags)
            logger.info("[N%d] 真值 %s 完成", n_agents, jt["task_id"])
    truth_elapsed = time.perf_counter() - t_truth

    # ---------------- 4) 评分 ----------------
    rows = []
    for key, pred in predictions.items():
        flags = truth[key]
        obs = flags.mean(axis=0)
        geom = geom_by_task[pred["task_id"]]
        for idx, (i, j) in enumerate(pred["pairs"]):
            rows.append({
                "group": pred["group"], "task_id": pred["task_id"],
                "map_id": pred["map_id"], "plan": pred["plan"], "p": pred["p"],
                "pair": f"{i}-{j}", "m0": float(pred["m0"][idx]),
                "m1": float(pred["m1"][idx]), "m2": float(pred["m2"][idx]),
                "obs": float(obs[idx]), "n_mc": args.n_mc,
                "geometry_min_distance_m": float(geom[i, j]),
                "crossing": bool(geom[i, j] < 0.5),
            })
    with (run_dir / "m1_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def sel_rows(**filters):
        out = rows
        for k, v in filters.items():
            out = [r for r in out if (abs(r[k] - v) < 1e-12 if isinstance(v, float) else r[k] == v)]
        return out

    def stats_for(sel):
        return {m: pairwise_stats([r[m] for r in sel], [r["obs"] for r in sel])
                for m in ("m0", "m1", "m2")}

    def task_mae_diff(sel, m):
        by_task = {}
        for tid in sorted({r["task_id"] for r in sel}):
            s = [r for r in sel if r["task_id"] == tid]
            mae_m = np.mean([abs(r[m] - r["obs"]) for r in s])
            mae_0 = np.mean([abs(r["m0"] - r["obs"]) for r in s])
            by_task[tid] = float(mae_m - mae_0)
        return bootstrap_task_ci(by_task)

    summary = {
        "stage": "phase2_m0m1m2_no_leakage",
        "n_mc": args.n_mc, "n_draws": args.n_draws, "ps": ps,
        "split": split_manifest,
        "calibration": calib_meta,
        "prediction_sha256": pred_hash,
        "truth_seeds": {"seed0": args.truth_seed0, "n_mc": args.n_mc,
                        "distinct_from_calib": True},
        "overall": stats_for(rows),
        "delta_mae_vs_m0": {
            "m1": task_mae_diff(rows, "m1"),
            "m2": task_mae_diff(rows, "m2"),
        },
        "by_group": {}, "by_p": {}, "by_plan": {},
    }
    for g in ("N2", "N4"):
        summary["by_group"][g] = stats_for(sel_rows(group=g))
    for p in ps:
        summary["by_p"][str(p)] = stats_for(sel_rows(p=p))
    for plan in ("orig", "w2plus"):
        summary["by_plan"][plan] = stats_for(sel_rows(plan=plan))
    for crossing_flag in (True, False):
        sel = [r for r in rows if r["crossing"] == crossing_flag]
        summary.setdefault("by_geometry", {})["crossing" if crossing_flag else "non_crossing"] = \
            stats_for(sel) if sel else {"n": 0}
    summary["reliability"] = {m: reliability_table([r[m] for r in rows],
                                                   [r["obs"] for r in rows])
                              for m in ("m0", "m1", "m2")}
    summary["runtime"] = {"prediction_s": pred_elapsed, "truth_mc_s": truth_elapsed,
                          "prediction_per_task_s": pred_elapsed / max(len(predictions), 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps({
        "command": " ".join(sys.argv),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ratios_requested": ratios,
        "total_s": pred_elapsed + truth_elapsed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图
    obs = np.array([r["obs"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, m, name in zip(axes, ("m0", "m1", "m2"),
                           ("M0 原始 First-Hit", "M1 均值时长修正", "M2 时长分布抽样")):
        pred = np.array([r[m] for r in rows])
        ax.scatter(pred, obs, s=18, alpha=0.75)
        lim = [0, max(0.6, float(max(pred.max(), obs.max())) * 1.05)]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(name)
        ax.set_ylabel("MPC MC 两两碰撞频率")
        st = pairwise_stats(pred, obs)
        ax.set_title(f"MAE={st['mae']:.3f}  ρ={st['spearman']:.3f}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "m0m1m2_calibration.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "n"}
                      for k, v in summary["overall"].items()}, ensure_ascii=False, indent=1))
    print(f"输出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
