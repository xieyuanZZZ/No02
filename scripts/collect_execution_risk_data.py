"""阶段二 §2.2：从阶段一已验证的 `simulate_mpc()` 采集 M3 校准数据。

只采集 **calib + val** 任务；测试任务在执行本脚本时完全不运行，避免
测试轨迹进入标定。每个样本包含：
  * 场景身份：map_id/task_id/plan/p/seed/split；
  * 静态特征可从 plan bank 直接重建（collector 保存 plan/wait 元数据）；
  * 训练可观测状态：每个控制步 s、v、mode、v_plan（不含测试标签）；
  * 评测标签：pair 碰撞/首次时刻/最小间距/完成/超时（仅保存，拟合不用）。

输出（目录可配）：
  manifest.json      任务/计划/p/种子/切分/覆盖率
  traces_n2.npz      固定 Tmax 的数组 + 每 run 有效长度
  traces_n4.npz
  events.jsonl       每 run 的每台机器人事件账本（只保留关键事件）
  run_index.json     每 run 的元数据与标签
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

from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams  # noqa: E402
from fhd.evaluation.risk_dataset import build_splits, load_bank, save_split  # noqa: E402
from fhd.execution.mpc_simulator import MPCSimConfig, simulate_mpc  # noqa: E402
from fhd.geometry.static_maps import load_obstacles_for_map  # noqa: E402
from fhd.optim.wait_policies import (  # noqa: E402
    WaitBudget, allowed_positions, make_mask, policy_w0, policy_w2plus,
)
from fhd.risk.multi_pair import FirstHitRisk  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

KEY_EVENTS = {"hold_start", "resume", "brake_start", "brake_stop", "pause_accepted",
              "pause_rejected", "infeasible_stop", "creep_start", "waypoint_arrival"}


def _plans_for_task(jt, n_agents, risk, cfg, ti):
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    budget = WaitBudget(time_steps=T, horizon=H)
    mask = make_mask(H, allowed_positions(H, 5), n_agents)
    paths = torch.tensor(np.asarray(jt["paths_m"]), dtype=torch.float64)
    w0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
    pause01 = torch.full((n_agents,), 0.1, dtype=torch.float64)
    r0 = policy_w0(w0, mask, budget, risk, paths, pause01, T)
    r2 = policy_w2plus(w0, mask, budget, risk, paths, pause01, T,
                       lam_T=0.05, max_evals=250, restarts=2, seed=ti)
    return {"orig": r0.wait.numpy(), "w2plus": r2.wait.numpy()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/Online_FirstHit_v2/04_dynamic_risk_m3/data")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--seeds", default="92001,92002,92003")
    ap.add_argument("--ps", default="0,0.05,0.1,0.2")
    ap.add_argument("--ratios", default="0.5,0.2,0.3")
    ap.add_argument("--max-runs", type=int, default=0, help="调试用；0=不限")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "base.yaml")
    set_seed(int(cfg["seed"]))
    T = int(cfg["execution"]["horizon_steps"])
    H = int(cfg["trajectory"]["horizon"])
    dt_ctrl = 0.1
    Tmax = int(np.ceil(T * 0.4 / dt_ctrl))
    risk = FirstHitRisk.from_config(cfg)
    d_safe = risk.d_safe
    seeds = [int(x) for x in args.seeds.split(",")]
    ps = [float(x) for x in args.ps.split(",")]
    ratios = tuple(float(x) for x in args.ratios.split(","))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.collect_risk", out / "collect.log")
    log_section(logger, "M3 校准数据采集（仅 calib/val）")

    tasks_by_n, trajs_by_n = {}, {}
    for n in (2, 4):
        tasks, trajs = load_bank(n, ROOT)
        tasks_by_n[n] = tasks
        trajs_by_n[n] = trajs
    splits = build_splits(tasks_by_n, ROOT, ratios=ratios, seed=20261105)
    split_manifest = {str(n): b.manifest for n, b in splits.items()}
    if args.split_file and Path(args.split_file).exists():
        split_manifest = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    save_split(split_manifest, out / "split.json")

    run_index = []
    events_fh = (out / "events.jsonl").open("w", encoding="utf-8")
    t_start = time.perf_counter()
    for n in (2, 4):
        tasks = splits[n].calib + splits[n].val
        # 依据 split manifest 标注 calib/val
        calib_ids = set(split_manifest[str(n)]["calib"])
        arrays = {k: [] for k in ("s", "v", "v_plan", "modes", "mask")}
        t_valid = []
        labels = []
        for ti, jt in enumerate(tasks):
            split = "calib" if jt["task_id"] in calib_ids else "val"
            paths = np.asarray(trajs_by_n[n][jt["task_id"]], dtype=np.float64)
            plans = _plans_for_task(jt, n, risk, cfg, ti)
            obs, obs_meta = load_obstacles_for_map(str(jt.get("map_id", "")))
            for plan_name, w in plans.items():
                tps = []
                for i in range(n):
                    w_full = np.zeros(H)
                    w_full[:H - 1] = w[i]
                    tps.append(TimeParameterizedPath(paths[i], w_full, TimingParams()))
                for p in ps:
                    for seed in seeds:
                        if args.max_runs and len(run_index) >= args.max_runs:
                            break
                        r = simulate_mpc(
                            paths, w, p,
                            MPCSimConfig(d_safe=d_safe, horizon_steps=T),
                            seed=int(seed), record=True, obstacles=obs,
                            obstacle_meta=obs_meta)
                        tv = int(r.s_trace.shape[0])
                        vp = np.zeros_like(r.s_trace)
                        for i in range(n):
                            vp[:, i] = np.interp(r.s_trace[:, i], tps[i].s_grid,
                                                 tps[i].v_grid)
                        for key, arr in (("s", r.s_trace), ("v", r.v_trace),
                                         ("v_plan", vp), ("modes", r.modes)):
                            padded = np.zeros((Tmax, n), dtype=np.float64)
                            padded[:tv] = arr
                            arrays[key].append(padded.astype(np.float32))
                        m = np.zeros((Tmax, n), dtype=bool)
                        m[:tv] = True
                        arrays["mask"].append(m)
                        t_valid.append(tv)
                        labels.append(np.asarray(r.pair_collision, dtype=bool))
                        run_index.append({
                            "run_id": len(run_index), "n_agents": n, "split": split,
                            "task_id": jt["task_id"], "map_id": jt.get("map_id", ""),
                            "plan": plan_name, "p": p, "seed": int(seed),
                            "T_valid": tv, "completion_time": float(r.completion_time),
                            "all_arrived": bool(r.all_arrived), "timeout": bool(r.timeout),
                            "collision": bool(r.collision), "infeasible_stops": int(r.infeasible_stops),
                            "mpc_failures": int(r.mpc_failures),
                            "min_pair_distance": float(np.nanmin(r.pair_min_distance)),
                            "static_status": r.static_obstacle_status,
                        })
                        events_fh.write(json.dumps({
                            "run_id": len(run_index) - 1,
                            "robot_events": [
                                [{k: v for k, v in e.items() if k in
                                  ("t", "type", "kind", "wp", "s", "duration", "held",
                                   "brake_time", "planned_hold", "shortfall", "reason")}
                                 for e in ev if e.get("type") in KEY_EVENTS]
                                for ev in r.robot_events],
                        }, ensure_ascii=False) + "\n")
            logger.info("[N%d] %s 校准数据完成", n, jt["task_id"])
            if args.max_runs and len(run_index) >= args.max_runs:
                break
        if not arrays["s"]:
            logger.info("[N%d] 没有采集到 run（可能被 --max-runs 截断），跳过", n)
            continue
        np.savez_compressed(
            out / f"traces_n{n}.npz",
            s=np.stack(arrays["s"]), v=np.stack(arrays["v"]),
            v_plan=np.stack(arrays["v_plan"]), modes=np.stack(arrays["modes"]),
            mask=np.stack(arrays["mask"]), T_valid=np.asarray(t_valid, dtype=np.int32),
            labels=np.stack(labels) if labels else np.zeros((0, 1), dtype=bool),
        )
        logger.info("[N%d] traces 保存：%d runs", n, len(t_valid))

    events_fh.close()
    (out / "run_index.json").write_text(json.dumps(run_index, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    manifest = {
        "stage": "phase2_m3_calibration_data",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": time.perf_counter() - t_start,
        "split": split_manifest,
        "seeds": seeds, "ps": ps, "Tmax": Tmax, "dt_ctrl": dt_ctrl,
        "n_runs": len(run_index),
        "note": "只运行 calib/val；测试任务不在此脚本中执行",
        "collision_label_usage": "仅保存用于诊断；M3 参数拟合不使用碰撞标签",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"输出：{out}  runs={len(run_index)}  耗时 {manifest['elapsed_s']/60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
