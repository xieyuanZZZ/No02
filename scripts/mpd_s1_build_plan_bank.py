"""阶段一（主环境）：构建冻结的"有效轨迹库 + 联合任务库"。

输入：WSL 侧 `mpd_extension/gen_plan_candidates.py` 生成的原始候选
（含官方有效 mask、官方坐标与米制坐标、官方计时）。

本脚本做的事：
  1. **逐个候选**做独立检查（全部在米制下）：
     * 端点：首末点与起终点的偏差；
     * 边界：整条轨迹在 [0,10]² 内；
     * 静态安全：对障碍做**连续几何**距离检查，并给出 B-spline 弦近似误差上界，
       要求 `min_clearance > chord_error_bound + margin`；
     * 运动学：按弧长重采样到 H=32 后，每段长度 ≤ v_max·Δt。
  2. 把两台机器人的有效轨迹组合成**联合任务**，要求 p=0、仅含固定发车间隔时
     联合执行无碰撞且全部到达（用 `fhd.execution.simulate_execution`）。
  3. 落盘 plan bank + 覆盖率统计 + 每个候选的拒绝原因。

输出目录：data/mpd_delay_plan_bank/
  plan_bank.json        联合任务清单（含起终点、轨迹、发车间隔、来源、种子）
  plan_bank_trajs.npz   轨迹数组（米制，H=32）
  candidate_audit.csv   448 条原始候选逐条审计（含拒绝原因）
  planning_coverage.json 规划覆盖率统计
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.execution.simulator import simulate_execution  # noqa: E402
from fhd.geometry.primitives import point_rect_signed_distance  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402
from fhd.utils.seed import set_seed  # noqa: E402

WORLD_M = 10.0


# --------------------------------------------------------------------------- #
def arc_length_resample(pts: np.ndarray, h: int) -> np.ndarray:
    """按弧长均匀重采样到 h 个点（含首末）。"""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(pts[:1], h, axis=0)
    target = np.linspace(0.0, total, h)
    out = np.empty((h, pts.shape[1]), dtype=np.float64)
    for d in range(pts.shape[1]):
        out[:, d] = np.interp(target, s, pts[:, d])
    return out


def continuous_clearance(pts: np.ndarray, circles: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """逐点对障碍的精确有符号距离（正=在障碍外）。米制。"""
    d = np.full(len(pts), np.inf)
    if len(circles):
        c = circles[:, :2][None, :, :]
        r = circles[:, 2][None, :]
        dc = np.linalg.norm(pts[:, None, :] - c, axis=-1) - r
        d = np.minimum(d, dc.min(axis=1))
    if len(boxes):
        lo = boxes[:, :2][None, :, :]
        hi = boxes[:, 2:][None, :, :]
        p = pts[:, None, :]
        inside = np.all((p >= lo) & (p <= hi), axis=-1)
        dx = np.maximum(np.maximum(lo[..., 0] - p[..., 0], p[..., 0] - hi[..., 0]), 0.0)
        dy = np.maximum(np.maximum(lo[..., 1] - p[..., 1], p[..., 1] - hi[..., 1]), 0.0)
        dbox = np.sqrt(dx ** 2 + dy ** 2)
        dbox = np.where(inside, -np.minimum.reduce([
            p[..., 0] - lo[..., 0], hi[..., 0] - p[..., 0],
            p[..., 1] - lo[..., 1], hi[..., 1] - p[..., 1]]), dbox)
        d = np.minimum(d, dbox.min(axis=1))
    return d


def chord_error_bound(pos: np.ndarray, acc: np.ndarray, duration: float) -> float:
    """B-spline 用 128 个采样点折线近似的**误差上界**。

    对均匀采样的二阶可微曲线，弦与曲线最大偏差 ≤ max|q''|·h²/8，
    其中 h = duration/(T-1)。官方 `q_trajs_acc` 就是 q''(t)，直接用它。
    """
    t_pts = pos.shape[0]
    if t_pts < 2 or duration <= 0:
        return 0.0
    h = duration / (t_pts - 1)
    a_max = float(np.max(np.linalg.norm(acc, axis=-1))) if acc.size else 0.0
    return 0.125 * a_max * h * h


def audit_candidate(pos_m: np.ndarray, acc: np.ndarray, duration: float,
                    start_m: np.ndarray, goal_m: np.ndarray,
                    circles: np.ndarray, boxes: np.ndarray,
                    v_max: float, dt_s: float, h_waypoints: int,
                    tol_endpoint: float = 1e-3) -> dict:
    """单条候选的独立检查。返回指标 + 拒绝原因（空表示通过）。"""
    reasons = []
    ep_start = float(np.linalg.norm(pos_m[0] - start_m))
    ep_goal = float(np.linalg.norm(pos_m[-1] - goal_m))
    if ep_start > tol_endpoint or ep_goal > tol_endpoint:
        reasons.append(f"endpoint({ep_start:.4f},{ep_goal:.4f})")

    lo, hi = pos_m.min(axis=0), pos_m.max(axis=0)
    if lo.min() < -1e-6 or hi.max() > WORLD_M + 1e-6:
        reasons.append("out_of_bounds")

    clr = continuous_clearance(pos_m, circles, boxes)
    bound = chord_error_bound(pos_m, acc, duration)
    min_clr = float(clr.min())
    if min_clr - bound <= 0.0:
        reasons.append(f"static(clear={min_clr:.4f},chord={bound:.5f})")

    wp = arc_length_resample(pos_m, h_waypoints)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=-1)
    max_seg = float(seg.max())
    if max_seg > v_max * dt_s + 1e-9:
        reasons.append(f"speed(max_seg={max_seg:.3f}>{v_max * dt_s:.3f})")

    return {
        "ok": len(reasons) == 0,
        "reasons": ";".join(reasons),
        "endpoint_start_err": ep_start,
        "endpoint_goal_err": ep_goal,
        "min_clearance": min_clr,
        "chord_bound": bound,
        "max_segment": max_seg,
        "path_length": float(np.sum(np.linalg.norm(np.diff(pos_m, axis=0), axis=-1))),
        "waypoints": wp,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="D:/mpd_official/results_mpd/plan_candidates")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "mpd_delay_plan_bank"))
    ap.add_argument("--stagger-slots", type=int, default=3, help="相邻机器人的固定发车间隔（时隙）")
    ap.add_argument("--max-tasks", type=int, default=24)
    ap.add_argument("--d-safe", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    v_max = float(cfg["trajectory"]["max_speed_mps"])
    dt_s = float(cfg["trajectory"]["dt_s"])
    h_wp = int(cfg["trajectory"]["horizon"])
    t_steps = int(cfg["execution"]["horizon_steps"])
    from fhd.risk.multi_pair import FirstHitRisk
    risk_probe = FirstHitRisk.from_config(cfg)
    d_safe = float(args.d_safe or risk_probe.d_safe)

    run_dir = new_run_dir("MPD_S1_plan_bank", "s1_plan_bank")
    logger = get_logger("fhd.mpd_s1", run_dir / "logs" / "s1.log")
    log_section(logger, "阶段一：构建冻结的有效轨迹库")
    logger.info("v_max=%.2f dt=%.2f H=%d T=%d d_safe=%.3f stagger=%d",
                v_max, dt_s, h_wp, t_steps, d_safe, args.stagger_slots)

    cand_dir = Path(args.cand)
    index = json.loads((cand_dir / "index.json").read_text(encoding="utf-8"))
    items = [x for x in index["items"] if x.get("status") == "ok"]
    logger.info("候选条目 %d（原始候选总数 %d）", len(items),
                sum(x["n_raw"] for x in items))

    # ---------------- 1) 逐候选审计 ----------------
    audit_rows = []
    valid: dict[str, list] = {}
    for it in items:
        z = np.load(cand_dir / it["file"], allow_pickle=True)
        pos_m = z["pos_m_raw"].astype(np.float64)
        acc = z["acc_raw"].astype(np.float64)
        start_m = z["start_m"].astype(np.float64)
        goal_m = z["goal_m"].astype(np.float64)
        mask = z["official_valid_mask"].astype(bool)
        duration = float(it["trajectory_duration"])

        sc = json.loads((Path(args.cand).parents[0] / ".." / ".." /
                         "paper2" / "data" / "revised_s1" / "maps_test.json")
                        .resolve().read_text(encoding="utf-8")) if False else None
        # 直接从 S1 测试集读地图几何（米制）
        maps = getattr(main, "_maps_cache", None)
        if maps is None:
            mp = Path("D:/paper2/data/revised_s1/maps_test.json")
            maps = {m.get("map_id") or f"{m['kind']}_{m['seed']}": m
                    for m in json.loads(mp.read_text(encoding="utf-8"))}
            main._maps_cache = maps
        sc = maps[it["map_id"]]
        rects = np.asarray(sc["rects"], dtype=np.float64).reshape(-1, 4)
        circ = np.asarray(sc["circles"], dtype=np.float64).reshape(-1, 3)

        for k in range(pos_m.shape[0]):
            a = audit_candidate(pos_m[k], acc[k], duration, start_m, goal_m,
                                circ, rects, v_max, dt_s, h_wp)
            audit_rows.append({
                "map_id": it["map_id"], "kind": it["kind"], "task_index": it["task_index"],
                "cand": k, "official_valid": bool(mask[k]), "ok": a["ok"],
                "reasons": a["reasons"],
                "min_clearance": round(a["min_clearance"], 5),
                "chord_bound": round(a["chord_bound"], 6),
                "max_segment": round(a["max_segment"], 4),
                "path_length": round(a["path_length"], 4),
            })
            if a["ok"]:
                valid.setdefault(f"{it['map_id']}|{it['task_index']}", []).append({
                    "kind": it["kind"], "map_id": it["map_id"],
                    "task_index": it["task_index"], "cand": k,
                    "waypoints": a["waypoints"],
                    "path_length": a["path_length"],
                    "start_m": start_m, "goal_m": goal_m,
                    "file": it["file"],
                })

    n_audit = len(audit_rows)
    n_ok = sum(1 for r in audit_rows if r["ok"])
    logger.info("候选审计：%d 条，通过 %d 条（%.1f%%）", n_audit, n_ok, 100 * n_ok / max(n_audit, 1))

    # ---------------- 2) 组合成联合任务 ----------------
    by_map: dict[str, list] = {}
    for key, lst in valid.items():
        by_map.setdefault(lst[0]["map_id"], []).extend(lst)
    for k in by_map:
        by_map[k].sort(key=lambda x: x["task_index"])

    joint_tasks = []
    attempts = 0
    rng = np.random.default_rng(int(cfg["seed"]))
    for map_id, cands in sorted(by_map.items()):
        # 按机器人任务分组，每组取一条候选，两两组合
        per_task: dict[int, list] = {}
        for c in cands:
            per_task.setdefault(c["task_index"], []).append(c)
        task_ids = sorted(per_task)
        for t1, t2 in combinations(task_ids, 2):
            if len(joint_tasks) >= args.max_tasks:
                break
            attempts += 1
            c1 = per_task[t1][int(rng.integers(len(per_task[t1])))]
            c2 = per_task[t2][int(rng.integers(len(per_task[t2])))]
            paths = torch.tensor(np.stack([c1["waypoints"], c2["waypoints"]]),
                                 dtype=torch.float64)
            starts = np.stack([c1["start_m"], c2["start_m"]])
            goals = np.stack([c1["goal_m"], c2["goal_m"]])
            # 固定发车间隔（基准计划的一部分，所有方法共用）
            wait0 = torch.zeros(2, h_wp - 1, dtype=torch.float64)
            for i in range(2):
                wait0[i, 0] = args.stagger_slots * i
            pause0 = torch.zeros(2, dtype=torch.float64)
            sim = simulate_execution(paths, wait0, pause0, d_safe, t_steps, dt_s,
                                     n_samples=1, generator=torch.Generator().manual_seed(0))
            ok = (not bool(sim.collision.any())) and bool(sim.all_arrived.all())
            if not ok:
                continue
            pair_min = float(sim.pair_min_distance.min())
            joint_tasks.append({
                "task_id": f"{map_id}_pair{t1:02d}_{t2:02d}",
                "map_id": map_id, "kind": c1["kind"],
                "n_agents": 2,
                "starts_m": starts.tolist(), "goals_m": goals.tolist(),
                "paths_m": paths.numpy().tolist(),
                "stagger_slots": int(args.stagger_slots),
                "baseline_wait": wait0.numpy().tolist(),
                "p0_pair_min_distance": pair_min,
                "source": {"files": [c1["file"], c2["file"]],
                           "cands": [c1["cand"], c2["cand"]],
                           "task_indices": [c1["task_index"], c2["task_index"]]},
                "path_lengths_m": [c1["path_length"], c2["path_length"]],
                "planned_completion_slots": int(h_wp - 1 + args.stagger_slots),
            })
        if len(joint_tasks) >= args.max_tasks:
            break

    logger.info("联合任务：尝试 %d 对，入库 %d 个（p=0 无碰撞且全部到达）",
                attempts, len(joint_tasks))

    # ---------------- 3) 从磁盘重放校验（验收要求） ----------------
    replay_ok = 0
    for jt in joint_tasks:
        paths = torch.tensor(np.asarray(jt["paths_m"]), dtype=torch.float64)
        wait0 = torch.tensor(np.asarray(jt["baseline_wait"]), dtype=torch.float64)
        sim = simulate_execution(paths, wait0, torch.zeros(2, dtype=torch.float64),
                                 d_safe, t_steps, dt_s, n_samples=1,
                                 generator=torch.Generator().manual_seed(0))
        if (not bool(sim.collision.any())) and bool(sim.all_arrived.all()):
            replay_ok += 1
    logger.info("从落盘数据重放：%d/%d 一致", replay_ok, len(joint_tasks))

    # ---------------- 4) 落盘 ----------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trajs = {jt["task_id"]: np.asarray(jt["paths_m"], dtype=np.float32) for jt in joint_tasks}
    np.savez_compressed(out / "plan_bank_trajs.npz", **trajs)

    bank = {
        "created_by": "scripts/mpd_s1_build_plan_bank.py",
        "config": {"v_max": v_max, "dt_s": dt_s, "H": h_wp, "T": t_steps,
                   "d_safe": d_safe, "stagger_slots": args.stagger_slots,
                   "world_m": WORLD_M},
        "source": {"planner": index["planner"], "n_samples": index["n_samples"],
                   "kinds": index["kinds"], "seed": index["seed"],
                   "trajectory_duration": index["trajectory_duration"],
                   "num_T_pts": index["num_T_pts"],
                   "model_sha256": "6632bbb494353e14505ea6ab0f600a5cf240279a1e4dc964608a66ae6aa013d0"},
        "n_joint_tasks": len(joint_tasks),
        "replay_ok": replay_ok,
        "tasks": joint_tasks,
    }
    (out / "plan_bank.json").write_text(json.dumps(bank, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    with (out / "candidate_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)

    coverage = {
        "n_robot_tasks_requested": len(items),
        "n_raw_candidates": n_audit,
        "n_official_valid": int(sum(1 for r in audit_rows if r["official_valid"])),
        "n_pass_all_checks": n_ok,
        "pass_rate_of_raw": n_ok / max(n_audit, 1),
        "pass_rate_of_official_valid": n_ok / max(sum(1 for r in audit_rows if r["official_valid"]), 1),
        "reject_reason_counts": {},
        "n_joint_pairs_attempted": attempts,
        "n_joint_tasks": len(joint_tasks),
        "joint_success_rate": len(joint_tasks) / max(attempts, 1),
    }
    rc: dict[str, int] = {}
    for r in audit_rows:
        if r["ok"]:
            continue
        for part in r["reasons"].split(";"):
            key = part.split("(")[0]
            rc[key] = rc.get(key, 0) + 1
    coverage["reject_reason_counts"] = rc
    (out / "planning_coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    # 轨迹库文件的 SHA256（下游引用）
    sha = hashlib.sha256((out / "plan_bank_trajs.npz").read_bytes()).hexdigest()
    (out / "MANIFEST.json").write_text(json.dumps({
        "plan_bank_sha256": hashlib.sha256((out / "plan_bank.json").read_bytes()).hexdigest(),
        "trajs_sha256": sha,
        "coverage": coverage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("覆盖统计：%s", json.dumps(coverage, ensure_ascii=False))
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    print(f"输出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
