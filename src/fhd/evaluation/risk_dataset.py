"""风险数据集切分、几何去重与评分工具（阶段二 §2.2 / §2.4）。

切分原则：优先按 `map_id` 分离校准/验证/测试，避免同一地图的近似路径跨 split。
数据量很小（N2 只有 6 张地图），因此允许 50/20/30 的实际比例，并在报告中
标注为"探索性小样本"，绝不宣称跨场景泛化。切分清单与哈希必须落盘。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _hash_paths(paths) -> str:
    arr = np.round(np.asarray(paths, dtype=np.float64), 6)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def split_by_map(tasks: list, ratios=(0.5, 0.2, 0.3), seed: int = 0) -> dict:
    """按 map_id 切分任务；返回含 split 清单、地图清单与哈希的字典。"""
    maps = sorted({str(t["map_id"]) for t in tasks})
    rng = np.random.default_rng(int(seed))
    order = list(maps)
    rng.shuffle(order)
    n = len(order)
    n_calib = max(1, int(round(ratios[0] * n)))
    n_val = max(1, int(round(ratios[1] * n)))
    if n_calib + n_val >= n:
        n_calib = max(1, n - 2)
        n_val = 1
    calib_maps = sorted(order[:n_calib])
    val_maps = sorted(order[n_calib:n_calib + n_val])
    test_maps = sorted(order[n_calib + n_val:])
    out = {"calib": [], "val": [], "test": [], "maps": {},
           "ratios_requested": list(ratios), "seed": int(seed)}
    for t in tasks:
        mid = str(t["map_id"])
        key = "calib" if mid in calib_maps else ("val" if mid in val_maps else "test")
        out[key].append(t["task_id"])
    out["maps"] = {"calib": calib_maps, "val": val_maps, "test": test_maps}
    out["counts"] = {k: len(out[k]) for k in ("calib", "val", "test")}
    out["ratios_actual"] = {k: out["counts"][k] / max(len(tasks), 1)
                            for k in ("calib", "val", "test")}
    # 几何重复检查：同一路径不能跨 split
    by_hash = {}
    for t in tasks:
        h = _hash_paths(t["paths_m"])
        by_hash.setdefault(h, []).append(t)
    dup_across = []
    for h, group in by_hash.items():
        split_of = set()
        for t in group:
            mid = str(t["map_id"])
            split_of.add("calib" if mid in calib_maps else
                         ("val" if mid in val_maps else "test"))
        if len(split_of) > 1:
            dup_across.append({"hash": h, "task_ids": [g["task_id"] for g in group],
                               "splits": sorted(split_of)})
    out["duplicate_geometry_across_splits"] = dup_across
    out["split_hash"] = hashlib.sha256(
        json.dumps({k: out[k] for k in ("calib", "val", "test")}, sort_keys=True)
        .encode()).hexdigest()[:16]
    return out


def save_split(split: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")


def _segment_segment_distance(a, b, c, d) -> float:
    def seg_point(p, q, x):
        pq = q - p
        L2 = float(pq @ pq)
        if L2 < 1e-12:
            return float(np.linalg.norm(x - p))
        t = float(np.clip((x - p) @ pq / L2, 0.0, 1.0))
        return float(np.linalg.norm(p + t * pq - x))

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


def path_pair_min_distance(paths: np.ndarray) -> np.ndarray:
    """两两折线之间的最小距离（米），用于几何交叉分组。"""
    n = paths.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a0, b0 = paths[i, :-1], paths[i, 1:]
            c0, d0 = paths[j, :-1], paths[j, 1:]
            best = np.inf
            for k in range(a0.shape[0]):
                for m in range(c0.shape[0]):
                    best = min(best, _segment_segment_distance(a0[k], b0[k],
                                                               c0[m], d0[m]))
            out[i, j] = out[j, i] = best
    return out


# --------------------------------------------------------------------------- #
# 评分
# --------------------------------------------------------------------------- #
def pairwise_stats(pred, obs) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    m = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[m], obs[m]
    err = pred - obs
    out = {
        "n": int(pred.size),
        "mae": float(np.mean(np.abs(err))) if pred.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(err ** 2))) if pred.size else float("nan"),
        "bias": float(np.mean(err)) if pred.size else float("nan"),
        "brier": float(np.mean(err ** 2)) if pred.size else float("nan"),
        "spearman": spearman(pred, obs),
    }
    return out


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def bootstrap_task_ci(diffs_by_task: dict, n_boot: int = 4000,
                      seed: int = 0) -> dict:
    """以**任务**为抽样单位的分组 bootstrap（不是把 6 个 pair 当 6 个任务）。"""
    keys = sorted(diffs_by_task)
    vals = np.array([diffs_by_task[k] for k in keys], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n_tasks": 0, "mean": None, "ci_lo": None, "ci_hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return {
        "n_tasks": int(vals.size),
        "mean": float(vals.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
    }


def reliability_table(pred, obs, bins=(0.0, 0.05, 0.1, 0.2, 0.4, 1.01)) -> list:
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum() == 0:
            continue
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "mean_pred": float(pred[m].mean()),
                     "empirical_rate": float(obs[m].mean())})
    return rows


def load_bank(n_agents: int, root: str | Path) -> tuple[list, dict]:
    root = Path(root)
    bank_json = "plan_bank.json" if n_agents == 2 else "plan_bank_n4.json"
    trajs_npz = "plan_bank_trajs.npz" if n_agents == 2 else "plan_bank_n4_trajs.npz"
    bank = json.loads((root / "data" / "mpd_delay_plan_bank" / bank_json)
                      .read_text(encoding="utf-8"))
    trajs = np.load(root / "data" / "mpd_delay_plan_bank" / trajs_npz)
    return bank["tasks"], {k: trajs[k] for k in trajs.files}


@dataclass
class SplitBundle:
    calib: list
    val: list
    test: list
    manifest: dict


def build_splits(tasks_by_n: dict[int, list], root: str | Path,
                 ratios=(0.5, 0.2, 0.3), seed: int = 0) -> dict[int, SplitBundle]:
    out = {}
    for n, tasks in tasks_by_n.items():
        manifest = split_by_map(tasks, ratios=ratios, seed=seed + n)
        by_id = {t["task_id"]: t for t in tasks}
        out[n] = SplitBundle(
            calib=[by_id[t] for t in manifest["calib"]],
            val=[by_id[t] for t in manifest["val"]],
            test=[by_id[t] for t in manifest["test"]],
            manifest=manifest,
        )
    return out
