"""统计工具：Wilson 区间、聚类 bootstrap 置信区间、配对比较（文档 §10.3）。"""

from __future__ import annotations

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """二项比例的点估计与 Wilson 区间。"""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    statistic: str = "mean",
    seed: int = 0,
) -> dict:
    """以“规划任务”为簇做 bootstrap（同一任务下的多次执行不独立）。

    values  : 每个观测（任务 × 执行复现）的数值
    clusters: 对应的任务 id
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if len(unique) == 0 or len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_clusters": 0}
    rng = np.random.default_rng(seed)
    idx_by_cluster = {c: np.nonzero(clusters == c)[0] for c in unique}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([idx_by_cluster[c] for c in chosen])
        vals = values[sample]
        boots[b] = np.mean(vals) if statistic == "mean" else np.median(vals)
    point = np.mean(values) if statistic == "mean" else np.median(values)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi),
            "n": int(len(values)), "n_clusters": int(len(unique))}


def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1,
) -> dict:
    """配对差 (A - B) 的聚类 bootstrap 置信区间。"""
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    clusters = np.asarray(clusters)
    diff = values_a - values_b
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    idx_by_cluster = {c: np.nonzero(clusters == c)[0] for c in unique}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([idx_by_cluster[c] for c in chosen])
        boots[b] = np.mean(diff[sample])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(np.mean(diff)), "lo": float(lo), "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0),
            "n_clusters": int(len(unique))}


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
