"""M3 对外接口：状态相关执行模型的首次碰撞风险预测。

严格区分：
  * 本模块的 `predict_pair_risks()` 是**采样式模型估计**，不是对真实 MPC
    风险的严格安全保证；
  * `team_collision_estimate` 来自**同一组联合样本**的任一 pair 首次事件，
    不是把两两风险相加；
  * 两两风险之和仍只作为并集上界，不能当全队概率。
"""

from __future__ import annotations

import numpy as np

from fhd.risk.conditional_execution_model import (
    ConditionalExecutionModel,
    ExecutionModelParams,
    compute_pair_first_hits,
)


def _as_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def predict_pair_risks(paths, wait, pause_p: float, horizon: float,
                       model: ConditionalExecutionModel | ExecutionModelParams,
                       d_safe: float = 0.35, n_samples: int = 200, seed: int = 0,
                       use_braking: bool = True, use_modes: bool = True,
                       use_speed: bool = True, return_traces: bool = False) -> dict:
    """预测每对机器人的首次碰撞概率。

    paths: [N,H,2] 几何路径（米）；wait: [N,H-1] 等待时隙；
    pause_p: 标量随机暂停概率；horizon: 观测时域（秒）。
    """
    paths = _as_np(paths)
    wait = _as_np(wait)
    if isinstance(model, ExecutionModelParams):
        model = ConditionalExecutionModel(model)
    out = model.predict(paths, wait, float(pause_p), float(horizon), float(d_safe),
                        n_samples=int(n_samples), seed=int(seed),
                        use_braking=use_braking, use_modes=use_modes,
                        use_speed=use_speed)
    result = {k: out[k] for k in
              ("pair_index", "pair_risks", "pair_min_distance_mean",
               "first_hit_time_distribution", "team_first_hit_times",
               "team_collision_estimate", "n_samples", "runtime_ms", "model_info")}
    # 预测文件不应为每个 key 重复保存上千个暂停样本；只保存参数摘要
    info = dict(result["model_info"])
    params = dict(info.get("params", {}))
    pauses = params.pop("pause_durations", [])
    params["pause_n"] = len(pauses)
    params["pause_mean_s"] = float(np.mean(pauses)) if pauses else 0.4
    info["params"] = params
    result["model_info"] = info
    if return_traces:
        result["trace_arrays"] = out["trace_arrays"]
    return result


def deterministic_first_hit(positions: np.ndarray, times: np.ndarray,
                            d_safe: float) -> dict:
    """对**确定性**轨迹序列做首次碰撞判定（用于解析测试与诊断）。

    positions [T,N,2]；返回每对首次碰撞时间/最小距离。
    """
    positions = _as_np(positions)
    times = _as_np(times)
    n = positions.shape[1]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    ev = compute_pair_first_hits(positions[:, None, :, :], times, pairs, d_safe)
    return {
        "pairs": pairs,
        "pair_first_time": ev["pair_first"][0].tolist(),
        "pair_min_distance": ev["pair_min"][0].tolist(),
        "team_first_time": (None if not np.isfinite(ev["team_first"][0])
                            else float(ev["team_first"][0])),
        "collision": bool(ev["team_hit"][0]),
    }
