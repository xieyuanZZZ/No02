"""M0/M1/M2 的纯预测入口（阶段二 §2.1）。

本模块**只依赖静态规划输入**（paths、wait、pause_p）与已拟合的
`DurationModel`；它不导入、不调用 `simulate_mpc`，因此测试任务的
真实执行结果不可能泄漏进预测。

  M0：原始 First-Hit DP（每段 1 时隙，不含额外时长信息）
  M1：条件均值段时长（由校准集 p=0 运行拟合）
  M2：从校准残差池按整条机器人轨迹抽样，得到时长分布后对多组时长
      分别跑 First-Hit DP，再对首次碰撞概率取平均。
"""

from __future__ import annotations

import numpy as np
import torch

from fhd.risk.duration_model import DurationModel, segment_features


def predict_m0(risk, paths: torch.Tensor, wait: torch.Tensor,
               pause_p: torch.Tensor, time_steps: int) -> np.ndarray:
    with torch.no_grad():
        out = risk(paths, wait, pause_p, time_steps, collision_mode="hard",
                   gate_mode="hard", return_diagnostics=False)
    return np.asarray(out["pair_risks"].detach().cpu(), dtype=np.float64)


def predict_m1(risk, paths: torch.Tensor, wait: torch.Tensor,
               pause_p: torch.Tensor, time_steps: int,
               durations_slots: np.ndarray) -> np.ndarray:
    dur = torch.as_tensor(durations_slots, dtype=torch.float64)
    with torch.no_grad():
        out = risk(paths, wait, pause_p, time_steps, collision_mode="hard",
                   gate_mode="hard", return_diagnostics=False,
                   segment_durations=dur,
                   duration_semantics="travel_excl_pause")
    return np.asarray(out["pair_risks"].detach().cpu(), dtype=np.float64)


def predict_m2(risk, paths: torch.Tensor, wait: torch.Tensor,
               pause_p: torch.Tensor, time_steps: int,
               duration_samples: np.ndarray) -> np.ndarray:
    """duration_samples [D,N,H-1] -> 对 D 次首次碰撞概率取平均。"""
    acc = np.zeros(len(pair_index(paths.shape[0])), dtype=np.float64)
    for d in range(duration_samples.shape[0]):
        acc += predict_m1(risk, paths, wait, pause_p, time_steps, duration_samples[d])
    return acc / max(duration_samples.shape[0], 1)


def pair_index(n_agents: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]


def predict_m0m1m2(risk, paths: torch.Tensor, wait: torch.Tensor,
                   pause_p: torch.Tensor, time_steps: int,
                   duration_model: DurationModel | None,
                   n_draws: int = 24,
                   rng: np.random.Generator | None = None,
                   dt_slot: float = 0.4) -> dict:
    """静态输入 -> M0/M1/M2 两两风险。**不读取任何测试执行结果。**"""
    n = paths.shape[0]
    m0 = predict_m0(risk, paths, wait, pause_p, time_steps)
    if duration_model is None:
        return {"m0": m0, "m1": m0.copy(), "m2": m0.copy(),
                "durations": np.ones((n, paths.shape[1] - 1)), "samples": None}
    feats = segment_features(paths.detach().cpu().numpy(),
                             wait.detach().cpu().numpy(), dt_slot=dt_slot)
    pred = np.maximum(duration_model.predict(feats), 0.0)
    m1 = predict_m1(risk, paths, wait, pause_p, time_steps, pred)
    rng = np.random.default_rng(0) if rng is None else rng
    samples = duration_model.sample(feats, rng, n_draws)
    m2 = predict_m2(risk, paths, wait, pause_p, time_steps, samples)
    return {"m0": m0, "m1": m1, "m2": m2, "durations": pred, "samples": samples}
