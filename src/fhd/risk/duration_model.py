"""M1/M2 的可解释段时长模型（阶段二 §2.1）。

语义（必须与阶段一 `trajectory_timing.py` / `progress.py` 对齐）：
  * 第 m 段是"离开航点 m → 到达航点 m+1"的**行驶**时间；
  * 航点等待 `w[m]` 单独由 `release_times_with_durations()` 加在累计时间上；
  * 因此这里拟合的 `duration[m]` **不包含**随机暂停与航点等待，
    校准数据必须来自 `p=0` 的 MPC 运行。若用含随机暂停的时长，
    `FirstHitRisk` 又乘 (1-p)，会把同一次等待重复计入——本模块
    显式拒绝这种输入。

M1：只用拟合的条件均值时长；
M2：在 M1 基础上，从校准运行的经验残差池中**按整条机器人轨迹**抽样
    （保留同一机器人在各段之间的相关性），逐机器人独立抽样时必须使用
    `pool[idx[n], :]`，不能像旧实现那样固定取 robot 0 的列。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "length_m", "v_cap_mps", "wait_m_slots", "wait_next_slots",
    "stop_at_m", "stop_at_m_plus_1", "cos_turn", "sin_turn", "progress_frac",
    "start_from_rest", "end_at_rest",
)


def segment_features(paths: np.ndarray, wait_slots: np.ndarray,
                     dt_slot: float = 0.4) -> np.ndarray:
    """构造 [N, H-1, F] 的**静态可预测**特征（不需要任何 MPC 运行）。"""
    paths = np.asarray(paths, dtype=np.float64)
    wait = np.asarray(wait_slots, dtype=np.float64)
    n, h, _ = paths.shape
    seg = np.diff(paths, axis=1)
    length = np.linalg.norm(seg, axis=-1)                   # [N,H-1]
    v_cap = length / max(dt_slot, 1e-9)
    wait_cur = wait[:, :h - 1]
    wait_next = np.concatenate([wait[:, 1:h - 1], np.zeros((n, 1))], axis=1)
    stop_m = (wait_cur > 1e-9).astype(np.float64)
    stop_next = (wait_next > 1e-9).astype(np.float64)
    # 转角在航点 m+1 处：相邻两段方向的夹角
    cos_turn = np.ones((n, h - 1))
    sin_turn = np.zeros((n, h - 1))
    for m in range(h - 2):
        a = seg[:, m]
        b = seg[:, m + 1]
        na = np.maximum(np.linalg.norm(a, axis=-1), 1e-12)
        nb = np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
        cos = np.clip((a * b).sum(-1) / (na * nb), -1.0, 1.0)
        cos_turn[:, m] = cos
        sin_turn[:, m] = np.sqrt(np.maximum(1.0 - cos ** 2, 0.0))
    frac = np.linspace(0.0, 1.0, h - 1)[None, :].repeat(n, axis=0)
    # 段 0 一定从静止出发；末段一定减速到终点静止；有等待的航点也是静止出发
    start_rest = np.zeros((n, h - 1))
    start_rest[:, 0] = 1.0
    start_rest = np.maximum(start_rest, stop_m)
    end_rest = np.zeros((n, h - 1))
    end_rest[:, -1] = 1.0
    end_rest = np.maximum(end_rest, stop_next)
    return np.stack([length, v_cap, wait_cur, wait_next, stop_m, stop_next,
                     cos_turn, sin_turn, frac, start_rest, end_rest], axis=-1)


@dataclass
class DurationModel:
    """岭回归均值 + 经验残差池（按整条机器人轨迹保存）。"""

    coef: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    intercept: float = 0.0
    alpha: float = 1e-3
    residual_pool: list = field(default_factory=list)   # 每项 [H-1]
    n_train_segments: int = 0
    coverage: float = 0.0
    train_rmse_slots: float = 0.0
    r2: float = 0.0

    # ------------------------------------------------------------------ #
    def fit(self, features: np.ndarray, durations: np.ndarray,
            run_ids: np.ndarray | None = None) -> "DurationModel":
        """features/kwargs: [R,H-1,F] 或 [M,F]；durations: [R,H-1] 或 [M]，NaN=删失。"""
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(durations, dtype=np.float64)
        squeeze = X.ndim == 2
        if squeeze:
            X = X[None, ...]
            y = y[None, ...]
        n_runs, h1, nf = X.shape
        flat_x = X.reshape(-1, nf)
        flat_y = y.reshape(-1)
        valid = np.isfinite(flat_y)
        self.n_train_segments = int(valid.sum())
        self.coverage = float(valid.mean()) if valid.size else 0.0
        if valid.sum() < nf + 2:
            raise ValueError(f"有效校准段太少：{int(valid.sum())}（特征维 {nf}）")
        self.mean = flat_x[valid].mean(axis=0)
        self.scale = flat_x[valid].std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        Z = (flat_x - self.mean) / self.scale
        Zv = Z[valid]
        yv = flat_y[valid]
        self.intercept = float(yv.mean())
        A = Zv.T @ Zv + self.alpha * np.eye(nf)
        self.coef = np.linalg.solve(A, Zv.T @ (yv - self.intercept))
        pred = self.intercept + Zv @ self.coef
        self.train_rmse_slots = float(np.sqrt(np.mean((pred - yv) ** 2)))
        ss_res = float(np.sum((pred - yv) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        self.r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
        # 整条机器人轨迹的残差池（pad 到 H-1，仍然在相同网格上）
        resid = y - self.predict(X)
        self.residual_pool = []
        for r in range(n_runs):
            row = np.where(np.isfinite(resid[r]), resid[r], 0.0)
            self.residual_pool.append(row.astype(np.float64))
        return self

    # ------------------------------------------------------------------ #
    def predict(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=np.float64)
        Z = (X - self.mean) / self.scale
        return self.intercept + Z @ self.coef

    def sample(self, features: np.ndarray, rng: np.random.Generator,
               n_draws: int) -> np.ndarray:
        """返回 [n_draws, N, H-1] 的段时长（时隙）。

        每个 draw、每台机器人独立选一条校准残差轨迹：
            dur[d, n, :] = pred[n, :] + pool[idx[d, n], :]
        旧实现 `samp[idx[:,None,None]][:,0,:]` 固定取了 robot 0，是 bug。
        """
        pred = self.predict(features)                    # [N,H-1]
        n = pred.shape[0]
        pool = np.asarray(self.residual_pool)
        if pool.size == 0:
            return np.repeat(pred[None, ...], n_draws, axis=0)
        idx = rng.integers(0, pool.shape[0], size=(n_draws, n))
        dur = pred[None, :, :] + pool[idx, :]
        return np.maximum(dur, 0.0)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "coef": self.coef.tolist() if self.coef is not None else None,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "scale": self.scale.tolist() if self.scale is not None else None,
            "intercept": self.intercept,
            "alpha": self.alpha,
            "residual_pool": [r.tolist() for r in self.residual_pool],
            "n_train_segments": self.n_train_segments,
            "coverage": self.coverage,
            "train_rmse_slots": self.train_rmse_slots,
            "r2": self.r2,
            "feature_names": list(FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DurationModel":
        m = cls(alpha=float(d.get("alpha", 1e-3)))
        m.coef = np.asarray(d["coef"], dtype=np.float64)
        m.mean = np.asarray(d["mean"], dtype=np.float64)
        m.scale = np.asarray(d["scale"], dtype=np.float64)
        m.intercept = float(d.get("intercept", 0.0))
        m.residual_pool = [np.asarray(r, dtype=np.float64) for r in d["residual_pool"]]
        m.n_train_segments = int(d.get("n_train_segments", 0))
        m.coverage = float(d.get("coverage", 0.0))
        m.train_rmse_slots = float(d.get("train_rmse_slots", 0.0))
        m.r2 = float(d.get("r2", 0.0))
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DurationModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
