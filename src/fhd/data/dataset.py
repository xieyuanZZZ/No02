"""数据集构建与加载（文档 §6.2）。

存储格式：
    data/dataset/maps_<split>.json   每张地图的连续几何描述
    data/dataset/traj_<split>.npz    paths [N,H,2]、starts [N,2]、goals [N,2]、map_idx [N]
    data/dataset/meta.json           种子、配置快照、丢弃统计

训练/验证/测试使用**互不重叠**的地图种子与任务。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fhd.data.preprocess import denormalise_paths, normalise_paths
from fhd.maps.generate import MAP_KINDS, WorldMap


def save_split(
    root: Path,
    split: str,
    maps: list[WorldMap],
    paths: list[torch.Tensor],
    starts: list[torch.Tensor],
    goals: list[torch.Tensor],
    map_idx: list[int],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"maps_{split}.json").write_text(
        json.dumps([m.to_dict() for m in maps], ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        root / f"traj_{split}.npz",
        paths=torch.stack(paths).numpy(),
        starts=torch.stack(starts).numpy(),
        goals=torch.stack(goals).numpy(),
        map_idx=np.asarray(map_idx, dtype=np.int64),
    )


def load_maps(root: Path, split: str) -> list[WorldMap]:
    data = json.loads((root / f"maps_{split}.json").read_text(encoding="utf-8"))
    return [WorldMap.from_dict(d) for d in data]


def load_split(root: Path, split: str) -> dict:
    z = np.load(root / f"traj_{split}.npz")
    return {
        "paths": torch.as_tensor(z["paths"], dtype=torch.float64),
        "starts": torch.as_tensor(z["starts"], dtype=torch.float64),
        "goals": torch.as_tensor(z["goals"], dtype=torch.float64),
        "map_idx": torch.as_tensor(z["map_idx"], dtype=torch.long),
    }


@dataclass
class Normalisation:
    width_m: float
    height_m: float

    def norm(self, paths: torch.Tensor) -> torch.Tensor:
        return normalise_paths(paths, self.width_m, self.height_m)

    def denorm(self, paths: torch.Tensor) -> torch.Tensor:
        return denormalise_paths(paths, self.width_m, self.height_m)


class TrajectoryDataset(Dataset):
    """单机器人条件轨迹数据集：返回 (x0[-1,1]^2, 条件起终点, 占据栅格)。"""

    def __init__(
        self,
        root: Path,
        split: str,
        grid_size: int = 64,
        robot_radius_m: float = 0.15,
        clearance_margin_m: float = 0.03,
        width_m: float = 10.0,
        height_m: float = 10.0,
    ):
        self.root = Path(root)
        self.split = split
        self.data = load_split(self.root, split)
        self.maps = load_maps(self.root, split)
        self.norm = Normalisation(width_m, height_m)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8")) \
            if (self.root / "meta.json").exists() else {}
        # 预计算 64×64 占据栅格（每张地图一次）
        grids = []
        for m in self.maps:
            occ = m.occupancy(torch.float32)
            if occ.shape[0] != grid_size:
                raise ValueError(f"栅格分辨率不一致: {occ.shape} vs {grid_size}")
            grids.append(occ)
        self.grids = torch.stack(grids, dim=0).unsqueeze(1) if grids else torch.zeros(0)
        self.paths_m = self.data["paths"]
        self.x0 = self.norm.norm(self.paths_m).to(torch.float32)
        self.starts_n = self.norm.norm(self.data["starts"]).to(torch.float32)
        self.goals_n = self.norm.norm(self.data["goals"]).to(torch.float32)

    def __len__(self) -> int:
        return int(self.x0.shape[0])

    def __getitem__(self, idx: int) -> dict:
        mi = int(self.data["map_idx"][idx])
        return {
            "x0": self.x0[idx],
            "start": self.starts_n[idx],
            "goal": self.goals_n[idx],
            "grid": self.grids[mi],
            "map_idx": torch.tensor(mi, dtype=torch.long),
            "path_m": self.paths_m[idx].to(torch.float32),
        }

    # 便于采样与评估
    def map_grid(self, map_idx: int) -> torch.Tensor:
        return self.grids[map_idx]

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm.denorm(x)


def dataset_stats(root: Path, split: str) -> dict:
    data = load_split(root, split)
    paths = data["paths"]
    lens = torch.linalg.vector_norm(paths[:, 1:] - paths[:, :-1], dim=-1)
    total = lens.sum(-1)
    return {
        "n_samples": int(paths.shape[0]),
        "n_maps": int(data["map_idx"].max().item()) + 1 if paths.numel() else 0,
        "length_mean_m": float(total.mean()),
        "length_std_m": float(total.std()),
        "length_min_m": float(total.min()),
        "length_max_m": float(total.max()),
        "max_segment_m": float(lens.max()),
        "start_goal_dist_mean_m": float(
            torch.linalg.vector_norm(data["goals"] - data["starts"], dim=-1).mean()
        ),
    }
