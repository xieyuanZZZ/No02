"""按 plan bank 的 `map_id` 载入静态障碍（阶段一 §1.3 静态障碍检查）。

plan bank 的 `map_id` 形如 `open_crossing_300000`，末尾数字是 revised_s1
地图的 `seed`。若能在 `data/revised_s1/maps_*.json` 中找到同 seed 且同 kind
的地图，就返回 `Obstacles`；否则返回 None，评测侧必须把静态障碍指标标为
`not_evaluated`，不能伪造 0 碰撞。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fhd.geometry.primitives import Obstacles

_ROOT = Path(__file__).resolve().parents[3]
_CACHE: dict | None = None


def _load_catalog() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    cat = {}
    base = _ROOT / "data" / "revised_s1"
    for name in ("maps_train.json", "maps_val.json", "maps_test.json"):
        f = base / name
        if not f.exists():
            continue
        for m in json.loads(f.read_text(encoding="utf-8")):
            cat[(int(m["seed"]), str(m["kind"]))] = m
    _CACHE = cat
    return cat


def parse_map_id(map_id: str) -> tuple[int | None, str | None]:
    parts = str(map_id).rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None, None
    return int(parts[1]), parts[0]


def load_obstacles_for_map(map_id: str) -> tuple[Obstacles | None, dict]:
    """返回 (Obstacles|None, meta)。meta 记录是否成功载入及原因。"""
    seed, kind = parse_map_id(map_id)
    if seed is None:
        return None, {"status": "not_evaluated", "reason": f"无法解析 map_id={map_id!r}"}
    cat = _load_catalog()
    rec = cat.get((seed, kind))
    if rec is None:
        return None, {"status": "not_evaluated",
                      "reason": f"地图库没有 seed={seed}, kind={kind}"}
    circles = torch.tensor(rec.get("circles", []), dtype=torch.float64).reshape(-1, 3)
    rects = torch.tensor(rec.get("rects", []), dtype=torch.float64).reshape(-1, 4)
    obs = Obstacles(circles=circles, rects=rects)
    return obs, {
        "status": "evaluated",
        "n_circles": int(circles.shape[0]),
        "n_rects": int(rects.shape[0]),
        "robot_radius_m": float(rec.get("robot_radius_m", 0.15)),
        "clearance_margin_m": float(rec.get("clearance_margin_m", 0.03)),
        "seed": int(seed),
        "kind": str(kind),
    }
