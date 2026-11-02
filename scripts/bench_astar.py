"""快速基准：A* 可用性与速度（用于选择内部栅格分辨率）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.data.astar import build_grid, plan_path  # noqa: E402
from fhd.geometry.collisions import path_segment_lengths  # noqa: E402
from fhd.maps.generate import generate_map  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402

cfg = load_config(ROOT / "configs" / "base.yaml")
v_max = float(cfg["trajectory"]["max_speed_mps"])
dt = float(cfg["trajectory"]["dt_s"])

for res in (128, 256):
    for kind in ("open", "dual_corridor", "narrow_gate", "bottleneck"):
        world = generate_map(kind, cfg, 20261103)
        rng = np.random.default_rng(0)
        starts = world.sample_free_points(rng, 4, min_sep=1.0, border_m=0.5)
        goals = world.sample_free_points(rng, 4, min_sep=0.8, border_m=0.5)
        t_grid = time.perf_counter()
        grid = build_grid(world, res)
        grid_ms = (time.perf_counter() - t_grid) * 1000
        t0 = time.perf_counter()
        ok = 0
        lens = []
        for s, g in zip(starts, goals):
            p = plan_path(world, s, g, resolution=res, h=32, rng=rng, noise_scale=0.35, grid=grid)
            if p is not None:
                ok += 1
                lens.append(float(path_segment_lengths(p.unsqueeze(0)).max()))
        dt_ms = (time.perf_counter() - t0) / 4 * 1000
        print(f"res={res:4d} {kind:15s} grid={grid_ms:6.1f}ms success={ok}/4 "
              f"mean={dt_ms:7.1f} ms/path maxseg={max(lens) if lens else float('nan'):.3f} m "
              f"(limit={v_max*dt:.2f})")
