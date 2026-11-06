"""Continuous baseline adapter tests (no frozen No02 algorithm changes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fhd.adapters.mmd_adapter import load_mmd_paths, path_deviation, validate_plan

ROOT = Path(__file__).resolve().parents[1]
MMD_JSON = (ROOT / "results" / "Continuous_Baseline_Comparison" /
            "native_reproduction" / "mmd" / "mmd_paths.json")


def test_mmd_paths_resample_to_frozen_h_and_validate():
    assert MMD_JSON.exists(), "MMD native export missing"
    plan = load_mmd_paths(MMD_JSON, h=32)
    assert plan.paths.shape[1:] == (32, 2)
    assert plan.paths.shape[0] == len(plan.native_paths) == 3
    report = validate_plan(plan)
    assert report["ok"], report["issues"]
    assert plan.planning_time_s > 0


def test_mmd_resampling_is_deterministic_and_keeps_endpoints():
    a = load_mmd_paths(MMD_JSON, h=32)
    b = load_mmd_paths(MMD_JSON, h=32)
    assert np.allclose(a.paths, b.paths)
    for i in range(3):
        assert np.linalg.norm(a.paths[i, 0] - a.native_paths[i][0, :2]) < 1e-9
        assert np.linalg.norm(a.paths[i, -1] - a.native_paths[i][-1, :2]) < 1e-9


def test_path_deviation_zero_on_self():
    plan = load_mmd_paths(MMD_JSON, h=32)
    assert path_deviation(plan.paths[0], plan.paths[0]) < 1e-9
