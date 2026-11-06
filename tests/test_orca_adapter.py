"""ORCA/RVO2 adapter smoke tests (requires the bundled Windows runner)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fhd.adapters.orca_adapter import path_deviation, run_orca

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "mpd_delay_plan_bank" / "plan_bank.json"
TRAJ = ROOT / "data" / "mpd_delay_plan_bank" / "plan_bank_trajs.npz"


def _paths():
    d = json.loads(BANK.read_text(encoding="utf-8"))
    tr = np.load(TRAJ)
    jt = [t for t in d["tasks"] if t["task_id"] == "open_crossing_300000_pair02_11"][0]
    return np.asarray(tr[jt["task_id"]], dtype=float)


def test_orca_runs_and_respects_speed():
    paths = _paths()
    r = run_orca(paths, steps=300)
    assert r["traj"].shape[1:] == (2, 2)
    assert not r["collision"]
    assert r["all_arrived"]
    assert r["max_speed"] <= 1.5 + 1e-3
    assert np.isfinite(path_deviation(r["traj"], paths))


def test_orca_pause_event_is_observed():
    paths = _paths()
    r = run_orca(paths, events=[(10, 0, 0.8)], steps=300)
    assert r["online_requests"] > 0
    assert r["paused"].any()
