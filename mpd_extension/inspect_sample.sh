#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
source "$HOME/mpd_env.sh" 2>/dev/null || true
cd /mnt/d/mpd_official

$PY - <<'PY'
import os, torch
from dotmap import DotMap
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml

cfg = DotMap(load_params_from_yaml("scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml"))
cfg.model_dir = os.path.expandvars(cfg.model_dir_ddpm_bspline)
ta = DotMap(load_params_from_yaml(os.path.join(cfg.model_dir, "args.yaml")))
ta.update(**cfg, gripper=True, reload_data=False, results_dir="/tmp/x",
          load_indices=True, tensor_args={"device": torch.device("cpu"), "dtype": torch.float32})
pt, tr, _, va, _ = get_planning_task_and_dataset(**ta)
s = va[0]
print("type:", type(s))
print("keys:", list(s.keys()) if hasattr(s, "keys") else dir(s)[:40])
for k in (list(s.keys()) if hasattr(s, "keys") else []):
    v = s[k]
    try:
        print(f"  {k}: {type(v).__name__} shape={tuple(v.shape) if hasattr(v,'shape') else ''}")
    except Exception as e:
        print(f"  {k}: {type(v).__name__} ({e})")
print("dataset attrs:", [a for a in dir(va.dataset) if not a.startswith('_')][:40])
PY
