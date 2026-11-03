#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
cd /mnt/d/mpd_official/scripts/inference

$PY - <<'PY'
import os
from mpd.utils.loaders import load_params_from_yaml
cfg = load_params_from_yaml('./cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml')
print("model_selection:", cfg["model_selection"])
print("planner_alg    :", cfg["planner_alg"])
print("sampling       :", cfg["diffusion_sampling_method"])
md = cfg["model_dir_ddpm_bspline"]
print("model_dir raw  :", md[:80], "...")
expanded = os.path.expandvars(md)
print("exists         :", os.path.isdir(expanded))
print("args.yaml      :", os.path.isfile(os.path.join(expanded, "args.yaml")))
print("ckpt           :", os.path.isfile(os.path.join(expanded, "checkpoints", "ema_model_current_state_dict.pth")))
print("n_samples      :", cfg["n_trajectory_samples"])
print("ddim cfg       :", dict(cfg["ddim"]))
PY
