#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims

echo "=== 权重 / 配置 SHA256 ==="
$PY - <<'PY'
import hashlib, json, os
from mpd.utils.loaders import load_params_from_yaml

cfg = load_params_from_yaml("/mnt/d/mpd_official/scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml")
d = os.path.expandvars(cfg["model_dir_ddpm_bspline"])
files = {
    "args.yaml": os.path.join(d, "args.yaml"),
    "ema_model_current_state_dict.pth": os.path.join(d, "checkpoints", "ema_model_current_state_dict.pth"),
    "model_current_state_dict.pth": os.path.join(d, "checkpoints", "model_current_state_dict.pth"),
}
out = {"model_dir": d, "planner_alg": cfg["planner_alg"],
       "model_selection": cfg["model_selection"],
       "diffusion_sampling_method": cfg["diffusion_sampling_method"],
       "n_trajectory_samples": cfg["n_trajectory_samples"],
       "ddim": dict(cfg["ddim"])}
for k, p in files.items():
    if os.path.isfile(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        out[k] = {"size_mb": round(os.path.getsize(p) / 1024**2, 2), "sha256": h.hexdigest()}
print(json.dumps(out, indent=2, ensure_ascii=False))
open("/mnt/d/mpd_official/mpd_weights_manifest.json", "w").write(
    json.dumps(out, indent=2, ensure_ascii=False))
PY
