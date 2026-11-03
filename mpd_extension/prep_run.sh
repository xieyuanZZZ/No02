#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
R=/mnt/d/mpd_official
CFGDIR=$R/scripts/inference/cfgs

echo "=== 1) 记录权重与配置的 SHA256 ==="
$PY - <<'PY'
import hashlib, json, os
from mpd.utils.loaders import load_params_from_yaml
import sys
sys.path.insert(0, "/mnt/d/mpd_official")
cfg = load_params_from_yaml("/mnt/d/mpd_official/scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml")
d = os.path.expandvars(cfg["model_dir_ddpm_bspline"])
files = {
    "args.yaml": os.path.join(d, "args.yaml"),
    "ema_model_current_state_dict.pth": os.path.join(d, "checkpoints", "ema_model_current_state_dict.pth"),
    "model_current_state_dict.pth": os.path.join(d, "checkpoints", "model_current_state_dict.pth"),
}
out = {"model_dir": d, "planner_alg": cfg["planner_alg"],
       "model_selection": cfg["model_selection"], "diffusion_sampling_method": cfg["diffusion_sampling_method"]}
for k, p in files.items():
    if os.path.isfile(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        out[k] = {"path": p, "size_mb": round(os.path.getsize(p)/1024**2, 2), "sha256": h.hexdigest()}
print(json.dumps(out, indent=2, ensure_ascii=False))
open("/mnt/d/mpd_official/mpd_weights_manifest.json", "w").write(json.dumps(out, indent=2, ensure_ascii=False))
PY

echo "=== 2) 生成 MPD 配置副本（不改官方原文件）==="
sed "s/planner_alg: 'diffusion_prior'/planner_alg: 'mpd'/" \
  "$CFGDIR/config_EnvSimple2D-RobotPointMass2D_00.yaml" \
  > "$CFGDIR/config_EnvSimple2D-RobotPointMass2D_mpd.yaml"
grep -n "planner_alg" "$CFGDIR/config_EnvSimple2D-RobotPointMass2D_mpd.yaml"
grep -n "planner_alg" "$CFGDIR/config_EnvSimple2D-RobotPointMass2D_00.yaml"
