#!/usr/bin/env bash
set -x
export PATH=$HOME/miniconda3/bin:$PATH
PIP=$HOME/miniconda3/envs/mpd/bin/pip
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official

$PIP install --quiet colour ghalton sobol_seq 2>&1 | tail -2

echo "=== mpd 包结构 ==="
ls /mnt/d/mpd_official/mpd/__init__.py 2>/dev/null && echo "mpd 有 __init__" || echo "mpd 无 __init__"

echo "=== 构建官方 EnvSimple2D（纯 torch，不需要 Isaac Gym） ==="
$PY - <<'PY'
from torch_robotics.environments.env_simple_2d import EnvSimple2D
import torch
env = EnvSimple2D()
print("EnvSimple2D ok: dim =", env.dim, "| n_objects =", len(env.object_list))
print("limits:", env.limits)
PY

echo "=== 直接用 torch_robotics 的 SDF 网格检查碰撞能力 ==="
$PY - <<'PY'
import torch
from torch_robotics.environments.env_simple_2d import EnvSimple2D
env = EnvSimple2D()
pts = torch.tensor([[0.0, 0.0], [-0.4337, 0.3334], [0.5, 0.5]])
print("free?", env.get_collision_bool_from_positions(pts) if hasattr(env, "get_collision_bool_from_positions") else "no method")
PY
