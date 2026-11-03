#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
export MPLBACKEND=Agg
source "$HOME/mpd_env.sh" 2>/dev/null || true
cd /mnt/d/mpd_official
PLANNER="${1:-diffusion_prior}"
sed -i 's/\r$//' mpd_extension/run_mpd_scenes.py
echo "=== 场景测试：$PLANNER ==="
$PY mpd_extension/run_mpd_scenes.py --planner "$PLANNER" --per-kind 10 2>&1 | tail -45
