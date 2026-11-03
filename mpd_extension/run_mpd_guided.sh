#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
export MPLBACKEND=Agg
source "$HOME/mpd_env.sh" 2>/dev/null || true
cd /mnt/d/mpd_official
sed -i 's/\r$//' mpd_extension/run_mpd_2d.py

echo "=== B: 官方 MPD（SDF 代价引导）==="
$PY mpd_extension/run_mpd_2d.py --planner mpd --n-tasks 30 --n-samples 8 2>&1 | tail -40
