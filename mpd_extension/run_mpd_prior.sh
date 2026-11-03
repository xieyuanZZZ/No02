#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
export MPLBACKEND=Agg
cd /mnt/d/mpd_official
sed -i 's/\r$//' mpd_extension/run_mpd_2d.py

echo "=== A: 官方 Diffusion Prior（无引导）==="
$PY mpd_extension/run_mpd_2d.py --planner diffusion_prior --n-tasks 30 --n-samples 8 2>&1 | tail -40
