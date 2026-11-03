#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
export MPLBACKEND=Agg
cd /mnt/d/mpd_official/scripts/inference

echo "=== 官方 2D 冒烟：diffusion_prior，8 条候选 ==="
$PY inference.py \
  --results_dir /mnt/d/mpd_official/results_mpd \
  --id 0 \
  --num_workers 0 \
  --n_trajectory_samples 8 \
  --planner_alg diffusion_prior \
  2>&1 | tail -60
echo "EXIT=$?"
