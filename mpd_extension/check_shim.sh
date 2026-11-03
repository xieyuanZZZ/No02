#!/usr/bin/env bash
# 在 shim + 官方代码下，检查 MPD 2D 推理链路能否 import
set -x
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
cd /mnt/d/mpd_official

echo "=== isaacgym shim ==="
$PY -c "import isaacgym; print('shim version', isaacgym.__version__)"

echo "=== pb_ompl shim ==="
$PY -c "from pb_ompl.pb_ompl import add_box; print('pb_ompl shim ok')"

echo "=== trajectory_bspline ==="
$PY -c "from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline; print('bspline ok')" 2>&1 | tail -6

echo "=== cost_guides ==="
$PY -c "from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory; print('cost_guides ok')" 2>&1 | tail -8

echo "=== mpd.inference.inference ==="
$PY -c "from mpd.inference.inference import GenerativeOptimizationPlanner; print('inference ok')" 2>&1 | tail -8
