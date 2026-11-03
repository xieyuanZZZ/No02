#!/usr/bin/env bash
set -x
export PATH=$HOME/miniconda3/bin:$PATH
PIP=$HOME/miniconda3/envs/mpd/bin/pip
PY=$HOME/miniconda3/envs/mpd/bin/python

$PIP install urdf-parser-py lxml wandb pybullet 2>&1 | tail -4

echo "=== 再查 import ==="
$PY -c "import lxml, wandb, urdf_parser_py, pybullet; print('simple deps ok')"
$PY -c "from torch_robotics.environments.env_simple_2d import EnvSimple2D; print('EnvSimple2D ok')" 2>&1 | tail -3
$PY -c "import torchkin; print('torchkin ok')" 2>&1 | tail -3
$PY -c "from mpd.inference.inference import GenerativeOptimizationPlanner; print('inference ok')" 2>&1 | tail -4

echo "=== 尝试 usercontent 直链下载（前 20MB 探测）==="
curl -sL -r 0-20000000 -o /tmp/probe.bin --max-time 120 \
  "https://drive.usercontent.google.com/download?id=1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K&export=download&confirm=t" \
  && ls -lh /tmp/probe.bin && head -c 200 /tmp/probe.bin | xxd | head -3
