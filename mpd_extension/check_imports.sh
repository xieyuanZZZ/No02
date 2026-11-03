#!/usr/bin/env bash
# 逐项检查官方 MPD 的 import 链路，定位缺口
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
cd /mnt/d/mpd_official
echo "=== python / torch ==="
$PY -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
echo "=== torch_robotics ==="
$PY -c "import torch_robotics; print('torch_robotics ok')" 2>&1 | tail -3
echo "=== torch_robotics.environments.env_simple_2d ==="
$PY -c "from torch_robotics.environments.env_simple_2d import EnvSimple2D; print('EnvSimple2D ok')" 2>&1 | tail -5
echo "=== torchkin / torchlie ==="
$PY -c "import torchkin, torchlie; print('torchkin', torchkin.__version__ if hasattr(torchkin,'__version__') else 'ok'); print('torchlie ok')" 2>&1 | tail -3
echo "=== isaacgym ==="
$PY -c "import isaacgym; print('isaacgym ok')" 2>&1 | tail -3
echo "=== pb_ompl ==="
$PY -c "import pb_ompl; print('pb_ompl ok')" 2>&1 | tail -3
echo "=== mpd.models ==="
$PY -c "import mpd.models as m; print('mpd.models ok')" 2>&1 | tail -5
echo "=== mpd.inference.inference ==="
$PY -c "from mpd.inference.inference import GenerativeOptimizationPlanner; print('inference module ok')" 2>&1 | tail -6
