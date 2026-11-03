#!/usr/bin/env bash
set -x
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
cd /mnt/d/mpd_official

sed -i 's/\r$//' /mnt/d/mpd_official/wsl_setup/shims/pb_ompl/pb_ompl.py

echo "=== mpd.inference.inference ==="
$PY -c "from mpd.inference.inference import GenerativeOptimizationPlanner; print('inference ok')" 2>&1 | tail -6

echo "=== 下载进度 ==="
ls -lh /home/xuan/mpd_data/data_public.tar.gz 2>/dev/null || echo "尚无文件"
