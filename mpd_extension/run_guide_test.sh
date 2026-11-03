#!/usr/bin/env bash
set -x
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims:/mnt/d/paper2/src
cd /mnt/d/mpd_official

sed -i 's/\r$//' /mnt/d/mpd_official/mpd_extension/*.py

$PY mpd_extension/test_first_hit_guide.py 2>&1 | tail -25

echo "=== 下载进度 ==="
ls -lh /home/xuan/mpd_data/data_public.tar.gz 2>/dev/null
