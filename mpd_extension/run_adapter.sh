#!/usr/bin/env bash
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims:/mnt/d/paper2/src
cd /mnt/d/mpd_official
sed -i 's/\r$//' mpd_extension/simple_scenes_adapter.py
$PY mpd_extension/simple_scenes_adapter.py /mnt/d/paper2/data/revised_s1
