#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export MPLBACKEND=Agg
cd /mnt/d/mpd_official
sed -i 's/\r$//' mpd_extension/plot_scenes.py
$PY mpd_extension/plot_scenes.py
