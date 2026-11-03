#!/usr/bin/env bash
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims:/mnt/d/paper2/src
sed -i 's/\r$//' /mnt/d/mpd_official/mpd_extension/debug_risk.py
$PY /mnt/d/mpd_official/mpd_extension/debug_risk.py
