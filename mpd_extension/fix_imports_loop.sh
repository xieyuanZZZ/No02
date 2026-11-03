#!/usr/bin/env bash
set -x
export PATH=$HOME/miniconda3/bin:$PATH
PIP=$HOME/miniconda3/envs/mpd/bin/pip
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
cd /mnt/d/mpd_official

for i in 1 2 3 4 5; do
  MISSING=$($PY -c "from mpd.inference.inference import GenerativeOptimizationPlanner; print('OK')" 2>&1 | grep -oP "No module named '\K[^']+" | head -1)
  echo "第 $i 轮，缺失模块：[$MISSING]"
  if [ -z "$MISSING" ]; then break; fi
  $PIP install "$MISSING" 2>&1 | tail -2
done

echo "=== 最终检查 ==="
$PY -c "from mpd.inference.inference import GenerativeOptimizationPlanner; print('inference ok')" 2>&1 | tail -6
