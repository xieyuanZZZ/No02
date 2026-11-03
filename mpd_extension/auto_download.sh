#!/usr/bin/env bash
# 自动等待 Drive 配额恢复并下载官方 data_public.tar.gz
# - 每 10 分钟试一次；配额未恢复时 gdown 秒失败，不浪费带宽
# - 一旦开始传输就让它跑完；若中途断流，删除残缺文件重来（该端点不支持续传）
set -u
P="$HOME/miniconda3/bin/python"
F="$HOME/mpd_data/data_public_full.tar.gz"
EXPECT=22514915612
URL="https://drive.google.com/uc?id=1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K"
mkdir -p "$HOME/mpd_data"
cd "$HOME/mpd_data"

size() { stat -c %s "$F" 2>/dev/null || echo 0; }

for round in $(seq 1 150); do
  cur=$(size)
  if [ "$cur" -ge "$EXPECT" ]; then
    echo "[$(date +%H:%M:%S)] DONE size=$cur"
    break
  fi
  echo "[$(date +%H:%M:%S)] round=$round 现已下载=$cur 目标=$EXPECT"
  # 每轮先删残缺文件（端点不支持 Range）
  rm -f "$F"
  timeout 21600 "$P" -m gdown "$URL" -O "$F"
  rc=$?
  echo "  gdown rc=$rc size_now=$(size)"
  if [ "$(size)" -ge "$EXPECT" ]; then
    echo "[$(date +%H:%M:%S)] DONE"
    break
  fi
  sleep 600
done
ls -lh "$F" 2>/dev/null
