#!/usr/bin/env bash
# 稳健下载：断点续传 + 自动重试，直到文件达到官方大小 22514915612 字节
set -u
F="$HOME/mpd_data/data_public.tar.gz"
URL="https://drive.usercontent.google.com/download?id=1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K&export=download&confirm=t"
EXPECT=22514915612
mkdir -p "$HOME/mpd_data"

size() { stat -c %s "$F" 2>/dev/null || echo 0; }

for i in $(seq 1 80); do
  cur=$(size)
  if [ "$cur" -ge "$EXPECT" ]; then
    echo "DONE size=$cur attempt=$i"
    break
  fi
  echo "attempt=$i have=$cur need=$EXPECT $(date +%H:%M:%S)"
  # -s 静默进度；--speed-limit 触发超时重连；-C - 断点续传
  curl -sS -L -C - \
       --connect-timeout 30 --max-time 3600 \
       --speed-limit 20000 --speed-time 120 \
       --retry 3 --retry-delay 5 --retry-all-errors \
       -o "$F" "$URL"
  rc=$?
  echo "  curl rc=$rc size_now=$(size)"
  sleep 5
done

echo "FINAL size=$(size) expect=$EXPECT"
ls -lh "$F" 2>/dev/null
