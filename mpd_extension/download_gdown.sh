#!/usr/bin/env bash
# Google Drive 的 usercontent 端点不支持 Range 续传（curl rc=33），
# 改用 gdown：它自己解析 confirm token 并支持 --continue。
set -u
P=$HOME/miniconda3/bin/python
F="$HOME/mpd_data/data_public.tar.gz"
EXPECT=22514915612
mkdir -p "$HOME/mpd_data"
cd "$HOME/mpd_data"

size() { stat -c %s "$F" 2>/dev/null || echo 0; }

for i in $(seq 1 20); do
  cur=$(size)
  echo "attempt=$i have=$cur need=$EXPECT $(date +%H:%M:%S)"
  if [ "$cur" -ge "$EXPECT" ]; then echo "DONE"; break; fi
  $P -m gdown --continue \
     "https://drive.google.com/uc?id=1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K" -O "$F"
  echo "  gdown rc=$? size_now=$(size)"
  sleep 10
done
echo "FINAL size=$(size)"
ls -lh "$F" 2>/dev/null
