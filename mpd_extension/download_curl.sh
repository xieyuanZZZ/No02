#!/usr/bin/env bash
# 用 usercontent 直链下载官方 data_public.tar.gz（支持断点续传）
set -x
mkdir -p "$HOME/mpd_data"
cd "$HOME/mpd_data"
curl -L -C - --retry 5 --retry-delay 5 \
  -o "$HOME/mpd_data/data_public.tar.gz" \
  "https://drive.usercontent.google.com/download?id=1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K&export=download&confirm=t"
echo "DOWNLOAD_EXIT=$?"
ls -lh "$HOME/mpd_data/data_public.tar.gz"
