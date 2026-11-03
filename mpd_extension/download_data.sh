#!/usr/bin/env bash
# 下载官方 data_public.tar.gz（约 22.5 GB），支持断点续传
set -x
mkdir -p "$HOME/mpd_data"
cd "$HOME/mpd_data"
python3 - <<'PY' 2>/dev/null || true
PY
"$HOME/miniconda3/bin/python" -m gdown --continue \
    "https://drive.google.com/uc?id=1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K" \
    -O "$HOME/mpd_data/data_public.tar.gz"
echo "DOWNLOAD_DONE"
ls -lh "$HOME/mpd_data/"
