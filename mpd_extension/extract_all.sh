#!/usr/bin/env bash
set -u
echo "=== 校验归档 ==="
ls -l /mnt/d/mpd_official/data_public.tar.gz

echo "=== 解压到 /mnt/d/mpd_official（预计 10–20 分钟）==="
cd /mnt/d/mpd_official
time tar -xzf data_public.tar.gz
rc=$?
echo "tar rc=$rc"

echo "=== 结果 ==="
ls -la /mnt/d/mpd_official/data_public/ 2>/dev/null
du -sh /mnt/d/mpd_official/data_public/* 2>/dev/null
