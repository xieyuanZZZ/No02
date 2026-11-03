#!/usr/bin/env bash
# 从已下载的 3.4 GB 前缀里看能否取出 2D 预训练模型
set -u
F="$HOME/mpd_data/data_public.tar.gz"
echo "=== 前缀大小 ==="
ls -lh "$F"
echo "=== 前 40 个条目 ==="
tar -tzf "$F" 2>/dev/null | head -40
echo "=== 含 EnvSimple2D 的条目（限制 40 行）==="
tar -tzf "$F" 2>/dev/null | grep -i "EnvSimple2D" | head -40
echo "=== 顶层目录统计 ==="
tar -tzf "$F" 2>/dev/null | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -rn | head -12
