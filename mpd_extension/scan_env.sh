#!/usr/bin/env bash
echo "=== home 目录 ==="
ls -la "$HOME" | head -25
echo "=== 大于 100MB 的文件 ==="
find "$HOME" -type f -size +100M 2>/dev/null | head -20
echo "=== 找 data_public / hdf5 / MMD 仓库 ==="
find "$HOME" /mnt/d -maxdepth 4 \( -name "data_public*" -o -name "*.hdf5" -o -name "mmd" -o -name "data_trained_models" \) 2>/dev/null | head -20
echo "=== 磁盘 ==="
df -h / | tail -1
