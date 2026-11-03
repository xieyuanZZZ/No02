#!/usr/bin/env bash
set -u
echo "=== 编译工具 ==="
which cmake g++ gcc make 2>/dev/null
cmake --version 2>/dev/null | head -1
g++ --version 2>/dev/null | head -1

echo "=== 头文件/库 ==="
for p in /usr/include/eigen3 /usr/include/boost /usr/local/include/ompl /usr/include/ompl; do
  [ -e "$p" ] && echo "有 $p" || echo "无 $p"
done
ls /usr/lib/x86_64-linux-gnu/libboost_* 2>/dev/null | head -3
ls /usr/lib/x86_64-linux-gnu/libompl* 2>/dev/null | head -3

echo "=== conda 环境里的 ompl/eigen ==="
ls "$HOME/miniconda3/envs/mpd/include" 2>/dev/null | grep -i -E "ompl|eigen|boost" | head
ls "$HOME/miniconda3/envs/mpd/lib" 2>/dev/null | grep -i ompl | head -3

echo "=== pybullet_ompl 结构 ==="
ls /mnt/d/mpd_official/deps/pybullet_ompl | head -20
sed -n '1,40p' /mnt/d/mpd_official/deps/pybullet_ompl/setup.py 2>/dev/null
