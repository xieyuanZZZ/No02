#!/usr/bin/env bash
set -u
echo "=== 停掉残留 tar 进程 ==="
pkill -f "tar -xzf data_public.tar.gz" 2>/dev/null
sleep 2

echo "=== 删除已解压的 data_trajectories（37 GB，用不到）==="
rm -rf /mnt/d/mpd_official/data_public
sync
df -h /mnt/d | tail -1

echo "=== 只解压 data_trained_models（不写 trajectories，快很多）==="
cd /mnt/d/mpd_official
time tar -xzf data_public.tar.gz --wildcards 'data_public/data_trained_models/*'
echo "tar rc=$?"

echo "=== 结果 ==="
du -sh /mnt/d/mpd_official/data_public/data_trained_models 2>/dev/null
ls -1 /mnt/d/mpd_official/data_public/data_trained_models 2>/dev/null
df -h /mnt/d | tail -1
