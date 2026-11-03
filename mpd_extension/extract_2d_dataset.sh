#!/usr/bin/env bash
# 从 3.5GB 前缀里抽取 EnvSimple2D 数据集（weights 在后面的 data_trained_models 段，拿不到）
set -u
F="$HOME/mpd_data/data_public.tar.gz"
OUT="$HOME/mpd_data/extracted"
mkdir -p "$OUT"
cd "$OUT"

echo "=== 抽取 EnvSimple2D 数据集 ==="
tar -xzf "$F" --wildcards \
  'data_public/data_trajectories/EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/*' \
  2>&1 | tail -5

echo "=== 抽取结果 ==="
find "$OUT" -type f -exec ls -lh {} \; | awk '{print $5, $9}'

echo "=== args.yaml ==="
cat "$OUT"/data_public/data_trajectories/EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/args.yaml 2>/dev/null | head -40

echo "=== hdf5 结构 ==="
$HOME/miniconda3/envs/mpd/bin/python - <<'PY'
import h5py, sys
p = "/home/xuan/mpd_data/extracted/data_public/data_trajectories/EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect/dataset_merged_doubled.hdf5"
try:
    with h5py.File(p, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")
        f.visititems(show)
        print("n_trajectories:", len(f["data"]))
except Exception as e:
    print("ERR", e)
PY
