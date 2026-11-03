#!/usr/bin/env bash
set -u
echo "=== 1) 权重目录 ==="
M=/mnt/d/mpd_official/data_public/data_trained_models
du -sh "$M"
echo "--- EnvSimple2D 相关 ---"
find "$M" -maxdepth 2 -type d -name "*EnvSimple2D*" | head

echo "=== 2) 配置里要求的那个目录是否存在 ==="
CFG="/mnt/d/mpd_official/scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml"
TARGET=$(grep "model_dir_ddpm_bspline" "$CFG" | sed "s/.*'\\\${HOME}\/Projects\/MotionPlanningDiffusion\/mpd-splines\///; s/'$//")
echo "配置里的相对路径:"
echo "$TARGET"
echo "拼到提取目录后:"
FULL="$M/${TARGET#data_trained_models/}"
echo "$FULL"
if [ -d "$FULL" ]; then echo "✅ 目录存在"; ls -la "$FULL" | head -12; else echo "❌ 目录不存在"; fi

echo "=== 3) 数据集（推理需要它做反归一化） ==="
D="$HOME/mpd_data/extracted/data_public/data_trajectories"
ls -1 "$D" 2>/dev/null
du -sh "$D" 2>/dev/null

echo "=== 4) 建出官方期望的目录结构（软链接，不改官方配置路径） ==="
BASE="$HOME/Projects/MotionPlanningDiffusion/mpd-splines"
mkdir -p "$BASE"
ln -sfn "$M" "$BASE/data_trained_models"
ln -sfn "$D" "$BASE/data_trajectories"
ls -la "$BASE"
