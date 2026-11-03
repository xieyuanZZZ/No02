#!/usr/bin/env bash
set -u
P="/mnt/d/mpd_official/data_public/data_trained_models/launch_train_diffusion_models-v04_2024-09-17_21-44-17/generative_model_class___GaussianDiffusionModel/dataset_subdir___EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect"
echo "=== 两层结构 ==="
find "$P" -maxdepth 2 | head -20
echo "=== 最深层（含权重文件）==="
find "$P" -name "*.pt" -o -name "*.pth" -o -name "args.yaml" | head -20
echo "=== 大小 ==="
du -sh "$P"
