#!/usr/bin/env bash
# MPD 官方环境安装（最小可用集）
# 目标：让 mpd 的 2D point-mass 推理链路可以 import 并运行
set -x
export PATH=$HOME/miniconda3/bin:$PATH
PIP=$HOME/miniconda3/envs/mpd/bin/pip
$PIP install --upgrade pip

# 官方 environment.yml 指定 torch==2.0.0+cu118 / torchvision==0.15.1+cu118
$PIP install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 官方 environment.yml 中的纯 python 依赖（去掉需要 C++ 构建的 theseus/pb_ompl）
$PIP install einops==0.8.0 h5py==3.11.0 hydra-core==1.3.2 omegaconf==2.3.0 \
    dotmap scikit-learn matplotlib scipy tqdm pyyaml termcolor \
    torchdiffeq==0.2.4 torch-kmeans==0.2.0 torch-summary==1.4.5 torchlie==0.1.0 trimesh

# 官方仓库内的两个纯 python 包
cd /mnt/d/mpd_official/mpd/torch_robotics && $PIP install -e . --no-deps
cd /mnt/d/mpd_official/deps/theseus/torchkin && $PIP install -e . --no-deps || echo "torchkin 跳过（需要 C++ 构建）"
cd /mnt/d/mpd_official/deps/experiment_launcher && $PIP install -e . --no-deps

echo "INSTALL_DONE"
