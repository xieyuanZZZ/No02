#!/usr/bin/env bash
echo "=== data_trained_models.tar.xz 内容（前 20 个唯一二级目录）==="
tar -tJf "$HOME/delay_robust_mmd/downloads/data_trained_models.tar.xz" 2>/dev/null | head -400 | cut -d/ -f1,2 | sort -u | head -20
echo "=== 全盘找 MPD 关键模块 ==="
find "$HOME" -name "env_simple_2d.py" 2>/dev/null | head
find "$HOME" -name "trajectory_bspline.py" 2>/dev/null | head
echo "=== 找 EnvSimple2D 命名的权重 ==="
find "$HOME" -iname "*EnvSimple2D*" 2>/dev/null | head -10
echo "=== MMD README 摘要 ==="
head -30 "$HOME/delay_robust_mmd/README.md"
