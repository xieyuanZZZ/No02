#!/usr/bin/env bash
echo "=== delay_robust_mmd 顶层 ==="
ls -la "$HOME/delay_robust_mmd" | head -20
echo "=== mmd 仓库 ==="
ls -la "$HOME/delay_robust_mmd/mmd" | head -20
echo "=== data_trained_models 结构（深度 3）==="
find "$HOME/delay_robust_mmd/mmd/data_trained_models" -maxdepth 3 2>/dev/null | head -30
echo "=== 找 EnvSimple2D 相关权重 ==="
find "$HOME/delay_robust_mmd" -path "*EnvSimple2D*" -name "*.pt" 2>/dev/null | head -20
echo "=== downloads 目录 ==="
ls -lh "$HOME/delay_robust_mmd/downloads" 2>/dev/null
echo "=== 是否 git 仓库 ==="
ls -d "$HOME/delay_robust_mmd/mmd/.git" 2>/dev/null && (cd "$HOME/delay_robust_mmd/mmd" && git log -1 --format='%H %s')
