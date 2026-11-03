#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
ENV=$HOME/miniconda3/envs/mpd
PY=$ENV/bin/python

echo "=== 现有 nvrtc / cudnn ==="
find "$ENV" -name "libnvrtc*" 2>/dev/null | head
find "$ENV" -name "libcudnn_cnn_infer*" 2>/dev/null | head

echo "=== 安装 nvidia-cuda-nvrtc-cu11 ==="
$PY -m pip install --quiet "nvidia-cuda-nvrtc-cu11==11.8.89" 2>&1 | tail -3

echo "=== 找装好的 so ==="
find "$ENV" -name "libnvrtc*" 2>/dev/null | head

NVRTC=$(find "$ENV" -name "libnvrtc.so.11*" 2>/dev/null | head -1)
echo "NVRTC=$NVRTC"
if [ -n "$NVRTC" ]; then
  DIR=$(dirname "$NVRTC")
  ln -sfn "$NVRTC" "$DIR/libnvrtc.so"
  echo "已建软链: $DIR/libnvrtc.so"
  echo "export LD_LIBRARY_PATH=$DIR:\${LD_LIBRARY_PATH:-}" > "$HOME/mpd_env.sh"
  echo "LD_LIBRARY_PATH=$DIR"
  ls -l "$DIR"/libnvrtc*
fi

echo "=== 验证 torch cuda + 卷积 ==="
export LD_LIBRARY_PATH=$(dirname "$NVRTC"):${LD_LIBRARY_PATH:-}
$PY - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
x = torch.randn(8, 4, 32, 32, device="cuda")
w = torch.randn(8, 4, 3, 3, device="cuda")
y = torch.nn.functional.conv2d(x, w, padding=1)
print("conv on cuda ok:", tuple(y.shape))
PY
