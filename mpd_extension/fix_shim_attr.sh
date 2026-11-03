#!/usr/bin/env bash
set -u
# 把各 shim 子模块的 __getattr__ 从 NotImplementedError 改为 AttributeError
# （模块协议要求：缺失属性必须是 AttributeError，否则 import 机制无法正常回退）
for f in \
  /mnt/d/mpd_official/wsl_setup/shims/isaacgym/gymapi.py \
  /mnt/d/mpd_official/wsl_setup/shims/isaacgym/gymutil.py \
  /mnt/d/mpd_official/wsl_setup/shims/isaacgym/gymtorch.py \
  /mnt/d/mpd_official/wsl_setup/shims/isaacgym/torch_utils.py \
  /mnt/d/mpd_official/wsl_setup/shims/isaacgym_pkg/python/isaacgym/torch_utils.py ; do
  sed -i 's/raise NotImplementedError(f"isaacgym\.\([a-z_.]*\)\.{name}: {_MSG}")/raise AttributeError(f"isaacgym.\1.{name}: {_MSG}")/' "$f"
  sed -i 's/raise NotImplementedError(f"\([a-zA-Z_.]*\)\.{name}: {_MSG}")/raise AttributeError(f"\1.{name}: {_MSG}")/' "$f"
  sed -i 's/raise NotImplementedError(f"isaacgym\.torch_utils\.{name}: {_MSG}")/raise AttributeError(f"isaacgym.torch_utils.{name}: {_MSG}")/' "$f"
  sed -i 's/raise NotImplementedError(f"deps\.isaacgym\.\.\.torch_utils\.{name}: {_MSG}")/raise AttributeError(f"deps...torch_utils.{name}: {_MSG}")/' "$f"
done
grep -n "raise " /mnt/d/mpd_official/wsl_setup/shims/isaacgym/gymutil.py
grep -n "raise " /mnt/d/mpd_official/wsl_setup/shims/isaacgym/torch_utils.py
