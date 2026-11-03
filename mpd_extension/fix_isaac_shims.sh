#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
R=/mnt/d/mpd_official

echo "=== 1) 在官方期望落点放占位包（软链接，不改官方仓库内容）==="
if [ -e "$R/deps/isaacgym" ] && [ ! -L "$R/deps/isaacgym" ]; then
  echo "deps/isaacgym 已存在（非软链），跳过"
else
  ln -sfn "$R/wsl_setup/shims/isaacgym_pkg" "$R/deps/isaacgym"
fi
ls -la "$R/deps/" | head

echo "=== 2) 补 cv2 / imageio / moviepy ==="
$PY -m pip install --quiet opencv-python-headless imageio moviepy 2>&1 | tail -3

echo "=== 3) 重新检查 import 链 ==="
export PYTHONPATH=$R:$R/wsl_setup/shims
$PY - <<'PY'
mods = [
    ("gymapi", "from isaacgym import gymapi"),
    ("gymutil", "from isaacgym import gymutil"),
    ("gymtorch", "from isaacgym import gymtorch"),
    ("parse_device_str", "from isaacgym.gymutil import parse_device_str"),
    ("isaacgym.torch_utils", "from isaacgym.torch_utils import *"),
    ("deps torch_utils", "from deps.isaacgym.python.isaacgym.torch_utils import to_torch"),
    ("motion_planning_envs", "from torch_robotics.isaac_gym_envs.motion_planning_envs import MotionPlanningIsaacGymEnv, MotionPlanningControllerIsaacGym"),
    ("RobotPanda", "from torch_robotics.robots import RobotPanda"),
    ("get_robot_path", "from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path"),
]
for name, code in mods:
    try:
        exec(code)
        print(f"  OK   {name}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {str(e)[:140]}")
PY
