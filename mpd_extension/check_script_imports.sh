#!/usr/bin/env bash
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/mpd/bin/python
export PYTHONPATH=/mnt/d/mpd_official:/mnt/d/mpd_official/wsl_setup/shims
cd /mnt/d/mpd_official/scripts/inference

echo "=== 逐步检查官方脚本的 import 链 ==="
$PY - <<'PY'
import traceback
mods = [
    ("isaacgym", "import isaacgym"),
    ("gymapi", "from isaacgym import gymapi"),
    ("gymutil", "from isaacgym import gymutil"),
    ("gymtorch", "from isaacgym import gymtorch"),
    ("parse_device_str", "from isaacgym.gymutil import parse_device_str"),
    ("isaacgym.torch_utils", "from isaacgym.torch_utils import *"),
    ("deps isaacgym torch_utils", "from deps.isaacgym.python.isaacgym.torch_utils import to_torch"),
    ("motion_planning_envs", "from torch_robotics.isaac_gym_envs.motion_planning_envs import MotionPlanningIsaacGymEnv"),
    ("RobotPanda", "from torch_robotics.robots import RobotPanda"),
    ("get_robot_path", "from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path"),
    ("experiment_launcher", "from experiment_launcher import single_experiment_yaml, run_experiment"),
]
for name, code in mods:
    try:
        exec(code)
        print(f"  OK   {name}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {str(e)[:160]}")
PY
