"""Isaac Gym 的**占位 shim**（仅用于让官方 MPD 的 import 通过）。

背景 / 证据：
  * 官方仓库里 `import isaacgym` 只出现在 3 个文件，其中
    `mpd/parametric_trajectory/trajectory_bspline.py` 与 `.../trajectory_waypoints.py`
    只是**导入**，全文没有任何 `isaacgym.` 属性调用；
  * 2D point-mass 环境的碰撞检查由 `torch_robotics` 的纯 PyTorch SDF 网格完成
    （`EnvSimple2D` 可在无 Isaac Gym 的情况下正常构建，已实测）；
  * Isaac Gym Preview 4 的官方 tarball 需要 NVIDIA 开发者账号登录，本机无法获取
    （记录见 `docs/MPD_复现与引导融合报告.md` 的 E1）。

因此这里提供一个最小 shim，**只暴露版本号**。任何对 Isaac Gym 真实功能的调用
都会立刻抛错，避免"看起来能跑、其实没用官方仿真"的静默错误。
"""

__version__ = "1.0.preview4-shim"

_MSG = (
    "这是 isaacgym 的 import-only shim：官方 MPD 的 2D 路径从不调用 Isaac Gym 的功能。"
    "如果你走到这里，说明用到了需要真实 Isaac Gym 的代码（3D/Panda 环境），"
    "请先安装官方 Isaac Gym Preview 4。"
)


def __getattr__(name):  # pragma: no cover - 只用于给出清晰错误
    raise RuntimeError(f"isaacgym.{name} 不可用：{_MSG}")
