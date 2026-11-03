"""pb_ompl 的**占位 shim**（仅用于让官方 MPD 的 import 通过）。

证据（读官方源码得到）：
  * `mpd/inference/inference.py` 顶部 `from pb_ompl.pb_ompl import add_box, fit_bspline_to_path`；
  * `add_box` 只被 `add_start_goal_marker()` 使用（pybullet 可视化辅助）；
  * `fit_bspline_to_path` 只在 rrtconnect 基线分支里使用；
  * MPD / diffusion-prior 的采样与引导路径**完全不调用**它们。

真实 pb_ompl 需要编译 OMPL（官方 setup.sh 里要 `make update_bindings` + 多轮 `make`，
本机 Python 3.8 没有 conda 预编译的 ompl 绑定，编译耗时很长且与本阶段目标无关）。

这里提供的 shim 在**被调用时直接抛 NotImplementedError**，
保证「rrtconnect 基线」不会被悄悄替换成假实现。
"""

_MSG = (
    "pb_ompl 未安装（需要编译 OMPL）。本 shim 只允许 import，"
    "不提供任何真实功能；如果被调用，说明用到了 rrtconnect 基线，"
    "请先按官方 setup.sh 构建 pb_ompl。"
)


def add_box(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(_MSG)


def fit_bspline_to_path(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(_MSG)


def __getattr__(name):  # pragma: no cover
    raise NotImplementedError(f"pb_ompl.{name}：{_MSG}")
