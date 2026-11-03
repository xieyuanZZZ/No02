"""pb_ompl.pb_ompl 子模块的 shim（与包级 shim 配套）。

官方 `scripts/generate_data/generate_trajectories.py` 还会导入
`PbOMPLRobot, PbOMPL, add_sphere, add_box`，这里全部以"调用即报错"的占位实现提供，
只为让 import 通过；任何真实的 OMPL 规划都会立刻失败并提示。
"""

from . import _MSG


def _unavailable(name):
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"pb_ompl.{name}: {_MSG}")

    _fn.__name__ = name
    return _fn


add_box = _unavailable("add_box")
add_sphere = _unavailable("add_sphere")
fit_bspline_to_path = _unavailable("fit_bspline_to_path")


class PbOMPLRobot:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"pb_ompl.PbOMPLRobot: {_MSG}")


class PbOMPL:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"pb_ompl.PbOMPL: {_MSG}")
