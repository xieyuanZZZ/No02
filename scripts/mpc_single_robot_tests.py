"""MPC 单机正确性测试：直线/曲线跟踪、加速、制动、重启、连续暂停、饱和。

对应实施安排 §4.3 的八项测试。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fhd.control.linear_mpc import MPCParams  # noqa: E402
from fhd.dynamics.double_integrator import DoubleIntegrator, DoubleIntegratorParams  # noqa: E402
from fhd.dynamics.trajectory_timing import TimeParameterizedPath, TimingParams  # noqa: E402
from fhd.execution.mpc_executor import SingleRobotExecutor  # noqa: E402
from fhd.utils.logging import get_logger, log_section  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402


def make_path(kind: str, n_wp: int = 32) -> np.ndarray:
    t = np.linspace(0, 1, n_wp)
    if kind == "line":
        return np.stack([1.0 + 8.0 * t, 5.0 * np.ones_like(t)], axis=-1)
    if kind == "curve":
        return np.stack([1.0 + 8.0 * t, 5.0 + 2.0 * np.sin(np.pi * t)], axis=-1)
    if kind == "sharp":
        # 高曲率：三次折返
        return np.stack([1.0 + 8.0 * t, 5.0 + 1.5 * np.sin(3 * np.pi * t)], axis=-1)
    if kind == "corner":
        return np.stack([np.concatenate([1.0 + 4.0 * np.minimum(t * 2, 1),
                                         5.0 + 4.0 * np.maximum(t * 2 - 1, 0)])],
                        axis=-1) if False else np.stack(
            [1.0 + 4.0 * np.minimum(t / 0.5, 1.0), 5.0 + 4.0 * np.maximum((t - 0.5) / 0.5, 0.0)],
            axis=-1)
    raise ValueError(kind)


def run_case(kind: str, pause_schedule, cfg, logger, name):
    wp = make_path(kind)
    wait = np.zeros((len(wp),))
    path = TimeParameterizedPath(wp, wait, TimingParams(v_max=cfg["v_max"],
                                                        a_acc=cfg["a_max"],
                                                        a_brake=cfg["a_brake"]))
    di = DoubleIntegrator(DoubleIntegratorParams(dt=0.05, dt_ctrl=0.1,
                                                 v_max=cfg["v_max"], a_max=cfg["a_max"],
                                                 a_brake=cfg["a_brake"]))
    ex = SingleRobotExecutor(path, di, MPCParams(horizon=18, v_max=cfg["v_max"],
                                                 a_max=cfg["a_max"]), dt_ctrl=0.1)
    ex.reset()
    ex.x = np.array([wp[0, 0], wp[0, 1], 0.0, 0.0])
    pause_at = dict(pause_schedule)
    t_end = 60.0
    while ex.t < t_end and not ex.done:
        req = False
        dur = 0.4
        for (tp, d) in list(pause_at.items()):
            if abs(ex.t - tp) < 1e-9:
                req, dur = True, d
        ex.step(pause_request=req, pause_duration=dur)
    arr = ex.log.arrays()
    pos_err = np.array([np.linalg.norm(ex.path.position_at(ex.path.project(arr["p"][k]))
                                       - arr["p"][k]) for k in range(len(arr["p"]))])
    row = {
        "case": name, "path_kind": kind,
        "done": bool(ex.done), "completion_s": float(ex.t),
        "planned_motion_s": float(path.duration_motion),
        "tracking_rmse_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "tracking_max_m": float(pos_err.max()),
        "max_speed": float(np.max(np.linalg.norm(arr["v"], axis=-1))),
        "max_accel": float(np.max(np.linalg.norm(arr["a"], axis=-1))),
        "vpause_count": int(ex.pause_count),
        "extra_brake_s": float(ex.extra_brake_time),
        "constraint_violations": int(ex.constraint_violations),
        "mpc_failures": int(ex.mpc.fail_count),
        "mean_solve_ms": float(1000 * ex.mpc.last_solve_time),
        "len": float(path.length),
    }
    logger.info("  %-22s done=%s 完成 %.2fs rmse=%.3f max_err=%.3f vmax=%.2f amax=%.2f "
                "违规=%d QP失败=%d", name, row["done"], row["completion_s"],
                row["tracking_rmse_m"], row["tracking_max_m"], row["max_speed"],
                row["max_accel"], row["constraint_violations"], row["mpc_failures"])
    return row, arr


def main() -> int:
    run_dir = new_run_dir("MPC_single_robot", "single")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = get_logger("fhd.mpc_single", run_dir / "logs" / "single.log")
    log_section(logger, "MPC 单机正确性测试")

    cfg = {"v_max": 1.5, "a_max": 2.0, "a_brake": 2.0}
    cases = [
        ("直线跟踪", "line", []),
        ("曲线跟踪", "curve", []),
        ("高曲率路径", "sharp", []),
        ("直角转弯", "corner", []),
        ("运行中暂停一次", "line", [(2.0, 1.0)]),
        ("连续两次暂停", "line", [(2.0, 0.6), (5.5, 0.8)]),
        ("暂停后重启", "curve", [(3.0, 1.5)]),
        ("长暂停", "line", [(1.5, 3.0)]),
    ]
    rows, traces = [], {}
    for name, kind, sched in cases:
        r, arr = run_case(kind, sched, cfg, logger, name)
        rows.append(r)
        traces[name] = {"t": arr["t"].tolist(),
                        "p": arr["p"].tolist(),
                        "v": arr["v"].tolist(),
                        "mode": [str(m) for m in arr["mode"]]}

    # 饱和测试：给一个不可能达到的参考速度
    wp = make_path("line")
    path = TimeParameterizedPath(wp, np.zeros(len(wp)),
                                 TimingParams(v_max=10.0, a_acc=10.0, a_brake=10.0))
    di = DoubleIntegrator(DoubleIntegratorParams(dt=0.05, dt_ctrl=0.1, v_max=1.5, a_max=2.0))
    ex = SingleRobotExecutor(path, di, MPCParams(horizon=18, v_max=1.5, a_max=2.0), 0.1)
    ex.reset()
    sat = {"max_speed": 0.0, "max_accel": 0.0, "viol": 0, "fails": 0}
    while ex.t < 30 and not ex.done:
        ex.step()
        if ex.log.v:
            sat["max_speed"] = max(sat["max_speed"], float(np.linalg.norm(ex.log.v[-1])))
            sat["max_accel"] = max(sat["max_accel"], float(np.linalg.norm(ex.log.a[-1])))
    sat["viol"] = int(ex.constraint_violations)
    sat["fails"] = int(ex.mpc.fail_count)
    logger.info("  饱和度测试：v_max=%.3f (上限 1.5)  a_max=%.3f (上限 2.0)  违规=%d",
                sat["max_speed"], sat["max_accel"], sat["viol"])

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for name in traces:
        arr = traces[name]
        axes[0].plot([p[0] for p in arr["p"]], [p[1] for p in arr["p"]], lw=1.2, label=name)
    axes[0].set_aspect("equal")
    axes[0].set_title("MPC 执行轨迹", fontsize=10)
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)
    for name in traces:
        arr = traces[name]
        axes[1].plot(arr["t"], np.linalg.norm(np.asarray(arr["v"]), axis=-1), lw=1.1, label=name)
    axes[1].axhline(1.5, ls="--", c="k", lw=0.8, label="v_max")
    axes[1].set_xlabel("时间 [s]")
    axes[1].set_ylabel("速度 [m/s]")
    axes[1].set_title("速度剖面（含暂停）", fontsize=10)
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "mpc_single_robot.png", dpi=130)
    plt.close(fig)

    (run_dir / "single_robot_tests.json").write_text(json.dumps(
        {"rows": rows, "saturation": sat,
         "criteria": {"v_max": 1.5, "a_max": 2.0}}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"ok": all(r["done"] for r in rows),
                      "max_rmse": max(r["tracking_rmse_m"] for r in rows),
                      "saturation": sat}, ensure_ascii=False))
    print(f"输出：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
