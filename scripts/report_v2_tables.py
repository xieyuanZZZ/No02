"""把 Online_FirstHit_v2 的 summary.json 汇总成报告用的 Markdown 表格。

只读结果文件，不重算任何指标；用于避免手工抄数出错。
用法：
  python scripts/report_v2_tables.py --phase1 results/.../02_mpc_benchmark \
      --baselines results/.../03_risk_baselines --m3 results/.../04_dynamic_risk_m3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def phase1_table(s: dict) -> str:
    ps = s["ps"]
    lines = ["| 组 | 计划 | p | D1 碰撞 | D2 碰撞 | D2 安全完成 | 超时 | 不可行停车 | MPC 失败 | 约束违反 | 求解均值/P95 (ms) | 跟踪 RMSE (m) | 最小间距 (m) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in ("N2", "N4"):
        for plan in ("orig", "w2plus"):
            for p in ps:
                c = s["by_group"][g][plan][str(p)]
                lines.append(
                    f"| {g} | {plan} | {p:g} | {_f(c['d1_collision_rate'])} | "
                    f"{_f(c['d2_collision_rate'])} | {_f(c['d2_safe_rate'])} | "
                    f"{_f(c['d2_timeout_rate'])} | {c['d2_infeasible_stops']} | "
                    f"{c['d2_mpc_failures']} | {c['d2_constraint_violations']} | "
                    f"{_f(c['d2_solve_mean_ms'],2)} / {_f(c['d2_solve_p95_ms'],2)} | "
                    f"{_f(c['d2_tracking_rmse'],4)} | {_f(c['d2_min_pair_distance'])} |")
    return "\n".join(lines)


def paired_table(s: dict) -> str:
    lines = ["| 组 | p | 任务数 | W2+ − 原始（D2 碰撞率） | 95% 区间 |",
             "|---|---:|---:|---:|---|"]
    for g in ("N2", "N4"):
        for p, v in s["paired_w2plus_minus_orig"][g].items():
            lines.append(f"| {g} | {p} | {v['n_tasks']} | {_f(v['mean_diff'])} | "
                         f"[{_f(v['ci_lo'])}, {_f(v['ci_hi'])}] |")
    return "\n".join(lines)


def m1m2_table(s: dict) -> str:
    lines = ["| 模型 | MAE | RMSE | bias | Spearman | Brier | n |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in ("m0", "m1", "m2"):
        st = s["overall"][m]
        lines.append(f"| {m.upper()} | {_f(st['mae'],4)} | {_f(st['rmse'],4)} | "
                     f"{_f(st['bias'],4)} | {_f(st['spearman'],3)} | "
                     f"{_f(st['brier'],4)} | {st['n']} |")
    return "\n".join(lines)


def m1m2_groups(s: dict) -> str:
    lines = ["| 分组 | M0 MAE / ρ | M1 MAE / ρ | M2 MAE / ρ |",
             "|---|---:|---:|---:|"]
    for g in ("N2", "N4"):
        d = s["by_group"][g]
        lines.append(f"| {g} | {_f(d['m0']['mae'],4)} / {_f(d['m0']['spearman'],3)} | "
                     f"{_f(d['m1']['mae'],4)} / {_f(d['m1']['spearman'],3)} | "
                     f"{_f(d['m2']['mae'],4)} / {_f(d['m2']['spearman'],3)} |")
    for p, d in s["by_p"].items():
        lines.append(f"| p={p} | {_f(d['m0']['mae'],4)} / {_f(d['m0']['spearman'],3)} | "
                     f"{_f(d['m1']['mae'],4)} / {_f(d['m1']['spearman'],3)} | "
                     f"{_f(d['m2']['mae'],4)} / {_f(d['m2']['spearman'],3)} |")
    return "\n".join(lines)


def m3_table(s: dict) -> str:
    models = ["m0", "m1", "m2", "m3"]
    names = {"m0": "M0", "m1": "M1", "m2": "M2", "m3": "M3"}
    lines = ["| 模型 | MAE | RMSE | bias | Spearman | Brier |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        st = s["overall"][m]
        lines.append(f"| {names[m]} | {_f(st['mae'],4)} | {_f(st['rmse'],4)} | "
                     f"{_f(st['bias'],4)} | {_f(st['spearman'],3)} | "
                     f"{_f(st['brier'],4)} |")
    return "\n".join(lines)


def m3_groups(s: dict) -> str:
    models = ["m0", "m1", "m2", "m3"]
    lines = ["| 分组 | " + " | ".join(m.upper() + " MAE/ρ" for m in models) + " |",
             "|---|" + "---:|" * len(models)]
    for g in ("N2", "N4"):
        d = s["by_group"][g]
        lines.append(f"| {g} | " + " | ".join(
            f"{_f(d[m]['mae'],4)} / {_f(d[m]['spearman'],3)}" for m in models) + " |")
    for p, d in s["by_p"].items():
        lines.append(f"| p={p} | " + " | ".join(
            f"{_f(d[m]['mae'],4)} / {_f(d[m]['spearman'],3)}" for m in models) + " |")
    for plan, d in s["by_plan"].items():
        lines.append(f"| {plan} | " + " | ".join(
            f"{_f(d[m]['mae'],4)} / {_f(d[m]['spearman'],3)}" for m in models) + " |")
    for geo, d in s["by_geometry"].items():
        lines.append(f"| {geo} | " + " | ".join(
            f"{_f(d[m]['mae'],4)} / {_f(d[m]['spearman'],3)}" for m in models) + " |")
    return "\n".join(lines)


def m3_bootstrap(s: dict) -> str:
    lines = ["| 模型 | 任务级 MAE 差（相对 M0） | 95% bootstrap 区间 | 任务数 |",
             "|---|---:|---|---:|"]
    for m, d in s["delta_mae_vs_m0"].items():
        lines.append(f"| {m} | {_f(d.get('mean'),4)} | "
                     f"[{_f(d.get('ci_lo'),4)}, {_f(d.get('ci_hi'),4)}] | "
                     f"{d.get('n_tasks', '—')} |")
    return "\n".join(lines)


def m3_ablation(s: dict) -> str:
    lines = ["| 配置 | MAE | RMSE | bias | Spearman |",
             "|---|---:|---:|---:|---:|"]
    for m in ("m3", "m3_no_braking", "m3_no_modes", "m3_no_speed"):
        st = s["overall"][m]
        lines.append(f"| {m} | {_f(st['mae'],4)} | {_f(st['rmse'],4)} | "
                     f"{_f(st['bias'],4)} | {_f(st['spearman'],3)} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="results/Online_FirstHit_v2/02_mpc_benchmark")
    ap.add_argument("--baselines", default="results/Online_FirstHit_v2/03_risk_baselines")
    ap.add_argument("--m3", default="results/Online_FirstHit_v2/04_dynamic_risk_m3")
    args = ap.parse_args()
    p1 = _load(Path(args.phase1) / "summary.json")
    m12 = _load(Path(args.baselines) / "summary.json")
    m3 = _load(Path(args.m3) / "summary.json")
    print("## 阶段一：D1/D2 分组结果\n")
    print(phase1_table(p1))
    print("\n## 阶段一：W2+ 配对差\n")
    print(paired_table(p1))
    print("\n## M0/M1/M2（无泄漏）\n")
    print(m1m2_table(m12))
    print()
    print(m1m2_groups(m12))
    print("\n## M0–M3 冻结盲测\n")
    print(m3_table(m3))
    print()
    print(m3_groups(m3))
    print("\n## 任务级 bootstrap\n")
    print(m3_bootstrap(m3))
    print("\n## M3 消融\n")
    print(m3_ablation(m3))
    print("\n## Top-K / 团队 / 开销\n")
    print(json.dumps({"top1": m3["top1_hit_rate"], "top3": m3["top3_hit_rate"],
                      "team": m3["team"], "runtime": m3["runtime"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
