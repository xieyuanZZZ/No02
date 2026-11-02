"""汇总各阶段结果，生成 docs/final_report.md 与仓库根目录的 RESULTS.md。

用法：python scripts/make_report.py [--results results/<run_id>]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

STAGES = [
    ("S0", "环境与几何基础", "S0_environment"),
    ("S1", "首次碰撞风险 DP", "S1_risk_dp"),
    ("S2", "无 Diffusion 的风险梯度优化", "S2_gradient_optimization"),
    ("S3", "单机器人条件 Diffusion", "S3_diffusion_training"),
    ("S4", "风险引导 DDIM 接入", "S4_guided_diffusion"),
    ("S5", "完整对照实验", "S5_evaluation"),
]


def load_stage(key: str) -> dict:
    base = ROOT / "docs" / "stages" / key
    out = {"key": key, "exists": base.exists()}
    if not base.exists():
        return out
    for name in ("report.md", "stage_status.json", "config_snapshot.yaml"):
        p = base / name
        if p.exists():
            out[name] = p.read_text(encoding="utf-8")
    metrics = base / "metrics.csv"
    if metrics.exists():
        try:
            out["metrics"] = pd.read_csv(metrics)
        except Exception:  # noqa: BLE001
            out["metrics"] = None
    plots = base / "plots"
    out["plots"] = sorted(p.name for p in plots.glob("*.png")) if plots.exists() else []
    return out


def key_metrics_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "（无指标）"
    lines = ["| 分组 | 指标 | 值 |", "|---|---|---|"]
    for _, row in df.iterrows():
        val = row["value"]
        try:
            val = f"{float(val):.6g}"
        except (TypeError, ValueError):
            val = str(val)[:120]
        lines.append(f"| {row['group']} | `{row['metric']}` | {val} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总生成最终报告")
    ap.add_argument("--out", default=str(ROOT / "docs" / "final_report.md"))
    args = ap.parse_args()

    stages = {key: load_stage(key) for key, _, _ in STAGES}
    lines = [
        "# 可微首次碰撞风险引导的二维多机器人扩散轨迹规划 —— 实验总报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "> 本报告由脚本自动汇总各阶段 `docs/stages/Sx/` 下的真实运行结果生成，"
        "所有数值均来自实际运行，不含手绘或估计。",
        "",
        "## 阶段总览",
        "",
        "| 阶段 | 名称 | 状态 | 完成时间 | 耗时 | Git commit | 指标数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, title, folder in STAGES:
        st = stages[key]
        if not st.get("exists") or "stage_status.json" not in st:
            lines.append(f"| {key} | {title} | 未运行 | - | - | - | - |")
            continue
        status = json.loads(st["stage_status.json"])
        lines.append(
            f"| {key} | {title} | {status.get('status', '?')} | {status.get('finished_at', '-')} | "
            f"{status.get('duration_s', '-')} s | {str(status.get('git_commit', '-'))[:8]} | "
            f"{status.get('n_metrics', 0)} |"
        )

    for key, title, folder in STAGES:
        st = stages[key]
        lines += ["", f"## {key}：{title}", ""]
        if not st.get("exists") or "report.md" not in st:
            lines.append("该阶段尚未运行。")
            continue
        lines.append(f"运行目录：`results/{folder}/`（完整记录），"
                     f"归档：`docs/stages/{key}/`。")
        lines += ["", "### 关键指标", "", key_metrics_table(st.get("metrics")), ""]
        if st.get("plots"):
            lines += ["### 图表", ""]
            for png in st["plots"]:
                lines.append(f"![{png}](stages/{key}/plots/{png})")
            lines.append("")
        lines += ["### 阶段报告原文", "", st["report.md"], ""]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    # RESULTS.md：只放结论与关键数字
    results_lines = ["# RESULTS（自动生成）", "", "本文件由 `scripts/make_report.py` 从各阶段真实运行结果汇总。", "",
                     "| 阶段 | 状态 | 关键结论 |", "|---|---|---|"]
    for key, title, folder in STAGES:
        st = stages[key]
        if not st.get("exists") or "stage_status.json" not in st:
            results_lines.append(f"| {key} {title} | 未运行 | - |")
            continue
        status = json.loads(st["stage_status.json"])
        results_lines.append(f"| {key} {title} | {status.get('status', '?')} | "
                             f"见 `docs/stages/{key}/report.md` |")
    results_lines += ["", "## 详细结果", "", "完整内容见 [`docs/final_report.md`](docs/final_report.md)。", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(results_lines), encoding="utf-8")

    print(f"已生成 {out_path} 与 RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
