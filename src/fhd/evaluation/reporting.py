"""阶段结果落盘：metrics.csv、plots/、report.md、stage_status.json，并同步到 docs/stages/Sx。

所有曲线的原始数值都必须先写成 CSV/JSONL 再画图（批复 §八），不允许只留 PNG。
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 中文字体：Windows 自带 Microsoft YaHei / SimHei
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def prepare_matplotlib() -> None:
    """确保 Agg 后端与中文字体生效。"""
    matplotlib.use("Agg")


class StageReport:
    """一个阶段运行的结果记录器。"""

    def __init__(self, stage: str, run_dir: Path, title: str, config: dict | None = None):
        self.stage = stage
        self.run_dir = Path(run_dir)
        self.title = title
        self.started = time.time()
        self.metrics: list[dict] = []
        self.sections: list[str] = []
        self.artifacts: list[str] = []
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "tables").mkdir(parents=True, exist_ok=True)
        if config is not None:
            import yaml

            with (self.run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)

    # --- 指标 -----------------------------------------------------------
    def metric(self, name: str, value, group: str = "main", note: str = "") -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist() if value.numel() > 1 else float(value.detach().cpu())
        self.metrics.append(
            {"group": group, "metric": name, "value": value, "note": note}
        )

    def metrics_from_dict(self, data: dict, group: str = "main") -> None:
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                self.metrics.append({"group": group, "metric": key,
                                     "value": json.dumps(value, ensure_ascii=False), "note": ""})
            else:
                self.metric(key, value, group=group)

    def table(self, name: str, rows: list[dict]) -> Path:
        """保存原始数值表为 CSV。"""
        path = self.run_dir / "tables" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            fieldnames = list(rows[0].keys())
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text("", encoding="utf-8")
        self.artifacts.append(str(path.relative_to(self.run_dir)))
        return path

    # --- 图 -------------------------------------------------------------
    def save_figure(self, fig: plt.Figure, name: str) -> Path:
        path = self.plots_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        self.artifacts.append(str(path.relative_to(self.run_dir)))
        return path

    # --- 文本 -----------------------------------------------------------
    def add_section(self, markdown: str) -> None:
        self.sections.append(markdown)

    def write_report(self, extra_header: str = "") -> Path:
        elapsed = time.time() - self.started
        lines = [
            f"# {self.title}",
            "",
            f"- 阶段：`{self.stage}`",
            f"- 运行目录：`{self.run_dir}`",
            f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"- 耗时：{elapsed:.1f} s",
            "",
        ]
        if extra_header:
            lines += [extra_header, ""]
        lines += self.sections
        lines += ["", "## 指标", "", "| 分组 | 指标 | 值 | 备注 |", "|---|---|---|---|"]
        for row in self.metrics:
            value = row["value"]
            if isinstance(value, float):
                value = f"{value:.6g}"
            lines.append(f"| {row['group']} | `{row['metric']}` | {value} | {row['note']} |")
        lines += ["", "## 产物", ""]
        for item in self.artifacts:
            lines.append(f"- `{item}`")
        report_path = self.run_dir / "report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    def write_metrics_csv(self) -> Path:
        path = self.run_dir / "metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["group", "metric", "value", "note"])
            writer.writeheader()
            for row in self.metrics:
                value = row["value"]
                if isinstance(value, float):
                    value = f"{value:.10g}"
                writer.writerow({**row, "value": value})
        return path

    def write_status(self, status: str, test_summary: dict | None = None, extra: dict | None = None) -> Path:
        from fhd.utils.runtime import git_commit

        data = {
            "stage": self.stage,
            "run_id": self.run_dir.name,
            "status": status,
            "started_at": datetime.fromtimestamp(self.started).astimezone().isoformat(timespec="seconds"),
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self.started, 1),
            "git_commit": git_commit(),
            "n_metrics": len(self.metrics),
            "tests": test_summary or {},
            "extra": extra or {},
        }
        path = self.run_dir / "stage_status.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    # --- 同步到 docs/stages --------------------------------------------
    def sync_to_docs(self) -> Path:
        """把报告、关键图、指标表复制进 docs/stages/Sx（纳入 Git）。"""
        stage_key = self.stage.split("_")[0]  # 例如 S1_risk_dp -> S1
        target = PROJECT_ROOT / "docs" / "stages" / stage_key
        target.mkdir(parents=True, exist_ok=True)
        for item in ("report.md", "metrics.csv", "stage_status.json", "config_snapshot.yaml"):
            src = self.run_dir / item
            if src.exists():
                shutil.copy2(src, target / item)
        plot_target = target / "plots"
        plot_target.mkdir(parents=True, exist_ok=True)
        for png in sorted(self.plots_dir.glob("*.png")):
            shutil.copy2(png, plot_target / png.name)
        table_target = target / "tables"
        table_target.mkdir(parents=True, exist_ok=True)
        for csv_file in sorted((self.run_dir / "tables").glob("*.csv")):
            shutil.copy2(csv_file, table_target / csv_file.name)
        return target


def update_project_progress(stages: list[dict] | None = None) -> Path:
    """维护 docs/PROJECT_PROGRESS.md：读取各阶段 stage_status.json 汇总。"""
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in ("S0", "S1", "S2", "S3", "S4", "S5"):
        status_file = docs / "stages" / key / "stage_status.json"
        if status_file.exists():
            data = json.loads(status_file.read_text(encoding="utf-8"))
            rows.append(
                "| {stage} | {status} | {t} | {dur} | {commit} | {n} |".format(
                    stage=data.get("stage", key),
                    status=data.get("status", "?"),
                    t=data.get("finished_at", "-"),
                    dur=f"{data.get('duration_s', 0)} s",
                    commit=str(data.get("git_commit", "-"))[:8],
                    n=data.get("n_metrics", 0),
                )
            )
        else:
            rows.append(f"| {key} | 未开始 | - | - | - | - |")

    extra_lines = []
    if stages:
        extra_lines = ["", "## 关键结论", ""]
        for item in stages:
            extra_lines.append(f"- **{item.get('stage')}**：{item.get('summary', '')}")

    content = [
        "# 项目进度",
        "",
        f"最后更新：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 阶段状态",
        "",
        "| 阶段 | 状态 | 完成时间 | 耗时 | Git commit | 指标数 |",
        "|---|---|---|---|---|---|",
        *rows,
        *extra_lines,
        "",
        "## 说明",
        "",
        "- 每个阶段的完整记录（配置快照、CSV 原始数据、PNG 图、报告）位于 `docs/stages/Sx/`，已纳入 Git。",
        "- 完整运行目录（含全部轨迹与检查点）保留在 `results/Sx/`，不入库。",
        "- 阶段状态由各次运行的 `stage_status.json` 自动汇总，不手工编辑。",
        "",
    ]
    path = docs / "PROJECT_PROGRESS.md"
    path.write_text("\n".join(content), encoding="utf-8")
    return path
