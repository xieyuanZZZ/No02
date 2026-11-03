"""F0：冻结修复前的基线。

产出：
  docs/audit_fix/00_baseline_inventory.md
  docs/audit_fix/manifest_baseline.json
  docs/audit_fix/tables/baseline_{runs,checkpoints,artifacts}.csv
  docs/audit_fix/tables/pytest_baseline.txt

用法：python scripts/audit_f0_freeze.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhd.evaluation.testing import run_pytest  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "docs" / "audit_fix"
TABLES = OUT / "tables"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_runs() -> list[dict]:
    rows = []
    for stage_dir in sorted((ROOT / "results").glob("*")):
        if not stage_dir.is_dir():
            continue
        for run in sorted(stage_dir.iterdir()):
            if not run.is_dir():
                continue
            status_file = run / "stage_status.json"
            status = {}
            if status_file.exists():
                try:
                    status = json.loads(status_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    status = {}
            files = [p for p in run.rglob("*") if p.is_file()]
            size = sum(p.stat().st_size for p in files)
            rows.append({
                "stage": stage_dir.name,
                "run_id": run.name,
                "status": status.get("status", "unknown"),
                "started_at": status.get("started_at", ""),
                "duration_s": status.get("duration_s", ""),
                "git_commit": status.get("git_commit", ""),
                "n_files": len(files),
                "size_mb": round(size / 1024**2, 2),
                "has_report": (run / "report.md").exists(),
                "has_metrics": (run / "metrics.csv").exists(),
            })
    return rows


def collect_checkpoints() -> list[dict]:
    rows = []
    for ck in sorted((ROOT / "results").rglob("*.pt")):
        try:
            meta = torch.load(ck, map_location="cpu", weights_only=False)
            step = int(meta.get("step", -1)) if isinstance(meta, dict) else -1
            best_val = float(meta.get("best_val", float("nan"))) if isinstance(meta, dict) else float("nan")
            cfg = meta.get("config", {}) if isinstance(meta, dict) else {}
            map_res = cfg.get("diffusion", {}).get("map_spatial_res", None) if isinstance(cfg, dict) else None
        except Exception:  # noqa: BLE001
            step, best_val, map_res = -1, float("nan"), None
        rows.append({
            "path": str(ck.relative_to(ROOT)),
            "size_mb": round(ck.stat().st_size / 1024**2, 2),
            "sha256": sha256_file(ck),
            "step": step,
            "best_val": best_val,
            "map_spatial_res": map_res,
        })
    return rows


def collect_artifacts() -> list[dict]:
    """数据、配置与依赖的哈希（用于复现）。"""
    targets = []
    for pattern in ("data/**/*", "configs/*.yaml", "pyproject.toml", "requirements.txt", "uv.lock"):
        for p in ROOT.glob(pattern):
            if p.is_file():
                targets.append(p)
    rows = []
    for p in sorted(set(targets)):
        rows.append({
            "path": str(p.relative_to(ROOT)),
            "size_kb": round(p.stat().st_size / 1024, 1),
            "sha256": sha256_file(p),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status_short = git("status", "--porcelain")
    tags = git("tag", "--list")
    remote = git("remote", "-v")

    runs = collect_runs()
    ckpts = collect_checkpoints()
    artifacts = collect_artifacts()

    write_csv(TABLES / "baseline_runs.csv", runs)
    write_csv(TABLES / "baseline_checkpoints.csv", ckpts)
    write_csv(TABLES / "baseline_artifacts.csv", artifacts)

    print("[F0] 运行全量测试 ...")
    tests = run_pytest(["tests"])
    (TABLES / "pytest_baseline.txt").write_text(
        json.dumps({k: v for k, v in tests.items() if k != "output_tail"}, ensure_ascii=False, indent=2)
        + "\n\n--- 输出尾部 ---\n" + tests.get("output_tail", ""),
        encoding="utf-8",
    )

    manifest = {
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": {"head": head, "branch": branch, "dirty": bool(status_short),
                "tags": tags.splitlines(), "remote": remote.splitlines()},
        "system": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "tests": {k: v for k, v in tests.items() if k != "output_tail"},
        "counts": {"runs": len(runs), "checkpoints": len(ckpts), "artifacts": len(artifacts)},
        "checkpoints": ckpts,
        "artifacts": artifacts,
    }
    (OUT / "manifest_baseline.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 人类可读清单 ----
    lines = [
        "# 00 基线清单（修复前冻结）",
        "",
        f"冻结时间：{manifest['frozen_at']}",
        "",
        "## 1. 代码状态",
        "",
        f"- 分支：`{branch}`",
        f"- HEAD：`{head}`",
        f"- 工作区有未提交改动：**{manifest['git']['dirty']}**",
        f"- 已有 tag：{', '.join(manifest['git']['tags']) or '（无）'}",
        f"- 远程：{'; '.join(manifest['git']['remote']) or '（无）'}",
        "",
        "## 2. 运行环境",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| Python | {manifest['system']['python']} |",
        f"| PyTorch | {manifest['system']['torch']} |",
        f"| CUDA | {manifest['system']['cuda']} |",
        f"| GPU | {manifest['system']['gpu']} |",
        f"| 平台 | {manifest['system']['platform']} |",
        "",
        "## 3. 修复前测试状态",
        "",
        f"- 通过 **{tests['passed']}**、失败 {tests['failed']}、错误 {tests['error']}、跳过 {tests['skipped']}",
        f"- 退出码：{tests['returncode']}",
        f"- 明细：`tables/pytest_baseline.txt`",
        "",
        "## 4. 实验运行记录（不可覆盖）",
        "",
        f"共 **{len(runs)}** 次运行。",
        "",
        "| 阶段 | run_id | 状态 | 文件数 | 体积 MB |",
        "|---|---|---|---:|---:|",
    ]
    for r in runs:
        lines.append(f"| {r['stage']} | `{r['run_id']}` | {r['status']} | {r['n_files']} | {r['size_mb']} |")

    lines += ["", "## 5. 模型检查点（本机，未入库）", "",
              "| 路径 | 步数 | best_val | MB | SHA256（前 16） |", "|---|---:|---:|---:|---|"]
    for c in ckpts:
        lines.append(f"| `{c['path']}` | {c['step']} | {c['best_val']:.5f} | {c['size_mb']} | "
                     f"`{c['sha256'][:16]}` |")

    lines += ["", "## 6. 数据与配置哈希", "", "| 文件 | KB | SHA256（前 16） |", "|---|---:|---|"]
    for a in artifacts:
        lines.append(f"| `{a['path']}` | {a['size_kb']} | `{a['sha256'][:16]}` |")

    lines += [
        "",
        "## 7. 关键基线数字（修复前，保留原值）",
        "",
        "| 项 | 值 | 出处 |",
        "|---|---|---|",
        "| S1 DP vs MC 最大绝对误差 | **0.00673775** | `results/S1_risk_dp/*/metrics.csv` |",
        "| S1 梯度有限差分最大相对误差 | 3.53e-7 | 同上 |",
        "| S2 交叉场景硬风险 | 0.1788 → 2.15e-6（A3） | `results/S2_gradient_optimization/*/tables/pareto_points.csv` |",
        "| S2 窄门口硬风险 | 1.0 → 1.0（饱和） | 同上 |",
        "| S3 主模型单机成功率 | 59.4%（489 条测试集，200 任务×6） | run `032008_s3_sample_0b5ace`，ckpt `383287` |",
        "| S3 归档报告成功率 | 60.75%（被 16×16 模型覆盖） | `docs/stages/S3/report.md`，ckpt `4b11c4` |",
        "| S5 规划有效率 | 11/36（7 个方法完全相同） | `results/S5_evaluation/*/tables/plans.csv` |",
        "| S5 p=0.1 安全完成率 | B0 18.24% / B2 19.44% / B3 25.65% / A2 22.31% / A3 22.59% | 同上 |",
        "| S5 A3−B3 配对差（跨 4 个 p 汇总） | −0.0366 [−0.0963, −0.0002] | `aggregate.json` |",
        "",
        "> 这些数字与目录一律作为 `baseline_before_fix` 保留，后续修复不得覆盖。",
        "",
    ]
    (OUT / "00_baseline_inventory.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[F0] 完成：{OUT / '00_baseline_inventory.md'}")
    print(f"     运行 {len(runs)} 次，检查点 {len(ckpts)} 个，工件 {len(artifacts)} 个")
    print(f"     测试：passed={tests['passed']} failed={tests['failed']} error={tests['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
