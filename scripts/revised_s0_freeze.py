"""第二版方案 · S0：冻结当前版本，建立新实验分支的基线。

与第一轮的 F0 相比，这里额外记录**实际生效的配置**：
  * YAML 原文（configs/*.yaml）
  * 解析并合并 extends 之后的完整配置
  * 每个训练 checkpoint 里嵌入的 config（训练时真正用的那份）
  * CLI 覆盖项（如果运行目录的 stage_status/日志里有记录）

产出：
  results/S0_baseline/<run_id>/{config_snapshot.yaml,manifest.json,metrics.csv,logs/,plots/,report.md}
  docs/revised/00_S0_freeze.md

用法：python scripts/revised_s0_freeze.py
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

import torch  # noqa: E402

from fhd.evaluation.testing import run_pytest  # noqa: E402
from fhd.utils.config import load_config  # noqa: E402
from fhd.utils.runtime import new_run_dir  # noqa: E402


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_effective_configs() -> dict:
    """YAML 原文 + 解析后的完整配置，逐项对比，暴露 extends 带来的实际差异。"""
    out = {}
    for yml in sorted((ROOT / "configs").glob("*.yaml")):
        try:
            cfg = load_config(yml)
            out[yml.name] = {
                "sha256": sha256_file(yml),
                "resolved": cfg.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            out[yml.name] = {"error": str(exc)}
    return out


def collect_checkpoints() -> list[dict]:
    rows = []
    for ck in sorted((ROOT / "results").rglob("*.pt")):
        info = {"path": str(ck.relative_to(ROOT)), "size_mb": round(ck.stat().st_size / 1024**2, 2),
                "sha256": sha256_file(ck)}
        try:
            meta = torch.load(ck, map_location="cpu", weights_only=False)
            if isinstance(meta, dict):
                info["step"] = int(meta.get("step", -1))
                info["best_val"] = float(meta.get("best_val", float("nan")))
                cfg = meta.get("config", {}) or {}
                info["trained_map_conditioning"] = cfg.get("diffusion", {}).get("map_conditioning")
                info["trained_map_res"] = cfg.get("diffusion", {}).get("map_spatial_res")
                info["trained_batch"] = cfg.get("diffusion", {}).get("batch_size")
                info["trained_steps"] = cfg.get("diffusion", {}).get("training_steps")
        except Exception:  # noqa: BLE001
            pass
        rows.append(info)
    return rows


def collect_runs() -> list[dict]:
    rows = []
    for stage in sorted((ROOT / "results").glob("*")):
        if not stage.is_dir():
            continue
        for run in sorted(stage.iterdir()):
            if not run.is_dir():
                continue
            st = run / "stage_status.json"
            meta = {}
            if st.exists():
                try:
                    meta = json.loads(st.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            files = [p for p in run.rglob("*") if p.is_file()]
            rows.append({
                "stage": stage.name, "run_id": run.name,
                "status": meta.get("status", "unknown"),
                "started_at": meta.get("started_at", ""),
                "duration_s": meta.get("duration_s", ""),
                "git_commit": str(meta.get("git_commit", ""))[:12],
                "n_files": len(files),
                "size_mb": round(sum(p.stat().st_size for p in files) / 1024**2, 2),
            })
    return rows


def collect_datasets() -> list[dict]:
    rows = []
    for meta in sorted((ROOT / "data").glob("*/meta.json")):
        rows.append({"path": str(meta.relative_to(ROOT)),
                     "sha256": sha256_file(meta),
                     "size_kb": round(meta.stat().st_size / 1024, 1)})
        for f in sorted(meta.parent.glob("*.npz")):
            rows.append({"path": str(f.relative_to(ROOT)),
                         "sha256": sha256_file(f),
                         "size_kb": round(f.stat().st_size / 1024, 1)})
    return rows


def main() -> int:
    run_dir = new_run_dir("S0_baseline", "revised_s0")
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    print(f"[S0] 运行目录 {run_dir}")

    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = git("status", "--porcelain")
    tags = git("tag", "--list").splitlines()

    configs = collect_effective_configs()
    ckpts = collect_checkpoints()
    runs = collect_runs()
    datasets = collect_datasets()

    print("[S0] 运行全量测试 ...")
    tests = run_pytest(["tests"])

    manifest = {
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": {"head": head, "branch": branch, "dirty": bool(dirty), "tags": tags},
        "system": {"python": sys.version.split()[0], "torch": torch.__version__,
                   "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "platform": platform.platform()},
        "tests": {k: v for k, v in tests.items() if k != "output_tail"},
        "counts": {"runs": len(runs), "checkpoints": len(ckpts), "datasets": len(datasets),
                   "configs": len(configs)},
        "effective_configs": configs,
        "checkpoints": ckpts,
        "datasets": datasets,
        "runs": runs,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    # 当前主配置的解析结果，作为 config_snapshot
    main_cfg = configs.get("diffusion_train.yaml", {}).get("resolved", {})
    (run_dir / "config_snapshot.yaml").write_text(
        json.dumps(main_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "logs" / "pytest.txt").write_text(
        json.dumps({k: v for k, v in tests.items() if k != "output_tail"}, ensure_ascii=False,
                   indent=2) + "\n\n" + tests.get("output_tail", ""), encoding="utf-8")

    for name, rows in (("runs", runs), ("checkpoints", ckpts), ("datasets", datasets)):
        p = run_dir / f"{name}.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    lines = [
        "# S0 旧版本冻结报告（第二版方案）",
        "",
        f"- 冻结时间：{manifest['frozen_at']}",
        f"- 分支：`{branch}`  HEAD：`{head}`",
        f"- 工作区干净：**{not bool(dirty)}**",
        f"- 已有 tag：{', '.join(tags)}",
        "",
        "## 环境",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| Python | {manifest['system']['python']} |",
        f"| PyTorch | {manifest['system']['torch']} |",
        f"| CUDA | {manifest['system']['cuda']} |",
        f"| GPU | {manifest['system']['gpu']} |",
        "",
        "## 修复前测试状态",
        "",
        f"- 通过 {tests['passed']} / 失败 {tests['failed']} / 错误 {tests['error']} / 跳过 {tests['skipped']}",
        "",
        "## 已冻结的资产",
        "",
        f"- 实验运行：**{len(runs)}** 个",
        f"- 模型检查点：**{len(ckpts)}** 个（含 SHA256）",
        f"- 数据集文件：**{len(datasets)}** 个（含 SHA256）",
        f"- 配置：**{len(configs)}** 份（YAML 原文 + 解析后的完整配置）",
        "",
        "## 实际生效配置的核对（YAML vs 训练时）",
        "",
        "| checkpoint | 步数 | map_conditioning | map_res | batch | 训练步数设置 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for c in ckpts:
        lines.append(f"| `{c['path']}` | {c.get('step', '')} | "
                     f"{c.get('trained_map_conditioning', '')} | {c.get('trained_map_res', '')} | "
                     f"{c.get('trained_batch', '')} | {c.get('trained_steps', '')} |")

    lines += ["", "## 配置解析差异（extends 展开后）", "",
              "| 配置文件 | 解析后关键项 |", "|---|---|"]
    for name, info in configs.items():
        r = info.get("resolved", {})
        if not r:
            lines.append(f"| `{name}` | 解析失败：{info.get('error', '')} |")
            continue
        d = r.get("diffusion", {})
        b = r.get("benchmark", {})
        lines.append(f"| `{name}` | map_conditioning={d.get('map_conditioning', '-')}, "
                     f"map_res={d.get('map_spatial_res', '-')}, "
                     f"training_steps={d.get('training_steps', '-')}, "
                     f"batch={d.get('batch_size', '-')}, "
                     f"grid={r.get('world', {}).get('grid_size', '-')}, "
                     f"benchmark.methods={len(b.get('methods', [])) if isinstance(b.get('methods'), list) else '-'} |")

    lines += ["", "> 注：`benchmark.methods` 这类项由 YAML 继承得到，实际运行时可能被 CLI 覆盖；"
              "本表只能证明「YAML 解析后」的值，CLI 覆盖需在运行目录的 manifest 里记录（本轮已在新脚本中强制写入）。", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    (run_dir / "stage_status.json").write_text(json.dumps({
        "stage": "S0_baseline", "status": "completed",
        "git_commit": head, "n_runs": len(runs), "n_checkpoints": len(ckpts),
        "tests": manifest["tests"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    docs = ROOT / "docs" / "revised"
    docs.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(run_dir / "report.md", docs / "00_S0_freeze.md")

    print(f"[S0] 完成：runs={len(runs)} ckpts={len(ckpts)} datasets={len(datasets)}")
    print(f"[S0] 报告：{run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
