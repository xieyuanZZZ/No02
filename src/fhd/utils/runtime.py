"""运行环境信息与实验目录管理：每个 run 独立目录，不覆盖旧结果。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def git_commit(short: bool = False) -> str:
    """当前 Git commit；无仓库或未提交时返回 'unknown'。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        value = out.stdout.strip()
        return value or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def system_info() -> dict[str, Any]:
    """记录 Python/PyTorch/CUDA/硬件信息（文档 §0.4 强制要求）。"""
    info: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": torch.get_num_threads(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / 1024**3, 2),
                "gpu_index": dev,
                "gpu_capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def new_run_dir(stage: str, name: str, root: str | Path | None = None) -> Path:
    """创建 results/<stage>/<时间戳>_<name>_<短id>/ 运行目录并填充子目录。"""
    root = Path(root) if root else PROJECT_ROOT / "results"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{name}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(root) / stage / run_id
    for sub in ("plots", "logs", "plans", "executions", "tables"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "run_id.txt").write_text(run_id, encoding="utf-8")
    return run_dir


class RunRecorder:
    """一个 run 的元数据记录器：config 快照、系统信息、耗时、状态。"""

    def __init__(self, run_dir: Path, config: dict | None = None, stage: str | None = None):
        self.run_dir = Path(run_dir)
        self.stage = stage
        self.started = time.time()
        self.status: dict[str, Any] = {
            "run_id": self.run_dir.name,
            "stage": stage,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "running",
            "events": [],
        }
        self.info = system_info()
        _write_json(self.run_dir / "system_info.json", self.info)
        _write_text(self.run_dir / "git_commit.txt", self.info["git_commit"] + "\n")
        if config is not None:
            _write_yaml(self.run_dir / "config_snapshot.yaml", config)
        self._flush_status()

    # --- 事件与指标 -------------------------------------------------
    def event(self, name: str, **details: Any) -> None:
        self.status["events"].append(
            {"t": round(time.time() - self.started, 3), "name": name, **details}
        )
        self._flush_status()

    def set(self, key: str, value: Any) -> None:
        self.status[key] = value
        self._flush_status()

    def finish(self, status: str = "completed", **summary: Any) -> None:
        self.status.update(summary)
        self.status["status"] = status
        self.status["duration_s"] = round(time.time() - self.started, 3)
        self.status["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self._flush_status()

    def _flush_status(self) -> None:
        _write_json(self.run_dir / "stage_status.json", self.status)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_yaml(path: Path, config: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)
