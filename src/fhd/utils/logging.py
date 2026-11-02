"""日志：同时输出到 stdout 与运行目录下的日志文件。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED: set[str] = set()


def get_logger(name: str = "fhd", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    key = f"{name}:{log_file}"
    if key in _CONFIGURED:
        return logger
    logger.propagate = False
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED.add(key)
    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("=" * 78)
    logger.info(title)
    logger.info("=" * 78)
