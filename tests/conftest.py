"""pytest 公共装置：配置加载、场景构造、独立暴力参考实现。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fhd.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "risk_only.yaml")


@pytest.fixture(scope="session")
def base_cfg():
    return load_config(ROOT / "configs" / "base.yaml")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1234)
    yield


def d_safe_of(cfg) -> float:
    return 2.0 * float(cfg["world"]["robot_radius_m"]) + float(cfg["world"]["safety_margin_m"])
