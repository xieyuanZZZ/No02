"""随机种子与设备/精度工具：保证同一 seed 下实验可复现。"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def set_seed(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """设置所有随机源。

    注意：CUDA 上的卷积/DDIM 采样存在非确定性算子，默认不开启全局
    deterministic（会显著降低速度）。deterministic=True 仅用于专门的
    可复现性测试。
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return {"seed": seed, "deterministic": bool(deterministic), "numpy_seed": seed % (2**32)}


def resolve_device(spec: str | None = "auto") -> torch.device:
    if spec is None or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 False")
    return device


def resolve_dtype(spec: str | torch.dtype | None) -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    if spec is None:
        return torch.float32
    try:
        return DTYPES[str(spec)]
    except KeyError as exc:
        raise ValueError(f"不支持的 dtype: {spec}") from exc


def seed_generator(seed: int | None, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device.type)
    gen.manual_seed(0 if seed is None else int(seed))
    return gen
