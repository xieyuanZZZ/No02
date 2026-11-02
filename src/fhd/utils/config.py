"""配置系统：YAML 为唯一参数来源，支持 extends 继承与命令行覆盖，并提供配置快照。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """支持属性访问的嵌套字典。"""

    def __init__(self, data: dict | None = None):
        super().__init__(data or {})

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - 明确报错
            raise AttributeError(f"配置项不存在: {name}") from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict:
        """递归转回普通 dict（便于 yaml/json 序列化）。"""

        def _plain(obj):
            if isinstance(obj, dict):
                return {k: _plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_plain(v) for v in obj]
            return obj

        return _plain(self)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")
    return data


def _resolve_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    if path in _seen:
        raise ValueError(f"配置继承出现循环: {path}")
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    parent_path = (path.parent / parent).resolve()
    base = _resolve_extends(parent_path, (*_seen, path))
    return deep_merge(base, data)


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并；override 中的标量与列表直接替换 base。"""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """加载配置：解析 extends 链，再应用覆盖项。"""
    path = Path(path).resolve()
    data = _resolve_extends(path)
    if overrides:
        data = deep_merge(data, overrides)
    cfg = Config(data)
    cfg["_config_path"] = str(path)
    return cfg


def parse_overrides(pairs: list[str]) -> dict:
    """把 "a.b=c" 形式的命令行覆盖解析成嵌套 dict（值按 YAML 解析类型）。"""
    result: dict = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"覆盖项格式应为 key=value: {item}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
