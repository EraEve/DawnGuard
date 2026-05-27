"""YAML-backed configuration loader with lightweight hot-reload support."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

from exceptions import ConfigError

T = TypeVar("T")


def _filter_dataclass_payload(config_cls: Type[T], payload: Dict[str, Any]) -> Dict[str, Any]:
    valid = {field.name for field in fields(config_cls)}
    return {key: value for key, value in payload.items() if key in valid}


def load_yaml_config(config_cls: Type[T], path: str | Path) -> T:
    """Load a dataclass configuration from YAML."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise ConfigError(f"Configuration file not found: {yaml_path}", code="CONFIG_FILE_MISSING")
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ConfigError("YAML root must be a mapping.", code="CONFIG_YAML_ROOT_INVALID")
        return config_cls(**_filter_dataclass_payload(config_cls, payload))
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError("Failed to parse YAML configuration.", code="CONFIG_YAML_PARSE_ERROR", cause=exc)


class HotReloadConfig:
    """Poll-based config loader for deployments that need YAML hot updates."""

    def __init__(self, config_cls: Type[T], path: str | Path) -> None:
        self.config_cls = config_cls
        self.path = Path(path)
        self._last_mtime: Optional[float] = None
        self._cached: Optional[T] = None

    def get(self) -> T:
        """Return the cached config or reload it when the YAML mtime changes."""
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if self._cached is None or mtime != self._last_mtime:
            self._cached = load_yaml_config(self.config_cls, self.path)
            self._last_mtime = mtime
        return self._cached
