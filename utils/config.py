"""YAML experiment configs with single-level inheritance via a `base:` key."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a config, resolving `base:` (relative to the file) and `a.b=c` overrides."""
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_merge(base, cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    cfg.setdefault("name", path.stem)
    return cfg
