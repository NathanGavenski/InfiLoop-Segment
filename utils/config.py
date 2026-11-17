"""YAML experiment configs with single-level inheritance via a `base:` key."""

from __future__ import annotations

import copy
import os
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
        base.pop("name", None)  # the name always comes from the leaf config
        cfg = _deep_merge(base, cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    cfg.setdefault("name", path.stem)
    return _expand_env(cfg)


def _parse_scalar(raw: str):
    """YAML parsing, plus floats like 3e-4 that YAML 1.1 leaves as strings."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _expand_env(node):
    """Expand $VAR / ${VAR} in string values, so configs can point at $WORK_DIR etc."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        out = os.path.expandvars(node)
        if "$" in out:
            raise KeyError(f"unset environment variable in config value {node!r}")
        return out
    return node
