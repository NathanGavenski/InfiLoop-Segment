"""MLflow helpers. Every run logs its config, git commit, seed and data split hash.

The tracking URI comes from MLFLOW_TRACKING_URI. On the HPC it points at a file
store on shared storage (see scripts/hpc/env.sh); locally it falls back to ./mlruns.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PREFIX = "camtrap-loop"


def tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", (REPO_ROOT / "mlruns").as_uri())
    if uri.startswith("file:"):
        # MLflow >= 3.x put the file store in maintenance mode and refuses it unless opted in.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri


def git_commit() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": str(bool(run("status", "--porcelain", "--untracked-files=no")))}


def split_hash(split_ids: dict[str, Iterable[Any]]) -> str:
    """Stable hash of {split_name: ids}. Order of ids inside a split does not matter."""
    canon = {k: sorted(str(i) for i in v) for k, v in sorted(split_ids.items())}
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@contextmanager
def start_run(cfg: dict, seed: int, data_split_hash: str, run_name: str | None = None):
    """Open an MLflow run with the standard tags and the full config attached."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(f"{EXPERIMENT_PREFIX}/{cfg.get('experiment', cfg['name'])}")
    with mlflow.start_run(run_name=run_name or cfg["name"]) as run:
        mlflow.set_tags({**git_commit(), "seed": str(seed), "data_split_hash": data_split_hash,
                         "config_name": cfg["name"]})
        params = {k: str(v)[:500] for k, v in _flatten(cfg).items()}
        mlflow.log_params(params)
        mlflow.log_dict(cfg, "config.yaml")
        yield run
