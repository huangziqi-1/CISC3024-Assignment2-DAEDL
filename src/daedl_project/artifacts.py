from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torchvision


def create_run_directory(
    root: str | Path, model_name: str, seed: int, prefix: str = "pilot"
) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir = Path(root) / f"{prefix}_{model_name}_seed{seed}_{timestamp}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, value) -> None:
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def environment_record(device_choice: dict) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": device_choice,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def save_predictions(path: str | Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows).sort_values("sample_index")
    frame.to_csv(path, index=False)
