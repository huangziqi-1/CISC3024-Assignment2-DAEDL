#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import torchvision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/environment_check.json"))
    args = parser.parse_args()

    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    cpu_value = (torch.arange(6, dtype=torch.float32).reshape(2, 3) @ torch.ones(3, 1)).sum().item()
    record["cpu_tensor_test"] = {"passed": cpu_value == 15.0, "value": cpu_value}
    if record["mps_available"]:
        try:
            tensor = torch.arange(6, dtype=torch.float32, device="mps").reshape(2, 3)
            value = (tensor @ torch.ones(3, 1, device="mps")).sum()
            torch.mps.synchronize()
            record["mps_tensor_test"] = {"passed": value.item() == 15.0, "value": value.item()}
        except Exception as error:  # exact failure is part of the environment record
            record["mps_tensor_test"] = {"passed": False, "error": repr(error)}
    else:
        record["mps_tensor_test"] = {"passed": False, "error": "MPS runtime unavailable"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["cpu_tensor_test"]["passed"]:
        raise SystemExit("CPU tensor test failed")
    if record["mps_available"] and not record["mps_tensor_test"]["passed"]:
        raise SystemExit("MPS is available but the tensor test failed")


if __name__ == "__main__":
    main()

