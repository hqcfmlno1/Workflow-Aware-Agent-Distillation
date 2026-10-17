"""Inspect the remote Kaggle runtime without loading a model or using local GPUs."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

if not Path("/kaggle/working").exists():
    raise SystemExit("This probe must run remotely on Kaggle.")

import torch

report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpus": [
        {
            "name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            "capability": torch.cuda.get_device_capability(i),
        }
        for i in range(torch.cuda.device_count())
    ],
    "packages": {},
    "disk_free_bytes": shutil.disk_usage("/kaggle/working").free,
    "cpu_count": os.cpu_count(),
}
for name in ("transformers", "peft", "bitsandbytes", "accelerate", "triton"):
    try:
        report["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report["packages"][name] = None
if shutil.which("nvidia-smi"):
    report["nvidia_smi"] = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False
    ).stdout
target = Path("outputs/training/gpu_probe.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
if not report["gpu_count"]:
    raise SystemExit("Kaggle supplied no GPU.")
