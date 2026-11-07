"""Export only the verified adapter, never optimizer shards or an NF4 base."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from .config import MODEL, REVISION, lora_config, sha256_file

CHECKPOINT_NAME = re.compile(r"checkpoint-\d{8,}$")


def read_manifest(checkpoint):
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if (
        not CHECKPOINT_NAME.fullmatch(checkpoint.name)
        or manifest.get("format_version") != 1
        or manifest.get("complete") is not True
        or manifest.get("progress", {}).get("global_step") != int(checkpoint.name.split("-")[1])
    ):
        raise ValueError("Require a complete optimizer-boundary checkpoint")
    identity = manifest["identity"]
    if identity.get("model") != MODEL or identity.get("revision") != REVISION:
        raise ValueError("Checkpoint base model/revision differs from this experiment")
    settings = identity["settings"]["config"]
    if (settings.get("rank"), settings.get("alpha"), settings.get("dropout")) != (32, 64, 0.05):
        raise ValueError("Checkpoint LoRA hyperparameters differ")
    if sha256_file(checkpoint / "adapter.pt") != manifest["files"]["adapter.pt"]:
        raise ValueError("adapter.pt checksum mismatch")
    return manifest


def download_checkpoint(
    repo, revision, checkpoint, destination, prefix="first-run", *, repo_type="dataset"
):
    """Resolve one immutable HF revision before reading the latest pointer or weights."""
    from huggingface_hub import HfApi, hf_hub_download

    if repo_type not in ("model", "dataset"):
        raise ValueError("Checkpoint repo type must be model or dataset")
    commit = HfApi().repo_info(repo, repo_type=repo_type, revision=revision).sha

    def fetch(name):
        return Path(hf_hub_download(repo, f"{prefix}/{name}", repo_type=repo_type, revision=commit))

    if checkpoint == "latest":
        checkpoint = json.loads(fetch("latest.json").read_text())["checkpoint"]
    if not CHECKPOINT_NAME.fullmatch(checkpoint):
        raise ValueError("Invalid checkpoint name")
    target = Path(destination) / checkpoint
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "adapter.pt"):
        shutil.copyfile(fetch(f"{checkpoint}/{name}"), target / name)
    return target, {"repo": repo, "repo_type": repo_type, "revision": commit, "prefix": prefix}


def expected_adapter_shapes():
    """Construct the exact PEFT graph on meta: config only, no base weights or GPU."""
    import torch
    from peft import get_peft_model, get_peft_model_state_dict
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    with torch.device("meta"):
        model = get_peft_model(Qwen3_5ForConditionalGeneration(config), lora_config())
    return {key: tuple(value.shape) for key, value in get_peft_model_state_dict(model).items()}


def normalize_state(state, expected):
    import torch

    normalized = {}
    for key, tensor in state.items():
        if ".lora_A.default.weight" not in key and ".lora_B.default.weight" not in key:
            raise ValueError(f"Unexpected non-LoRA parameter: {key}")
        name = key.replace(".default.weight", ".weight")
        if name in normalized:
            raise ValueError(f"Duplicate adapter key: {name}")
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Expected finite FP32 adapter weights: {key}")
        normalized[name] = tensor.contiguous()
    if {key: tuple(t.shape) for key, t in normalized.items()} != expected:
        raise ValueError("Adapter names/shapes do not match the pinned PEFT model")
    return normalized


def export_adapter(checkpoint, output, *, source=None):
    import torch
    from safetensors.torch import save_file

    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new export directory: {output}")
    manifest = read_manifest(checkpoint)
    state = normalize_state(
        torch.load(checkpoint / "adapter.pt", map_location="cpu", weights_only=True),
        expected_adapter_shapes(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".adapter-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "adapter"
        stage.mkdir()
        lora_config().save_pretrained(stage)
        save_file(state, stage / "adapter_model.safetensors", metadata={"format": "pt"})
        shutil.copyfile(checkpoint / "manifest.json", stage / "training_manifest.json")
        provenance = {
            "format_version": 1,
            "base_model": MODEL,
            "base_revision": REVISION,
            "base_dtype": "float16",
            "adapter_dtype": "float32",
            "checkpoint": checkpoint.name,
            "progress": manifest["progress"],
            "source": source,
            "files": {p.name: sha256_file(p) for p in sorted(stage.iterdir())},
        }
        (stage / "export.json").write_text(json.dumps(provenance, indent=2) + "\n")
        stage.rename(output)
    return provenance


def verify_export(directory):
    directory = Path(directory)
    info = json.loads((directory / "export.json").read_text())
    if (
        info.get("format_version") != 1
        or info.get("base_model") != MODEL
        or info.get("base_revision") != REVISION
        or info.get("base_dtype") != "float16"
        or info.get("adapter_dtype") != "float32"
        or set(info.get("files", {}))
        != {
            "adapter_config.json",
            "adapter_model.safetensors",
            "training_manifest.json",
        }
    ):
        raise ValueError("Invalid FP16 benchmark export")
    for name, checksum in info["files"].items():
        if sha256_file(directory / name) != checksum:
            raise ValueError(f"Export checksum mismatch: {name}")
    return info
