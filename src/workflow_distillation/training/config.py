"""Immutable experiment settings and separately validated hardware profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\..*"
    r"(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|"
    r"in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)"
)


@dataclass(frozen=True)
class GPUProfile:
    name: str
    device_name: str
    minimum_vram_bytes: int
    per_gpu_batch_size: int
    activation_storage: str
    gpu_budget_bytes: int
    attention: str = "t4_repeated_kv"
    world_size: int = 2
    allow_serial: bool = False

    def __post_init__(self):
        if (
            not self.name
            or not self.device_name
            or type(self.minimum_vram_bytes) is not int
            or self.minimum_vram_bytes <= 0
            or type(self.per_gpu_batch_size) is not int
            or self.per_gpu_batch_size not in (1, 2)
            or type(self.world_size) is not int
            or self.world_size != 2
            or type(self.allow_serial) is not bool
            or self.attention != "t4_repeated_kv"
            or self.activation_storage not in ("cpu", "hybrid")
            or type(self.gpu_budget_bytes) is not int
            or self.gpu_budget_bytes < 0
            or (self.activation_storage == "cpu" and self.gpu_budget_bytes != 0)
            or (
                self.activation_storage == "hybrid"
                and not 0 < self.gpu_budget_bytes < self.minimum_vram_bytes
            )
            or (self.allow_serial and self.per_gpu_batch_size != 1)
        ):
            raise ValueError("Unsupported GPU execution profile")

    def validate_devices(self, devices):
        if len(devices) != self.world_size or any(
            d["name"] != self.device_name or d["total_memory"] < self.minimum_vram_bytes
            for d in devices
        ):
            raise ValueError(
                f"GPU admission rejected: expected {self.world_size} x {self.device_name}"
            )

    def to_dict(self):
        return asdict(self)


def load_profile(path):
    return GPUProfile(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Experiment:
    """The existing objective and numerical configuration; hardware lives elsewhere."""

    def effective_config(self, profile):
        return {
            "model": MODEL,
            "revision": REVISION,
            "quantization": "NF4 double quantization",
            "compute_dtype": "float16",
            "adapter_dtype": "float32",
            "embedding_and_head_dtype": "float16",
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "optimizer": "PagedAdamW32bit",
            "optimizer_state_dtype": "float32",
            "learning_rate": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "initial_gradient_scale": 8.0,
            "attention": profile.attention,
            "loss_chunk_size": 64,
            "gradient_checkpointing": "nonreentrant",
            "saved_inputs": profile.activation_storage,
            "delta_rule_block_size": 1024,
            "use_cache": False,
            "loss_weights": [0, 1, 0.75, 1.5, 2],
            "vision": "CPU offload",
            "world_size": profile.world_size,
            "per_gpu_batch_size": profile.per_gpu_batch_size,
            "global_batch_size": profile.world_size * profile.per_gpu_batch_size,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "scheduler": None,
            # Retained identity fields; the explicit distributed implementation
            # uses no reducer, buckets or automatic gradient synchronization.
            "ddp_find_unused_parameters": False,
            "ddp_broadcast_buffers": False,
            "ddp_gradient_as_bucket_view": True,
            "ddp_bucket_cap_mb": 25,
        }
