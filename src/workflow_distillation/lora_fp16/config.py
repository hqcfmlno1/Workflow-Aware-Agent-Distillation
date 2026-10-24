"""Validated fresh-run settings; no NF4 or legacy continuation defaults."""

import math
from dataclasses import asdict, dataclass

from workflow_distillation.training.config import MODEL, REVISION

WEIGHTS = (0.0, 1.0, 0.75, 1.5, 2.0)


@dataclass(frozen=True)
class RunConfig:
    model: str = MODEL
    revision: str = REVISION
    base_dtype: str = "float16"
    quantization: str = "none"
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    epochs: int = 2
    accumulation: int = 4
    seed: int = 42
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    loss_chunk_size: int = 128
    gdn_backend: str = "fla"
    activation_storage: str = "gpu"
    gpu_budget_bytes: int = 0
    save_every: int = 10
    keep_recovery: int = 3
    minimum_vram_bytes: int = 30 * 2**30
    minimum_headroom_bytes: int = 2 * 2**30

    def __post_init__(self):
        if self.model != MODEL or self.revision != REVISION:
            raise ValueError("Use the pinned Qwen3.5-4B base")
        if self.base_dtype != "float16" or self.quantization != "none":
            raise ValueError("This experiment requires an unquantized FP16 base")
        for key in (
            "rank",
            "alpha",
            "epochs",
            "accumulation",
            "loss_chunk_size",
            "save_every",
            "keep_recovery",
            "minimum_vram_bytes",
            "minimum_headroom_bytes",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Positive integer required: {key}")
        for key in ("learning_rate", "warmup_ratio", "weight_decay", "dropout", "max_grad_norm"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")
        if (
            not 0 <= self.dropout < 1
            or not 0 <= self.warmup_ratio < 1
            or self.learning_rate <= 0
            or self.max_grad_norm <= 0
        ):
            raise ValueError("Invalid learning rate/dropout/warmup/clipping")
        if self.gdn_backend not in ("fla", "reference"):
            raise ValueError("Expected fla or reference Gated DeltaNet backend")
        if (
            self.activation_storage not in ("gpu", "cpu", "hybrid")
            or self.gpu_budget_bytes < 0
            or (self.activation_storage == "hybrid" and self.gpu_budget_bytes == 0)
        ):
            raise ValueError("Invalid activation storage")

    def to_dict(self):
        return asdict(self)
