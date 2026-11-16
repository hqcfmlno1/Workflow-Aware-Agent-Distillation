"""Explicit recipe for the WebKD / harmonic-selection CoReD adaptation."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class CoRedConfig:
    experts: int = 3
    rank: int = 32
    alpha: int = 64
    repulsion_fraction: float = 0.1
    supervision_dropout: float = 0.1
    ortho_coefficient: float = 0.01
    repulsion_coefficient: float = 1.0
    repulsion_gamma: float = 1.0
    uncertainty_coefficient: float = 0.5
    disagreement_coefficient: float = 0.5
    disagreement_epsilon: float = 1e-3
    standardization_epsilon: float = 1e-8
    kl_coefficient: float = 1.0
    ties_density: float = 0.2
    ties_scale: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    epochs: int = 2
    accumulation: int = 4
    max_grad_norm: float = 1.0
    projection_chunk: int = 16
    save_every: int = 10
    seed: int = 42
    dtype: str = "bfloat16"
    gdn_backend: str = "reference"
    target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )

    def __post_init__(self):
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Nonfinite configuration: {f.name}")
        for name in (
            "experts",
            "rank",
            "alpha",
            "epochs",
            "accumulation",
            "projection_chunk",
            "save_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Positive integer required: {name}")
        if self.experts < 2 or type(self.seed) is not int:
            raise ValueError("A council needs >=2 experts and an integer seed")
        for name in ("repulsion_fraction", "ties_density"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"Expected (0,1]: {name}")
        for name in ("supervision_dropout", "warmup_ratio"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"Expected [0,1): {name}")
        for name in (
            "ortho_coefficient",
            "repulsion_coefficient",
            "kl_coefficient",
            "uncertainty_coefficient",
            "disagreement_coefficient",
            "weight_decay",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"Nonnegative coefficient required: {name}")
        for name in (
            "repulsion_gamma",
            "learning_rate",
            "max_grad_norm",
            "ties_scale",
            "disagreement_epsilon",
            "standardization_epsilon",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Positive coefficient required: {name}")
        if self.uncertainty_coefficient + self.disagreement_coefficient > 1:
            raise ValueError("Council weight coefficients must sum to <=1")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("Use float32 (CPU) or unquantized bfloat16; FP16 is unsupported")
        if self.gdn_backend not in ("reference", "fla") or not self.target_modules:
            raise ValueError("Invalid model backend/LoRA targets")

    def to_dict(self):
        return dict(asdict(self), target_modules=list(self.target_modules))
