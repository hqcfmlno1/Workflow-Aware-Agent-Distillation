"""Explicit inference budget; never inherit the training quantization settings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from workflow_distillation.training.config import LORA_TARGETS, MODEL, REVISION

MINI_VERSION = "2.4.6"
DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class BenchConfig:
    model_name: str = "qwen35-4b-fp16"
    base_dtype: str = "float16"
    max_context_tokens: int | None = 32768
    max_new_tokens: int | None = 4096
    prefill_chunk_tokens: int = 512
    prefill_mode: str = "full"
    prefill_validation: str = "strict"
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 42
    reasoning_loop_guard: bool = False
    split_at_layer: int | None = 16
    enable_thinking: bool = True
    agent_step_limit: int = 250
    request_timeout_seconds: int | None = 900

    def __post_init__(self):
        from .decoding import Decoding

        Decoding(self.temperature, self.top_p, self.top_k, self.seed)
        if self.prefill_mode not in ("auto", "full", "chunked"):
            raise ValueError("prefill_mode must be auto, full or chunked")
        if self.prefill_validation not in ("strict", "report"):
            raise ValueError("prefill_validation must be strict or report")
        if self.prefill_validation == "report" and self.prefill_mode != "chunked":
            raise ValueError("Report-only validation requires explicit chunked prefill")
        if type(self.reasoning_loop_guard) is not bool:
            raise ValueError("reasoning_loop_guard must be boolean")
        if self.base_dtype not in ("float16", "bfloat16", "nf4"):
            raise ValueError("base_dtype must be float16 or nf4")
        for name in ("max_context_tokens", "max_new_tokens", "request_timeout_seconds"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")
        for name in ("prefill_chunk_tokens",):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.agent_step_limit) is not int or self.agent_step_limit < 0:
            raise ValueError("agent_step_limit must be zero (unlimited) or a positive integer")
        if (
            self.max_new_tokens is not None
            and self.max_context_tokens is not None
            and self.max_new_tokens >= self.max_context_tokens
        ):
            raise ValueError("Output budget must leave room for the prompt")
        if self.split_at_layer is not None and (
            type(self.split_at_layer) is not int or not 0 < self.split_at_layer < 32
        ):
            raise ValueError("split_at_layer must be null (one GPU) or between 1 and 31")
        if type(self.enable_thinking) is not bool or not self.model_name:
            raise ValueError("Invalid thinking mode or model name")

    def to_dict(self):
        return asdict(self)


def load_config(path=DEFAULT_CONFIG):
    return BenchConfig(**json.loads(Path(path).read_text()))


def lora_config():
    from peft import LoraConfig

    return LoraConfig(
        base_model_name_or_path=MODEL,
        revision=REVISION,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
        inference_mode=True,
    )
