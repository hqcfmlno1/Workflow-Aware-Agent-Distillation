"""Validated decoding with an independent RNG for each completion."""

import math
from dataclasses import asdict, dataclass
from functools import cached_property


@dataclass(frozen=True)
class Decoding:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 42

    def __post_init__(self):
        for name, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1)):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not lower <= value <= upper or (name == "top_p" and value == 0):
                raise ValueError(f"Invalid {name}")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer between 0 and 2**63-1")

    @cached_property
    def processors(self):
        from transformers import (
            LogitsProcessorList,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        result = LogitsProcessorList()
        if self.temperature > 0:
            result.append(TemperatureLogitsWarper(self.temperature))
            if self.top_k:
                result.append(TopKLogitsWarper(self.top_k))
            if self.top_p < 1:
                result.append(TopPLogitsWarper(self.top_p))
        return result

    def select_token(self, logits, *, generator):
        import torch

        scores = logits.float()
        if self.temperature == 0:
            return scores.argmax(-1).item()
        # These warpers only inspect scores; no history-dependent penalties.
        scores = self.processors(None, scores)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.multinomial(probabilities, 1, generator=generator).item()

    def to_dict(self):
        return {
            **asdict(self),
            "do_sample": self.temperature > 0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "no_repeat_ngram_size": 0,
        }
