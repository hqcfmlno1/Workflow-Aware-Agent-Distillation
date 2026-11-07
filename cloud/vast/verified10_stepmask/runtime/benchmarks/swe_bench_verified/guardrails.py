"""Task budgets and stopping-only loop detection, without changing model decoding."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from minisweagent.exceptions import LimitsExceeded
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent


@dataclass(frozen=True)
class RunLimits:
    task_seconds: int = 1800
    max_completion_tokens: int = 16000
    max_total_tokens: int = 200000
    max_calls: int = 60
    repeat_limit: int = 3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (2 if name == "repeat_limit" else 1):
                raise ValueError(f"Invalid benchmark limit: {name}")

    def to_dict(self):
        return asdict(self)


def quota_stop(reason):
    return LimitsExceeded(
        {"role": "exit", "content": reason, "extra": {"exit_status": reason, "submission": ""}}
    )


def usage(messages):
    values = [m.get("extra", {}).get("response", {}).get("usage", {}) for m in messages]
    return {
        key: sum(value.get(key, 0) for value in values)
        for key in ("completion_tokens", "total_tokens")
    }


def tool_cycles(messages):
    cycles = []
    current = None
    for message in messages:
        if message.get("role") == "assistant":
            if current is not None and current[1]:
                cycles.append(current)
            actions = message.get("extra", {}).get("actions", [])
            current = ([action.get("command") for action in actions], []) if actions else None
        elif message.get("role") == "tool" and current is not None:
            current[1].append(message.get("content", ""))
    if current is not None and current[1]:
        cycles.append(current)
    return cycles


class BoundedAgent(ProgressTrackingAgent):
    def __init__(self, *args, limits, **kwargs):
        super().__init__(*args, **kwargs)
        self.limits = limits
        self.started = time.monotonic()

    def query(self):
        budget = self.limits
        remaining = budget.task_seconds - (time.monotonic() - self.started)
        totals = usage(self.messages)
        if remaining <= 0:
            raise quota_stop("TaskTimeLimit")
        if totals["completion_tokens"] >= budget.max_completion_tokens:
            raise quota_stop("CompletionTokenLimit")
        if totals["total_tokens"] >= budget.max_total_tokens:
            raise quota_stop("TotalTokenLimit")
        responses = sum("response" in m.get("extra", {}) for m in self.messages)
        if responses >= budget.max_calls:
            raise quota_stop("ModelCallLimit")
        cycles = tool_cycles(self.messages)
        if len(cycles) >= budget.repeat_limit and all(
            cycle == cycles[-1] for cycle in cycles[-budget.repeat_limit :]
        ):
            raise quota_stop("LoopDetected")
        # Only the HTTP model needs per-request caps. Scripted/offline models use
        # the same preflight budget checks without an HTTP transport.
        from .client import Fp16Model

        if isinstance(self.model, Fp16Model):
            self.model.request_budget = {
                "seconds": remaining,
                "completion_tokens": budget.max_completion_tokens - totals["completion_tokens"],
                "total_tokens": budget.max_total_tokens - totals["total_tokens"],
            }
        try:
            return super().query()
        except Exception as exc:
            import httpx

            if (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code == 400
                and "Total token budget exhausted" in exc.response.text
            ):
                raise quota_stop("TotalTokenLimit") from exc
            raise

    def serialize(self, *extra_dicts):
        return super().serialize(
            {"info": {"limits": self.limits.to_dict(), "token_usage": usage(self.messages)}},
            *extra_dicts,
        )
