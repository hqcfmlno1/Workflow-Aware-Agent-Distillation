"""mini-SWE-agent model adapter for our FP16 server, preserving native tool history."""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
import uuid
from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict


def temporary_dns_failure(exc):
    """Only EAI_AGAIN is safe to retry before an HTTP request is transmitted."""
    pending, visited = [exc], set()
    while pending:
        error = pending.pop()
        if id(error) in visited:
            continue
        visited.add(id(error))
        if isinstance(error, socket.gaierror) and error.errno == socket.EAI_AGAIN:
            return True
        pending.extend(e for e in (error.__cause__, error.__context__) if e is not None)
    return False


def completion_request(endpoint, headers, payload, timeout, *, client=None, poll_seconds=1):
    """Use short authenticated polls; never retry an accepted generation."""
    if client is None:
        with httpx.Client(transport=httpx.HTTPTransport(retries=3)) as session:
            return completion_request(
                endpoint, headers, payload, timeout, client=session, poll_seconds=poll_seconds
            )
    deadline = None if timeout is None else time.monotonic() + timeout

    def remaining_time():
        return float("inf") if deadline is None else deadline - time.monotonic()

    base = endpoint.rstrip("/")
    known_id = uuid.uuid4().hex
    headers = {**headers, "X-Benchmark-Job-ID": known_id}
    try:
        while True:
            remaining = remaining_time()
            if remaining <= 0:
                raise TimeoutError("DNS unavailable before completion submission deadline")
            try:
                response = client.post(
                    base + "/completion-jobs",
                    headers=headers,
                    json=payload,
                    timeout=min(30, remaining),
                )
                break
            except httpx.ConnectError as exc:
                if not temporary_dns_failure(exc):
                    raise
                logging.getLogger(__name__).warning(
                    "Temporary DNS failure before submission; retaining the same job ID"
                )
                time.sleep(min(poll_seconds, max(0, remaining_time())))
    except BaseException:
        # Acceptance can precede a lost response; cancellation also works before
        # a delayed submission reaches the server, using the client-owned ID.
        with suppress(httpx.HTTPError):
            client.delete(base + "/completion-jobs/" + known_id, headers=headers, timeout=10)
        raise
    # Older local/Vast servers may only expose the synchronous OpenAI route.
    if response.status_code == 404:
        response = client.post(
            base + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=None if deadline is None else max(0.001, remaining_time()),
        )
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    job_id = response.json()["id"]
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise ValueError("Invalid completion job ID")
    try:
        poll_failures = 0
        last_progress_log = float("-inf")
        while True:
            remaining = remaining_time()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for completion; generation was not resubmitted"
                )
            try:
                response = client.get(
                    base + "/completion-jobs/" + job_id, headers=headers, timeout=min(30, remaining)
                )
                response.raise_for_status()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in (
                    502,
                    503,
                    504,
                    520,
                    521,
                    522,
                    524,
                    530,
                ):
                    raise
                dns_outage = temporary_dns_failure(exc)
                poll_failures += not dns_outage
                if poll_failures >= 5:
                    raise
                logging.getLogger(__name__).warning(
                    "Transient completion polling failure (%s); retaining same job (%s)",
                    "temporary DNS; until request deadline" if dns_outage else f"{poll_failures}/5",
                    type(exc).__name__,
                )
                time.sleep(min(poll_seconds, max(0, remaining_time())))
                continue
            poll_failures = 0
            body = response.json()
            if body.get("progress") and time.monotonic() - last_progress_log >= 10:
                logging.getLogger(__name__).info(json.dumps(
                    {"event": "completion_progress", "job_id": job_id, **body["progress"]},
                    ensure_ascii=False))
                last_progress_log = time.monotonic()
            if body["status"] == "completed":
                return body["result"]
            if body["status"] == "cancelled":
                raise TimeoutError(body.get("detail", "Generation cancelled"))
            if body["status"] == "failed":
                failure = httpx.Response(
                    body["error"]["status_code"],
                    json={"detail": body["error"]["detail"]},
                    request=response.request,
                )
                failure.raise_for_status()
            if body["status"] != "running":
                raise ValueError("Invalid completion job state")
            time.sleep(min(poll_seconds, max(0, remaining_time())))
    except BaseException:
        # A client deadline or Ctrl-C must not leave accepted GPU work running.
        with suppress(httpx.HTTPError):
            client.delete(base + "/completion-jobs/" + job_id, headers=headers, timeout=10)
        raise


class Fp16ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    api_base: str
    server_fingerprint: str
    max_new_tokens: int | None = 4096
    request_timeout_seconds: int | None = 900
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 42
    adapter_mode: Literal["loaded", "base", "lora"] = "loaded"
    adapter_scale: float = 1.0
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n"
        "{% endif %}<returncode>{{output.returncode}}</returncode>\n"
        "<output>\n{{output.output}}</output>"
    )


class Fp16Model:
    def __init__(self, **kwargs):
        self.config = Fp16ModelConfig(**kwargs)
        from .decoding import Decoding

        Decoding(self.config.temperature, self.config.top_p, self.config.top_k, self.config.seed)
        self.request_budget = None
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY to the server's BENCH_API_KEY")

    def query(self, messages, **kwargs):
        from minisweagent.exceptions import FormatError
        from minisweagent.models import GLOBAL_MODEL_STATS
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions

        if kwargs:
            raise ValueError("Per-request overrides are disabled for this fixed benchmark")
        cfg = self.config
        prepared = [{k: v for k, v in message.items() if k != "extra"} for message in messages]
        budget = self.request_budget
        timeout = cfg.request_timeout_seconds
        max_tokens = cfg.max_new_tokens
        if budget:
            timeout = min(timeout, budget["seconds"]) if timeout is not None else budget["seconds"]
            max_tokens = (
                min(max_tokens, budget["completion_tokens"])
                if max_tokens is not None
                else budget["completion_tokens"]
            )
        extra = (
            {"max_seconds": timeout, "max_total_tokens": budget["total_tokens"]} if budget else {}
        )
        body = completion_request(
            cfg.api_base,
            {
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "X-Benchmark-Fingerprint": cfg.server_fingerprint,
            },
            {
                "model": cfg.model_name,
                "messages": prepared,
                "tools": [BASH_TOOL],
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "seed": cfg.seed,
                "adapter_mode": cfg.adapter_mode,
                "adapter_scale": cfg.adapter_scale,
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                **extra,
                "stream": False,
                "parallel_tool_calls": False,
            },
            timeout,
        )
        choice = body["choices"][0]
        message = deepcopy(choice["message"])
        calls = [
            SimpleNamespace(
                id=call["id"],
                function=SimpleNamespace(**call["function"]),
            )
            for call in message.get("tool_calls") or []
        ]
        GLOBAL_MODEL_STATS.add(0)
        try:
            actions = parse_toolcall_actions(
                calls,
                format_error_template=cfg.format_error_template,
                template_kwargs={"finish_reason": choice["finish_reason"]},
            )
        except FormatError as exc:
            exc.messages[0]["extra"].update(response=body, cost=0.0)
            raise
        message["extra"] = {
            "actions": actions,
            "response": body,
            "cost": 0.0,
            "timestamp": time.time(),
        }
        return message

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages

        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs):
        return self.config.model_dump()

    def serialize(self):
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
