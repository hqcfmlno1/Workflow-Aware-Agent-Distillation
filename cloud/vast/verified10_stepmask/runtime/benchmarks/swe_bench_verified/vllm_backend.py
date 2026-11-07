"""vLLM execution with the benchmark's original prompt, XML parser and job API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from .config import MODEL, REVISION, fingerprint, sha256_file
from .decoding import Decoding
from .export_adapter import verify_export
from .generation import GenerationControl
from .protocol import normalize_messages, parse_completion
from .serving import resolve_output_limit

logger = logging.getLogger(__name__)
VLLM_VERSION = "0.32.0"


@dataclass(frozen=True)
class VllmSettings:
    chunk_tokens: int = 2048
    gpu_memory_utilization: float = 0.88
    prefix_caching: bool = True
    enforce_eager: bool = False

    def __post_init__(self):
        if type(self.chunk_tokens) is not int or self.chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")


def engine_options(config, settings, *, has_adapter, native_context=262144):
    if config.base_dtype != "bfloat16":
        raise ValueError("This pilot requires an unquantized float16 base")
    if config.split_at_layer is not None:
        raise ValueError("This vLLM pilot uses one GPU; split_at_layer must be null")
    if config.prefill_mode != "chunked":
        raise ValueError("vLLM pilot requires explicit chunked prefill")
    if config.reasoning_loop_guard:
        raise ValueError("No reasoning guard is enabled for this pilot")
    return {
        "model": MODEL,
        "revision": REVISION,
        "tokenizer": MODEL,
        "tokenizer_revision": REVISION,
        "dtype": "bfloat16",
        "generation_config": "vllm",
        "seed": config.seed,
        "max_model_len": config.max_context_tokens or native_context,
        "max_num_seqs": 1,
        "max_num_batched_tokens": settings.chunk_tokens,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": settings.prefix_caching,
        "mamba_cache_mode": "align" if settings.prefix_caching else "none",
        "mamba_ssm_cache_dtype": "float32",
        "language_model_only": True,
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "enforce_eager": settings.enforce_eager,
        "max_cudagraph_capture_size": 1,
        "enable_lora": has_adapter,
        "max_loras": 2,
        "max_lora_rank": 32,
        "lora_dtype": "bfloat16",
        "disable_log_stats": False,
    }


async def collect_completion(engine, ids, params, lora, control, request_id):
    """Abort even during prefill, when a FINAL_ONLY iterator yields no tokens."""
    control = control or GenerationControl(None)

    async def consume():
        final = None
        async for output in engine.generate(
            {"prompt_token_ids": ids}, params, request_id=request_id, lora_request=lora
        ):
            final = output
        if final is None:
            raise RuntimeError("vLLM ended without a completion")
        return final

    control.check()
    task = asyncio.create_task(consume())
    finished = False
    try:
        while True:
            control.check()
            done, _ = await asyncio.wait({task}, timeout=0.05)
            if done:
                result = task.result()
                control.check()
                finished = True
                return result
    finally:
        if not finished:
            try:
                await engine.abort(request_id)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task


class VllmBackend:
    def __init__(self, config, adapter=None, *, settings=None):
        import torch
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import RequestOutputKind
        from vllm.v1.engine.async_llm import AsyncLLM

        if version("vllm").split("+", 1)[0] != VLLM_VERSION:
            raise RuntimeError(f"Use the reviewed vllm=={VLLM_VERSION} environment")
        self.config, self.settings = config, settings or VllmSettings()
        exported = verify_export(adapter) if adapter else None
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        native_config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
        text_config = getattr(native_config, "text_config", native_config)
        self.native_context_tokens = text_config.max_position_embeddings
        generation = GenerationConfig.from_model_config(native_config)
        eos = generation.eos_token_id
        self.eos_token_ids = sorted(
            set(eos if isinstance(eos, list) else [eos]).union({self.tokenizer.eos_token_id})
            - {None}
        )
        self.sampling_class, self.final_output_kind = SamplingParams, RequestOutputKind.FINAL_ONLY
        self.lora_request = (
            LoRARequest(exported["checkpoint"], 1, str(Path(adapter).resolve()))
            if adapter
            else None
        )
        options = engine_options(
            config,
            self.settings,
            has_adapter=bool(adapter),
            native_context=self.native_context_tokens,
        )
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="vllm-loop", daemon=True)
        self.thread.start()

        async def start():
            self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
            if self.lora_request:
                if not await self.engine.add_lora(self.lora_request):
                    raise RuntimeError("vLLM did not load the requested adapter")
                await self.engine.pin_lora(1)

        try:
            asyncio.run_coroutine_threadsafe(start(), self.loop).result()
        except BaseException:
            self.close()
            raise
        self.metadata = {
            "backend": "vllm",
            "base_model": MODEL,
            "base_revision": REVISION,
            "base_dtype": "bfloat16",
            "adapter_dtype": "bfloat16" if adapter else None,
            "adapter_storage_dtype": "float32" if adapter else None,
            "export": exported,
            "config": config.to_dict(),
            "engine_options": options,
            "settings": asdict(self.settings),
            "packages": {p: version(p) for p in ("vllm", "torch", "transformers", "peft")},
            "gpus": [torch.cuda.get_device_name(0)],
            "capabilities": [
                "cancel_jobs",
                "client_job_identity",
                "seeded_sampling",
                "request_adapter_selection",
                "chunked_prefill",
                "prefix_caching",
            ],
            "adapter_modes": ["lora", "base"] if adapter else ["base"],
            "reasoning_loop_guard": False,
            "tokenizer": {
                "chat_template_sha256": hashlib.sha256(
                    self.tokenizer.chat_template.encode()
                ).hexdigest(),
                "chat_eos_token_id": self.tokenizer.eos_token_id,
                "model_eos_token_id": generation.eos_token_id,
                "thinking_end_token_id": self.tokenizer.convert_tokens_to_ids("</think>"),
            },
            "native_context_tokens": self.native_context_tokens,
            "effective_context_tokens": options["max_model_len"],
            "effective_prefill_chunk_tokens": self.settings.chunk_tokens,
            "implementation": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in (
                    "vllm_backend.py",
                    "serving.py",
                    "protocol.py",
                    "decoding.py",
                    "generation.py",
                    "config.py",
                )
            },
        }
        self.metadata["fingerprint"] = fingerprint(self.metadata)

    def close(self):
        async def shutdown():
            if getattr(self, "engine", None) is not None:
                self.engine.shutdown()

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            self.loop.close()

    def run_request(self, ids, params, lora, control):
        future = asyncio.run_coroutine_threadsafe(
            collect_completion(self.engine, ids, params, lora, control, uuid.uuid4().hex), self.loop
        )
        return future.result()

    def complete(self, request, *, control=None):
        cfg = self.config
        if request.get("model") != cfg.model_name:
            raise ValueError(f"Unknown model; expected {cfg.model_name}")
        if request.get("stream") or request.get("n", 1) != 1:
            raise ValueError("Only non-streaming single completions are supported")
        mode, scale = request.get("adapter_mode", "loaded"), request.get("adapter_scale", 1.0)
        if mode not in ("loaded", "base", "lora"):
            raise ValueError("adapter_mode must be loaded, base or lora")
        if type(scale) not in (int, float) or scale != 1:
            raise ValueError("vLLM pilot only supports adapter_scale=1")
        lora = None if mode == "base" else self.lora_request
        if mode == "lora" and lora is None:
            raise ValueError("This host has no LoRA adapter")
        decoding = Decoding(
            **{
                name: request.get(name, getattr(cfg, name))
                for name in ("temperature", "top_p", "top_k", "seed")
            }
        )
        for name, neutral in (
            ("repetition_penalty", 1),
            ("presence_penalty", 0),
            ("frequency_penalty", 0),
            ("no_repeat_ngram_size", 0),
        ):
            if request.get(name, neutral) != neutral:
                raise ValueError(f"{name} is disabled for this comparison")
        if request.get("stop") or request.get("tool_choice", "auto") not in (None, "auto"):
            raise ValueError("Custom stops and forced tool selection are unsupported")
        messages, tools = normalize_messages(request.get("messages")), request.get("tools", [])
        if not tools or any(t.get("function", {}).get("name") != "bash" for t in tools):
            raise ValueError("mini-SWE-agent's native bash tool is required")
        ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
            tokenize=True,
            return_dict=False,
        )
        limit = resolve_output_limit(
            cfg, request, prompt_tokens=len(ids), model_context_tokens=self.native_context_tokens
        )
        params = self.sampling_class(
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k or -1,
            seed=decoding.seed,
            max_tokens=limit,
            stop_token_ids=self.eos_token_ids,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            skip_special_tokens=False,
            ignore_eos=False,
            output_kind=self.final_output_kind,
        )
        started = time.monotonic()
        output = self.run_request(ids, params, lora, control)
        if len(output.outputs) != 1 or output.outputs[0].finish_reason not in ("stop", "length"):
            raise RuntimeError("vLLM did not finish a single completion normally")
        completion = output.outputs[0]
        tokens = list(completion.token_ids)
        stopped = completion.finish_reason == "stop"
        if stopped and tokens and tokens[-1] in self.eos_token_ids:
            text_tokens = tokens[:-1]
        else:
            text_tokens = tokens
        text = self.tokenizer.decode(text_tokens, skip_special_tokens=False)
        message, calls = parse_completion(text, thinking=cfg.enable_thinking, truncated=not stopped)
        elapsed = time.monotonic() - started
        cached = getattr(output, "num_cached_tokens", None) or 0
        usage = {
            "prompt_tokens": len(ids),
            "completion_tokens": len(tokens),
            "total_tokens": len(ids) + len(tokens),
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        effective = {
            "backend": "vllm",
            "decoding": decoding.to_dict(),
            "adapter_mode": "lora" if lora else "base",
            "adapter_scale": 1.0 if lora else 0.0,
            "adapter_dtype": "bfloat16" if lora else None,
            "prefill_mode": "chunked",
            "prefill_chunk_tokens": self.settings.chunk_tokens,
            "prefix_caching": self.settings.prefix_caching,
            "eos_token_ids": self.eos_token_ids,
        }
        metrics = {"seconds": round(elapsed, 3), "cached_prompt_tokens": cached}
        engine_metrics = getattr(output, "metrics", None)
        if engine_metrics is not None:
            metrics["time_to_first_token_seconds"] = getattr(
                engine_metrics, "first_token_latency", None
            )
            first = getattr(engine_metrics, "first_token_ts", 0)
            last = getattr(engine_metrics, "last_token_ts", 0)
            if last > first > 0:
                metrics["decode_tokens_per_second"] = round((len(tokens) - 1) / (last - first), 3)
        logger.info(
            json.dumps(
                {
                    "event": "completion",
                    **usage,
                    **metrics,
                    "finish_reason": completion.finish_reason,
                    "effective_settings": effective,
                }
            )
        )
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": cfg.model_name,
            "usage": usage,
            "guardrail_stop": None,
            "effective_settings": effective,
            "backend_metrics": metrics,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if calls else completion.finish_reason,
                }
            ],
        }
