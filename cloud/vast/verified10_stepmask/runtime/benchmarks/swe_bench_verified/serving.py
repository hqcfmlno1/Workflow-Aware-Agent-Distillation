"""Serialized FP16/NF4 inference with bounded text prefill and native tools."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
import secrets
import threading
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from .adapter_variants import inference_variant
from .config import MODEL, REVISION, fingerprint, sha256_file
from .decoding import Decoding
from .export_adapter import verify_export
from .protocol import normalize_messages, parse_completion

logger = logging.getLogger(__name__)


def inference_attention(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs
):
    """Dynamic KV is updated by the layer before this function; no training patch."""
    from contextlib import nullcontext

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if kwargs.get("output_attentions") or kwargs.get("cache") is not None:
        raise ValueError("Paged attention and attention outputs are unsupported")
    repeats, remainder = divmod(query.shape[1], key.shape[1])
    if remainder:
        raise ValueError("Incompatible query/KV heads")
    key, value = key.repeat_interleave(repeats, 1), value.repeat_interleave(repeats, 1)
    causal = bool(
        attention_mask is None
        and query.shape[-2] > 1
        and (getattr(module, "is_causal", True) if is_causal is None else is_causal)
    )
    backend = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if query.is_cuda else nullcontext()
    with backend:
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=causal,
        )
    return output.transpose(1, 2).contiguous(), None


def register_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register("benchmark_efficient_sdpa", inference_attention)
    AttentionMaskInterface.register("benchmark_efficient_sdpa", sdpa_mask)


def generate_tokens(
    model,
    input_ids,
    *,
    max_new_tokens,
    prefill_chunk_tokens,
    eos_token_ids,
    decoding,
    control=None,
    repetition_check=None,
):
    """Retain every input token, reusing both KV and recurrent state between chunks."""
    import torch

    cache = None
    generator = None
    report = getattr(control, "report_progress", None)
    if report:
        report(stage="prefill", prompt_tokens=input_ids.shape[1], prefilled_tokens=0)
    with torch.inference_mode():
        for start in range(0, input_ids.shape[1], prefill_chunk_tokens):
            if control is not None:
                control.check()
            output = model(
                input_ids=input_ids[:, start : start + prefill_chunk_tokens],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache, logits = output.past_key_values, output.logits[:, -1, :]
            del output
            if report:
                report(stage="prefill", prefilled_tokens=min(start + prefill_chunk_tokens, input_ids.shape[1]))
        generated = []
        stopped = False
        position = 0
        while max_new_tokens is None or position < max_new_tokens:
            if control is not None:
                control.check()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite inference logits")
            if generator is None:
                generator = torch.Generator(device=logits.device).manual_seed(decoding.seed)
            token = decoding.select_token(logits, generator=generator)
            generated.append(token)
            if report and (len(generated) == 1 or len(generated) % 32 == 0 or token in eos_token_ids):
                report(stage="decode", tokens=generated)
            if token in eos_token_ids:
                stopped = True
                break
            if (
                repetition_check is not None
                and len(generated) % 64 == 0
                and repetition_check(generated)
            ):
                break
            position += 1
            if max_new_tokens is None or position < max_new_tokens:
                output = model(
                    input_ids=torch.tensor([[token]], device=input_ids.device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache, logits = output.past_key_values, output.logits[:, -1, :]
                del output
    if report:
        report(stage="finished", tokens=generated)
    return generated, stopped


def greedy_generate(model, input_ids, **kwargs):
    """Reference path retained for native-generation/cache equivalence checks."""
    return generate_tokens(
        model, input_ids, decoding=Decoding(temperature=0, top_p=1, top_k=0), **kwargs
    )


def model_loading_options(config):
    """Keep NF4 explicit and use the same quantization settings as training."""
    import torch

    from workflow_distillation.optimized_qlora import device_plan

    options = dict(
        dtype=torch.float16,
        device_map=device_plan(32, split_at=config.split_at_layer),
        attn_implementation="benchmark_efficient_sdpa",
    )
    if config.base_dtype == "nf4":
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["lm_head", "model.visual"],
            llm_int8_enable_fp32_cpu_offload=True,
        )
    return options


def prepare_inference_model(model, config):
    """Match training's floating-point preparation without enabling autograd."""
    import torch

    if config.base_dtype == "nf4":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
        if embedding.weight is not head.weight:
            raise RuntimeError("Expected tied embedding/head for the NF4 comparison")
        embedding.to(dtype=torch.float16)
    model.eval().requires_grad_(False)
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    return model


def resolve_output_limit(config, request, *, prompt_tokens, model_context_tokens):
    """Null caps use native context capacity; never reserve an invented output quota."""
    limit = request.get("max_tokens", config.max_new_tokens)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_tokens must be null or a positive integer")
    if config.max_new_tokens is not None:
        if limit is None:
            limit = config.max_new_tokens
        if limit > config.max_new_tokens:
            raise ValueError("max_tokens exceeds the configured output budget")
    context = config.max_context_tokens or model_context_tokens
    if type(context) is not int or context < 1:
        raise ValueError("Model configuration does not declare a native context window")
    if prompt_tokens >= context:
        label = "Model native context" if config.max_context_tokens is None else "Context budget"
        raise ValueError(
            f"{label} exceeded: {prompt_tokens} prompt >= {context}; history is never truncated"
        )
    total_budget = request.get("max_total_tokens")
    if total_budget is not None:
        if type(total_budget) is not int or total_budget <= prompt_tokens:
            raise ValueError("Total token budget exhausted by prompt")
        limit = (
            min(limit, total_budget - prompt_tokens)
            if limit is not None
            else total_budget - prompt_tokens
        )
    # Preserve explicitly bounded experiments; unbounded runs use all remaining
    # native positions without rejecting a prompt for an unused output reserve.
    if (
        config.max_context_tokens is not None
        and limit is not None
        and prompt_tokens + limit > context
    ):
        raise ValueError(
            f"Context budget exceeded: {prompt_tokens} prompt + {limit} output > "
            f"{context}; history is never silently truncated"
        )
    remaining = context - prompt_tokens
    return min(limit, remaining) if limit is not None else remaining


class TransformersBackend:
    def __init__(self, config, adapter=None):
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.config = config
        exported = verify_export(adapter) if adapter else None
        required = 1 if config.split_at_layer is None else 2
        if not torch.cuda.is_available() or torch.cuda.device_count() < required:
            raise RuntimeError(f"This configuration requires {required} CUDA GPU(s)")
        register_attention()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        options = model_loading_options(config)
        self.model = prepare_inference_model(
            Qwen3_5ForConditionalGeneration.from_pretrained(
                MODEL,
                revision=REVISION,
                **options,
            ),
            config,
        )
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
            for name, parameter in self.model.named_parameters():
                if ".lora_" in name:
                    parameter.data = parameter.data.float()
        self.model.eval().requires_grad_(False)
        self.model.config.use_cache = True
        self.model.gradient_checkpointing_disable()
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.native_context_tokens = getattr(text_config, "max_position_embeddings", None)
        self.context_tokens = config.max_context_tokens or self.native_context_tokens
        if type(self.context_tokens) is not int or self.context_tokens < 1:
            raise ValueError("Model configuration does not declare a native context window")
        from .generation import choose_prefill, validate_sampling

        # Cover multiple prefill chunks and both full/linear attention states on
        # the actual loaded NF4 model/adapter, before exposing a public endpoint.
        check_ids = torch.arange(1, 1026, device=self.model.get_input_embeddings().weight.device)
        strict_prefill = config.prefill_validation == "strict"
        self.prefill_chunk_tokens, validation = choose_prefill(
            self.model,
            check_ids.view(1, -1),
            chunk_size=config.prefill_chunk_tokens,
            max_context_tokens=self.context_tokens,
            mode=config.prefill_mode,
            strict=strict_prefill,
        )
        base_validation = validation
        if adapter:
            with inference_variant(self.model, mode="base"):
                base_chunk, base_validation = choose_prefill(
                    self.model,
                    check_ids.view(1, -1),
                    chunk_size=config.prefill_chunk_tokens,
                    max_context_tokens=self.context_tokens,
                    mode=config.prefill_mode,
                    strict=strict_prefill,
                )
            if base_chunk != self.prefill_chunk_tokens:
                # Both variants must use one strategy, even in automatic mode.
                self.prefill_chunk_tokens, validation = choose_prefill(
                    self.model,
                    check_ids.view(1, -1),
                    chunk_size=config.prefill_chunk_tokens,
                    max_context_tokens=self.context_tokens,
                    mode="full",
                )
                with inference_variant(self.model, mode="base"):
                    _, base_validation = choose_prefill(
                        self.model,
                        check_ids.view(1, -1),
                        chunk_size=config.prefill_chunk_tokens,
                        max_context_tokens=self.context_tokens,
                        mode="full",
                    )
        decoding = Decoding(config.temperature, config.top_p, config.top_k, config.seed)
        validation.update(
            validate_sampling(
                self.model,
                check_ids.view(1, -1),
                chunk_size=self.prefill_chunk_tokens,
                decoding=decoding,
                strict=strict_prefill,
            )
        )
        if adapter:
            with inference_variant(self.model, mode="base"):
                base_validation.update(
                    validate_sampling(
                        self.model,
                        check_ids.view(1, -1),
                        chunk_size=self.prefill_chunk_tokens,
                        decoding=decoding,
                        strict=strict_prefill,
                    )
                )
        del check_ids
        gc.collect()
        torch.cuda.empty_cache()
        self.metadata = {
            "base_model": MODEL,
            "base_revision": REVISION,
            "base_dtype": config.base_dtype,
            "adapter_dtype": "float32" if adapter else None,
            "export": exported,
            "config": config.to_dict(),
            "attention": "benchmark_efficient_sdpa",
            "logits_to_keep": 1,
            "packages": {p: version(p) for p in ("torch", "transformers", "peft")},
            "gpus": [torch.cuda.get_device_name(i) for i in range(required)],
            "capabilities": [
                "generation_progress_v1",
                "cancel_jobs",
                "client_job_identity",
                "bounded_generation",
                "prefill_self_check",
                "seeded_sampling",
                "request_adapter_selection",
            ],
            "inference_validation": validation,
            "base_inference_validation": base_validation,
            "prefill_validation_policy": config.prefill_validation,
            "adapter_modes": ["lora", "base"] if adapter else ["base"],
            "reasoning_loop_guard": config.reasoning_loop_guard,
            "tokenizer": {
                "chat_template_sha256": hashlib.sha256(
                    self.tokenizer.chat_template.encode()
                ).hexdigest(),
                "chat_eos_token_id": self.tokenizer.eos_token_id,
                "model_eos_token_id": self.model.generation_config.eos_token_id,
                "thinking_end_token_id": self.tokenizer.convert_tokens_to_ids("</think>"),
            },
            "effective_prefill_chunk_tokens": self.prefill_chunk_tokens,
            "native_context_tokens": self.native_context_tokens,
            "effective_context_tokens": self.context_tokens,
            "implementation": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in (
                    "serving.py",
                    "generation.py",
                    "protocol.py",
                    "decoding.py",
                    "adapter_variants.py",
                )
            },
        }
        if config.base_dtype == "nf4":
            self.metadata["quantization"] = options["quantization_config"].to_dict()
            self.metadata["float_preparation"] = "training_kbit_with_fp16_tied_embedding_head"
            self.metadata["packages"]["bitsandbytes"] = version("bitsandbytes")
        self.metadata["fingerprint"] = fingerprint(self.metadata)

    def complete(self, request, *, control=None):
        import torch

        cfg = self.config
        if request.get("model") != cfg.model_name:
            raise ValueError(f"Unknown model; expected {cfg.model_name}")
        if request.get("stream") or request.get("n", 1) != 1:
            raise ValueError("Only non-streaming, single-completion requests are supported")
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
                raise ValueError(f"{name} is disabled for this controlled comparison")
        if request.get("stop") or request.get("tool_choice", "auto") not in (None, "auto"):
            raise ValueError("Custom stop strings and forced tool selection are unsupported")
        messages = normalize_messages(request.get("messages"))
        tools = request.get("tools", [])
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
            cfg,
            request,
            prompt_tokens=len(ids),
            model_context_tokens=getattr(self, "native_context_tokens", cfg.max_context_tokens),
        )
        if control is not None:
            control.set_text_decoder(self.tokenizer.decode)
        input_ids = torch.tensor([ids], device=self.model.get_input_embeddings().weight.device)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]) - {None}
        # The base end-of-text and tokenizer end-of-turn IDs can differ.
        if self.tokenizer.eos_token_id is not None:
            eos.add(self.tokenizer.eos_token_id)
        devices = range(1 if cfg.split_at_layer is None else 2)
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        try:
            from .generation import repetitive_reasoning

            with inference_variant(
                self.model,
                mode=request.get("adapter_mode", "loaded"),
                scale=request.get("adapter_scale", 1.0),
            ) as variant:
                generated, stopped = generate_tokens(
                    self.model,
                    input_ids,
                    max_new_tokens=limit,
                    prefill_chunk_tokens=getattr(
                        self, "prefill_chunk_tokens", cfg.prefill_chunk_tokens
                    ),
                    eos_token_ids=eos,
                    decoding=decoding,
                    control=control,
                    repetition_check=(
                        lambda tokens: repetitive_reasoning(
                            self.tokenizer.decode(tokens, skip_special_tokens=False)
                        )
                    )
                    if cfg.enable_thinking and cfg.reasoning_loop_guard
                    else None,
                )
            effective = {
                "decoding": decoding.to_dict(),
                **variant,
                "prefill_mode": "full"
                if getattr(self, "prefill_chunk_tokens", cfg.prefill_chunk_tokens)
                >= (cfg.max_context_tokens or self.native_context_tokens)
                else "chunked",
                "eos_token_ids": sorted(eos),
            }
            guardrail_stop = "reasoning_loop" if not stopped and len(generated) < limit else None
            text = self.tokenizer.decode(
                generated[:-1] if stopped else generated,
                skip_special_tokens=False,
            )
            message, calls = parse_completion(
                text,
                thinking=cfg.enable_thinking,
                truncated=not stopped,
            )
            elapsed = time.monotonic() - started
            usage = {
                "prompt_tokens": len(ids),
                "completion_tokens": len(generated),
                "total_tokens": len(ids) + len(generated),
            }
            logger.info(
                json.dumps(
                    {
                        "event": "completion",
                        **usage,
                        "seconds": round(elapsed, 3),
                        "peak_allocated_bytes": [
                            torch.cuda.max_memory_allocated(d) for d in devices
                        ],
                        "finish_reason": "stop" if stopped else "length",
                        "guardrail_stop": guardrail_stop,
                        "effective_settings": effective,
                    }
                )
            )
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": cfg.model_name,
                "usage": usage,
                "guardrail_stop": guardrail_stop,
                "effective_settings": effective,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls"
                        if calls
                        else ("stop" if stopped else "length"),
                    }
                ],
            }
        finally:
            # No request-to-request prefix cache. Release failed requests as well.
            del input_ids
            gc.collect()
            torch.cuda.empty_cache()


def create_app(backend, api_key):
    import inspect

    from fastapi import FastAPI, Header, HTTPException

    from .generation import GenerationControl, GenerationStopped

    if not api_key:
        raise ValueError("Set BENCH_API_KEY before starting the server")
    app = FastAPI(title="Qwen3.5 FP16 SWE-bench adapter")
    lock = threading.Lock()
    jobs_lock = threading.Lock()
    jobs = {}
    controls = {}
    accepts_control = "control" in inspect.signature(backend.complete).parameters

    def control_for(request):
        import math

        ceiling = getattr(backend.config, "request_timeout_seconds", 900)
        seconds = request.get("max_seconds", ceiling)
        if seconds is None and ceiling is not None:
            seconds = ceiling
        if seconds is not None and (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
            or (ceiling is not None and seconds > ceiling)
        ):
            raise HTTPException(400, "Invalid generation deadline")
        return GenerationControl(seconds)

    def invoke(request, control):
        return (
            backend.complete(request, control=control)
            if accepts_control
            else backend.complete(request)
        )

    def authorize(authorization):
        if not secrets.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(401, "Invalid benchmark API key")

    def admit(authorization, expected_fingerprint):
        authorize(authorization)
        if expected_fingerprint is not None and expected_fingerprint != backend.metadata.get(
            "fingerprint"
        ):
            raise HTTPException(409, "Serving model fingerprint changed")

    @app.get("/health")
    def health(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return backend.metadata

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"object": "list", "data": [{"id": backend.config.model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def complete(
        request: dict,
        authorization: str | None = Header(default=None),
        x_benchmark_fingerprint: str | None = Header(default=None),
    ):
        admit(authorization, x_benchmark_fingerprint)
        control = control_for(request)
        # One active request; reject concurrency instead of retaining queued prompt bodies.
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "One request is already running; use one agent worker")
        try:
            return invoke(request, control)
        except GenerationStopped as exc:
            raise HTTPException(408, str(exc)) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            lock.release()

    @app.post("/v1/completion-jobs", status_code=202)
    def submit_job(
        request: dict,
        authorization: str | None = Header(default=None),
        x_benchmark_fingerprint: str | None = Header(default=None),
        x_benchmark_job_id: str | None = Header(default=None),
    ):
        admit(authorization, x_benchmark_fingerprint)
        job_id = x_benchmark_job_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            raise HTTPException(400, "Invalid completion job ID")
        control = control_for(request)
        with jobs_lock:
            if job_id in jobs:
                return {"id": job_id, "status": jobs[job_id]["status"]}
            if not lock.acquire(blocking=False):
                raise HTTPException(429, "One request is already running; use one agent worker")
            # Retain at most eight results, with no waiting generation queue.
            while len(jobs) >= 8:
                jobs.pop(next(iter(jobs)))
            jobs[job_id] = {"id": job_id, "status": "running"}
            controls[job_id] = control

        def generate():
            try:
                result = {"status": "completed", "result": invoke(request, control)}
            except GenerationStopped as exc:
                result = {"status": "cancelled", "detail": str(exc)}
            except (ValueError, KeyError, TypeError) as exc:
                result = {"status": "failed", "error": {"status_code": 400, "detail": str(exc)}}
            except Exception:
                logger.exception("Completion job failed")
                result = {
                    "status": "failed",
                    "error": {"status_code": 500, "detail": "Inference failed; see server log"},
                }
            finally:
                result["progress"] = control.progress_snapshot()
                with jobs_lock:
                    jobs[job_id] = {"id": job_id, **result}
                    controls.pop(job_id, None)
                    lock.release()

        try:
            threading.Thread(target=generate, name="completion-job", daemon=True).start()
        except Exception:
            with jobs_lock:
                jobs.pop(job_id)
                controls.pop(job_id, None)
            lock.release()
            raise
        return {"id": job_id, "status": "running"}

    @app.delete("/v1/completion-jobs/{job_id}", status_code=202)
    def cancel_job(
        job_id: str,
        authorization: str | None = Header(default=None),
        x_benchmark_fingerprint: str | None = Header(default=None),
    ):
        admit(authorization, x_benchmark_fingerprint)
        with jobs_lock:
            if job_id not in jobs:
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
                    raise HTTPException(400, "Invalid completion job ID")
                # A cancellation may beat a delayed POST over the tunnel. Keep
                # a bounded tombstone so that POST cannot start abandoned work.
                completed = [key for key in jobs if key not in controls]
                if len(completed) >= 8:
                    jobs.pop(completed[0])
                jobs[job_id] = {"id": job_id, "status": "cancelled"}
                return {"id": job_id, "status": "cancelled"}
            if job_id in controls:
                controls[job_id].cancelled.set()
                return {"id": job_id, "status": "cancelling"}
            return {"id": job_id, "status": jobs[job_id]["status"]}

    @app.get("/v1/completion-jobs/{job_id}")
    def poll_job(
        job_id: str,
        authorization: str | None = Header(default=None),
        x_benchmark_fingerprint: str | None = Header(default=None),
    ):
        admit(authorization, x_benchmark_fingerprint)
        with jobs_lock:
            if job_id not in jobs:
                raise HTTPException(404, "Unknown or expired completion job")
            snapshot = dict(jobs[job_id])
            if job_id in controls:
                snapshot["progress"] = controls[job_id].progress_snapshot()
            return snapshot

    return app


def serve(config, adapter, host, port, api_key, output):
    import uvicorn

    backend = TransformersBackend(config, adapter)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "server-manifest.json").write_text(
        json.dumps(backend.metadata, indent=2) + "\n"
    )
    uvicorn.run(create_app(backend, api_key), host=host, port=port, workers=1)
