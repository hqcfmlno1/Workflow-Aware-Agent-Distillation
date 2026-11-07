"""Cooperative request cancellation checked at every prefill/decode boundary."""

import re
import threading
import time
from collections import Counter


class PrefillMismatch(RuntimeError):
    """A prefill strategy disagrees with the native full-sequence reference."""


class GenerationStopped(RuntimeError):
    pass


class GenerationControl:
    def __init__(self, seconds):
        self.deadline = None if seconds is None else time.monotonic() + seconds
        self.cancelled = threading.Event()
        self._progress_lock = threading.Lock()
        self._started = time.monotonic()
        self._updated = self._started
        self._decode_started = None
        self._last_progress_log = float("-inf")
        self._text_decoder = None
        self._progress = {"stage": "queued", "prompt_tokens": 0,
                          "prefilled_tokens": 0, "generated_tokens": 0,
                          "text_tail": "", "thinking_end_seen": False}

    def set_text_decoder(self, decoder):
        self._text_decoder = decoder

    def report_progress(self, *, stage, prompt_tokens=None, prefilled_tokens=None, tokens=None):
        # Observation only: no RNG, model calls, stopping conditions or prompt changes.
        text = None
        if tokens is not None and self._text_decoder is not None:
            text = self._text_decoder(tokens[-512:], skip_special_tokens=False)
        now = time.monotonic()
        with self._progress_lock:
            self._progress["stage"] = stage
            if prompt_tokens is not None:
                self._progress["prompt_tokens"] = int(prompt_tokens)
            if prefilled_tokens is not None:
                self._progress["prefilled_tokens"] = int(prefilled_tokens)
            if tokens is not None:
                self._progress["generated_tokens"] = len(tokens)
                if self._decode_started is None:
                    self._decode_started = now
            if text is not None:
                self._progress["text_tail"] = text[-2048:]
                self._progress["thinking_end_seen"] |= "</think>" in text
            self._updated = now
            should_log = now - self._last_progress_log >= 10
            if should_log:
                self._last_progress_log = now
        if should_log:
            import json
            import logging
            logging.getLogger(__name__).info(json.dumps(
                {"event": "generation_progress", **self.progress_snapshot()}, ensure_ascii=False))

    def progress_snapshot(self):
        now = time.monotonic()
        with self._progress_lock:
            result = dict(self._progress)
            result["elapsed_seconds"] = round(now - self._started, 2)
            result["last_progress_age_seconds"] = round(now - self._updated, 2)
            result["updated_at"] = time.time() - (now - self._updated)
            decode_seconds = now - self._decode_started if self._decode_started is not None else 0
            result["decode_tokens_per_second"] = round(
                result["generated_tokens"] / decode_seconds, 2) if decode_seconds > 0 else 0.0
        return result

    def check(self):
        if self.cancelled.is_set():
            raise GenerationStopped("Generation cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise GenerationStopped("Generation deadline exceeded")


def repetitive_reasoning(text):
    """Stop exact long prose loops inside thinking; never inspect emitted tool/code content."""
    if "</think>" in text:
        return False
    prose = re.sub(r"```.*?(?:```|$)", "", text, flags=re.DOTALL)
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", prose)]
    counts = Counter(part for part in paragraphs if len(part) >= 120)
    return any(count >= 3 for count in counts.values())


def validate_inference(model, input_ids, *, chunk_size=512, strict=True):
    """Compare real hybrid-cache logits and native generate before admitting tasks."""
    import torch

    from .serving import greedy_generate

    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=True, logits_to_keep=1).logits.float()
        cache = None
        for start in range(0, input_ids.shape[1], chunk_size):
            output = model(
                input_ids=input_ids[:, start : start + chunk_size],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
        actual = output.logits.float()
        if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
            raise FloatingPointError("Nonfinite prefill validation logits")
        logit_error = None
        try:
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        except AssertionError as exc:
            if strict:
                raise PrefillMismatch(str(exc)) from exc
            logit_error = str(exc)
        max_error = (actual - expected).abs().max().item()
        del output, cache, expected, actual
        native = model.generate(
            input_ids=input_ids,
            max_new_tokens=4,
            do_sample=False,
            eos_token_id=[],
            pad_token_id=0,
            logits_to_keep=1,
        )
        generated, _ = greedy_generate(
            model, input_ids, max_new_tokens=4, prefill_chunk_tokens=chunk_size, eos_token_ids=set()
        )
        expected_tokens = native[0, input_ids.shape[1] :].tolist()
        tokens_match = generated == expected_tokens
        if not tokens_match and strict:
            raise PrefillMismatch(
                "Chunked generation differs from native generate; refusing benchmark"
            )
    return {
        "cache_logits_match": logit_error is None,
        "native_generate_match": tokens_match,
        "prompt_tokens": input_ids.shape[1],
        "max_logit_error": max_error,
        **({"logit_validation_error": logit_error} if logit_error else {}),
        "native_greedy_tokens": expected_tokens,
        "custom_greedy_tokens": generated,
    }


def validate_sampling(model, input_ids, *, chunk_size, decoding=None, strict=True):
    """Check the actual loaded model's sampled path without altering global RNG."""
    import torch

    from .decoding import Decoding
    from .serving import generate_tokens

    decoding = decoding or Decoding()
    if decoding.temperature == 0:
        decoding = Decoding(seed=decoding.seed)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(decoding.seed)
        native = model.generate(
            input_ids,
            max_new_tokens=4,
            do_sample=True,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            eos_token_id=[],
            pad_token_id=0,
            logits_to_keep=1,
        )
        generated, _ = generate_tokens(
            model,
            input_ids,
            max_new_tokens=4,
            prefill_chunk_tokens=chunk_size,
            eos_token_ids=set(),
            decoding=decoding,
        )
        expected_tokens = native[0, input_ids.shape[1] :].tolist()
        tokens_match = generated == expected_tokens
        if not tokens_match and strict:
            raise PrefillMismatch("Custom sampling differs from native generate")
    return {
        "native_sampling_match": tokens_match,
        "native_sampling_tokens": expected_tokens,
        "custom_sampling_tokens": generated,
        "checked_decoding": decoding.to_dict(),
    }


def choose_prefill(model, input_ids, *, chunk_size, max_context_tokens, mode="auto", strict=True):
    """Retain strict checks; retry with native full prefill on cache disagreement."""
    import gc
    import logging

    import torch

    if mode not in ("auto", "full", "chunked"):
        raise ValueError("prefill_mode must be auto, full or chunked")
    if not strict and mode != "chunked":
        raise ValueError("Report-only validation requires explicit chunked prefill")
    if mode == "full":
        validation = validate_inference(model, input_ids, chunk_size=max_context_tokens)
        return max_context_tokens, {**validation, "prefill_mode": "full"}
    error = None
    try:
        validation = validate_inference(model, input_ids, chunk_size=chunk_size, strict=strict)
    except PrefillMismatch as exc:
        error = str(exc)
    if error is None:
        if not strict:
            logging.getLogger(__name__).warning(
                "EXPERIMENTAL chunked prefill: validation is report-only: %s", validation
            )
        return chunk_size, {**validation, "prefill_mode": "chunked"}
    if mode == "chunked":
        raise PrefillMismatch(error)
    # Leave the exception scope before retrying so its traceback cannot retain
    # the failed forward's CUDA tensors/cache during the reference validation.
    logging.getLogger(__name__).warning(
        "Chunked prefill validation failed; checking full-sequence prefill: %s", error
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    validation = validate_inference(model, input_ids, chunk_size=max_context_tokens)
    return max_context_tokens, {
        **validation,
        "prefill_mode": "full",
        "fallback_from_chunk_tokens": chunk_size,
        "chunked_validation_error": error,
    }
