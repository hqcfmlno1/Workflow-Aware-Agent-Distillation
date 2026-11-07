"""Offline quality annotation. Never execute, rewrite or train on source commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

import httpx

REASON_CODES = (
    "OFF_TASK",
    "REDUNDANT_REASONING",
    "IGNORES_EVIDENCE",
    "UNSUPPORTED_CLAIM",
    "REASONING_ACTION_CONFLICT",
    "REDUNDANT_ACTION",
    "AVOIDABLE_ACTION_ERROR",
    "TEST_GAMING",
)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_record(record):
    """Lossless field mapping; step IDs number assistant messages, not tool calls."""
    sources, steps, messages = {}, {}, []
    step = 0
    for index, message in enumerate(record["messages"]):
        if message["role"] == "assistant":
            step += 1
            steps[step] = index
        fields = {}
        for name, value in message.items():
            if name == "role":
                continue
            source_id = f"message_{index}.{name}"
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            sources[source_id] = text
            fields[name] = {"source_id": source_id, "value": value}
        messages.append(
            {
                "message_index": index,
                "step_id": step or None,
                "role": message["role"],
                "fields": fields,
            }
        )
    if not steps:
        raise ValueError("No assistant steps")
    document = {
        "trajectory_id": record["id"],
        "task": record.get("instance_id"),
        "messages": messages,
        "final_model_patch": record.get("model_patch"),
        "step_count": len(steps),
    }
    return document, sources, steps


def validate_mask(annotation, trajectory_id, sources, steps, *, quote_required=True):
    if not isinstance(annotation, dict) or set(annotation) != {"trajectory_id", "mask_steps"}:
        raise ValueError("Require exactly trajectory_id and mask_steps")
    if annotation["trajectory_id"] != trajectory_id:
        raise ValueError("Wrong trajectory_id")
    if not isinstance(annotation["mask_steps"], list):
        raise ValueError("mask_steps must be an array")
    seen = set()
    for item in annotation["mask_steps"]:
        expected = {"step_id", "reason_codes", "reason"}
        if quote_required:
            expected.add("evidence")
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("Unexpected mask fields")
        step_id = item["step_id"]
        if type(step_id) is not int or step_id not in steps or step_id in seen:
            raise ValueError("Unknown, duplicate or invalid step_id")
        seen.add(step_id)
        codes = item["reason_codes"]
        if (
            not isinstance(codes, list)
            or not codes
            or any(not isinstance(c, str) or c not in REASON_CODES for c in codes)
            or len(set(codes)) != len(codes)
        ):
            raise ValueError("Invalid reason_codes")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("Missing reason")
        if not quote_required:
            continue
        if not isinstance(item["evidence"], list) or not item["evidence"]:
            raise ValueError("Missing evidence")
        own_evidence = False
        for evidence in item["evidence"]:
            if not isinstance(evidence, dict) or set(evidence) != {"source_id", "quote"}:
                raise ValueError("Invalid evidence fields")
            source, quote = evidence["source_id"], evidence["quote"]
            if (
                not isinstance(source, str)
                or source not in sources
                or not isinstance(quote, str)
                or not quote.strip()
                or quote not in sources[source]
            ):
                raise ValueError("Unknown source_id or nonliteral evidence quote")
            own_evidence |= source.startswith(f"message_{steps[step_id]}.")
        if not own_evidence:
            raise ValueError("Mask lacks evidence from its own assistant message")
    return {
        "mask_step_ids": sorted(seen),
        "mask_message_indices": [steps[s] for s in sorted(seen)],
        "assistant_steps": len(steps),
    }


def annotation_schema(*, quote_required=True):
    string = {"type": "string"}
    evidence = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"source_id": string, "quote": string},
        "required": ["source_id", "quote"],
    }
    mask = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "step_id": {"type": "integer"},
            "reason_codes": {"type": "array", "items": {"type": "string", "enum": REASON_CODES}},
            "reason": string,
            "evidence": {"type": "array", "items": evidence},
        },
        "required": ["step_id", "reason_codes", "reason", "evidence"],
    }
    if not quote_required:
        del mask["properties"]["evidence"]
        mask["required"].remove("evidence")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"trajectory_id": string, "mask_steps": {"type": "array", "items": mask}},
        "required": ["trajectory_id", "mask_steps"],
    }


def extract_output(raw):
    if "choices" in raw:
        if len(raw["choices"]) != 1 or raw["choices"][0].get("finish_reason") != "stop":
            raise ValueError("Chat output did not finish normally")
        text = raw["choices"][0]["message"].get("content")
    else:
        if raw.get("status") != "completed":
            raise ValueError("Responses output incomplete")
        parts = []
        for output in raw.get("output", []):
            if output.get("type") == "message":
                for part in output.get("content", []):
                    if part.get("type") == "refusal":
                        raise ValueError("Judge refused")
                    if part.get("type") == "output_text":
                        parts.append(part["text"])
        text = "".join(parts)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No annotation output")
    return json.loads(text)


def read_completed(path, request_sha256, trajectory_id, sources, steps):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("status") != "completed" or data.get("request_sha256") != request_sha256:
            return None
        validated = validate_mask(
            data["annotation"],
            trajectory_id,
            sources,
            steps,
            quote_required=data.get("config", {}).get("schema_version") != "no_quote_v2",
        )
        return {**data, **validated}
    except (ValueError, KeyError, TypeError):
        return None


def request_identity(record, config, policy):
    return fingerprint({"source_record": record, "config": config, "policy": policy})


def judge_record(
    record,
    output_dir,
    base_url,
    key,
    config,
    policy,
    *,
    client=None,
    timeout=900,
    retries=2,
    emit=None,
):
    ident = record["id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", ident) or ident in {".", ".."}:
        raise ValueError("Unsafe trajectory filename")
    document, sources, steps = prepare_record(record)
    identity = request_identity(record, config, policy)
    destination = output_dir / "results" / f"{ident}.json"
    prior = read_completed(destination, identity, ident, sources, steps)
    if prior:
        return {**prior, "status": "resumed"}
    payload = {"model": config["model"]}
    quote_required = config.get("schema_version") != "no_quote_v2"
    structured = {
        "name": "trajectory_step_mask",
        "strict": True,
        "schema": annotation_schema(quote_required=quote_required),
    }
    serialized = json.dumps(document, ensure_ascii=False)
    if config["api_kind"] == "responses":
        endpoint = "/responses"
        payload.update(
            instructions=policy,
            input=serialized,
            max_output_tokens=config["max_output_tokens"],
            store=False,
        )
        if config.get("effort"):
            payload["reasoning"] = {"effort": config["effort"]}
        if config["structured"]:
            payload["text"] = {"format": {"type": "json_schema", **structured}}
    else:
        endpoint = "/chat/completions"
        payload.update(
            messages=[
                {"role": "system", "content": policy},
                {"role": "user", "content": serialized},
            ],
            max_completion_tokens=config["max_output_tokens"],
        )
        if config.get("effort"):
            payload["reasoning_effort"] = config["effort"]
        if config["structured"]:
            payload["response_format"] = {"type": "json_schema", "json_schema": structured}
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=False)
    started = time.monotonic()
    try:
        for attempt in range(retries + 1):
            attempt_id = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
            receipt = {
                "trajectory_id": ident,
                "request_sha256": identity,
                "attempt": attempt + 1,
                "status": "started",
                "config": config,
            }
            attempt_path = output_dir / "attempts" / ident / f"{attempt_id}.json"
            atomic_json(attempt_path, receipt)
            retryable = True
            try:
                response = client.post(
                    base_url.rstrip("/") + endpoint,
                    json=payload,
                    headers={"Authorization": "Bearer " + key},
                )
                if response.status_code >= 400:
                    receipt["http_status"] = response.status_code
                    retryable = response.status_code == 429 or response.status_code >= 500
                    raise ValueError(f"HTTP {response.status_code}")
                raw = response.json()
                receipt["api_response"] = raw
                receipt["usage"] = raw.get("usage", {})
                annotation = extract_output(raw)
                validated = validate_mask(
                    annotation, ident, sources, steps, quote_required=quote_required
                )
                result = {
                    "status": "completed",
                    "trajectory_id": ident,
                    "task": record.get("instance_id"),
                    "request_sha256": identity,
                    "source_record_sha256": fingerprint(record),
                    "annotation": annotation,
                    **validated,
                    "usage": raw.get("usage", {}),
                    "config": config,
                    "seconds": round(time.monotonic() - started, 3),
                    "attempt_file": str(attempt_path),
                }
                receipt["status"] = "completed"
                atomic_json(attempt_path, receipt)
                atomic_json(destination, result)
                return result
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                # Never log HTTP bodies, headers or exception reprs containing credentials.
                receipt.update(status="error", error_type=type(error).__name__)
                if isinstance(error, ValueError):
                    receipt["validation_error"] = str(error)[:240]
                atomic_json(attempt_path, receipt)
                if emit:
                    emit(
                        {
                            "event": "attempt_error",
                            "trajectory_id": ident,
                            "attempt": attempt + 1,
                            "http_status": receipt.get("http_status"),
                            "error_type": type(error).__name__,
                            "validation_error": receipt.get("validation_error"),
                        }
                    )
                if attempt == retries or not retryable:
                    return {
                        "status": "error",
                        "trajectory_id": ident,
                        "request_sha256": identity,
                        "attempt_file": str(attempt_path),
                        "seconds": round(time.monotonic() - started, 3),
                    }
                time.sleep(min(5 * 2**attempt, 30))
    finally:
        if owns_client:
            client.close()
