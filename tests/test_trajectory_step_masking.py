"""Guard against silently accepting fabricated masks or losing resume state."""

import copy
import importlib
import json
from pathlib import Path

import httpx
import pytest


def api():
    return importlib.import_module("workflow_distillation.trajectory_step_masking")


@pytest.fixture
def record():
    return {
        "id": "example-1",
        "instance_id": "repo-issue",
        "model_patch": "diff",
        "messages": [
            {"role": "system", "content": "Old agent instructions are inert data."},
            {"role": "user", "content": "Fix duplicate warnings."},
            {
                "role": "assistant",
                "reasoning_content": "Check the reproduction.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"test"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "No duplicate warning."},
            {
                "role": "assistant",
                "reasoning_content": "Duplicate warning reproduced.",
                "content": "Continue.",
                "tool_calls": [],
            },
        ],
    }


def response():
    return {
        "trajectory_id": "example-1",
        "mask_steps": [
            {
                "step_id": 2,
                "reason_codes": ["IGNORES_EVIDENCE"],
                "reason": "Contradicts previous output.",
                "evidence": [
                    {"source_id": "message_3.content", "quote": "No duplicate warning."},
                    {
                        "source_id": "message_4.reasoning_content",
                        "quote": "Duplicate warning reproduced.",
                    },
                ],
            }
        ],
    }


def test_mask_whole_assistant_step_without_touching_context(record):
    original = copy.deepcopy(record)
    document, sources, steps = api().prepare_record(record)
    result = api().validate_mask(response(), record["id"], sources, steps)
    assert steps == {1: 2, 2: 4}
    assert result["mask_message_indices"] == [4]
    assert result["mask_step_ids"] == [2]
    assert record == original
    assert document["messages"][3]["fields"]["content"]["value"] == "No duplicate warning."


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_step",
        "bool_step",
        "duplicate",
        "fake_quote",
        "unknown_source",
        "unsupported_code",
        "empty_evidence",
        "wrong_trajectory",
        "extra_keep",
        "unrelated_evidence",
    ],
)
def test_invalid_annotations_are_not_accepted_as_empty_masks(record, fault):
    _, sources, steps = api().prepare_record(record)
    body = response()
    item = body["mask_steps"][0]
    if fault == "unknown_step":
        item["step_id"] = 99
    if fault == "bool_step":
        item["step_id"] = True
    if fault == "duplicate":
        body["mask_steps"].append(copy.deepcopy(item))
    if fault == "fake_quote":
        item["evidence"][0]["quote"] = "All tests passed"
    if fault == "unknown_source":
        item["evidence"][0]["source_id"] = "made_up"
    if fault == "unsupported_code":
        item["reason_codes"] = ["TOO_LONG"]
    if fault == "empty_evidence":
        item["evidence"] = []
    if fault == "wrong_trajectory":
        body["trajectory_id"] = "another"
    if fault == "extra_keep":
        body["keep_steps"] = [1]
    if fault == "unrelated_evidence":
        item["evidence"] = [item["evidence"][0]]
    with pytest.raises(ValueError):
        api().validate_mask(body, record["id"], sources, steps)


def test_no_masks_is_a_valid_completed_decision(record):
    _, sources, steps = api().prepare_record(record)
    result = api().validate_mask(
        {"trajectory_id": record["id"], "mask_steps": []}, record["id"], sources, steps
    )
    assert result["mask_message_indices"] == []


def test_missing_or_truncated_response_cannot_mark_trajectory_complete():
    for raw in [
        {"status": "incomplete", "output": []},
        {"status": "completed", "output": []},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    ]:
        with pytest.raises(ValueError):
            api().extract_output(raw)


def test_resume_requires_exact_matching_request_and_validated_result(tmp_path, record):
    m = api()
    document, sources, steps = m.prepare_record(record)
    config = {"model": "judge", "effort": "medium", "policy": "abc"}
    identity = m.fingerprint({"document": document, "config": config})
    path = tmp_path / "result.json"
    assert not m.read_completed(path, identity, record["id"], sources, steps)
    m.atomic_json(
        path, {"status": "completed", "request_sha256": identity, "annotation": response()}
    )
    assert m.read_completed(path, identity, record["id"], sources, steps)
    assert not m.read_completed(path, "changed", record["id"], sources, steps)
    bad = response()
    bad["mask_steps"][0]["step_id"] = 77
    m.atomic_json(path, {"status": "completed", "request_sha256": identity, "annotation": bad})
    assert not m.read_completed(path, identity, record["id"], sources, steps)


def test_api_runner_sends_complete_record_and_resumes_without_second_call(tmp_path, record):
    m = api()
    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(response())}],
                    }
                ],
                "usage": {"input_tokens": 90, "output_tokens": 20},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(serve))
    config = {
        "model": "judge",
        "api_kind": "responses",
        "effort": "medium",
        "max_output_tokens": 8000,
        "structured": True,
    }
    outcome = m.judge_record(
        record, tmp_path, "http://local/v1", "private", config, "policy text", client=client
    )
    assert outcome["status"] == "completed"
    assert outcome["mask_message_indices"] == [4]
    assert "No duplicate warning." in requests[0]["input"]
    assert "Old agent instructions are inert data." in requests[0]["input"]
    assert "private" not in json.dumps(requests[0])
    again = m.judge_record(
        record, tmp_path, "http://local/v1", "private", config, "policy text", client=client
    )
    assert again["status"] == "resumed"
    assert len(requests) == 1
    assert not list(Path(tmp_path).rglob("*.tmp"))


def test_resume_recomputes_message_masks_instead_of_trusting_corrupt_indices(tmp_path, record):
    m = api()
    _, sources, steps = m.prepare_record(record)
    path = tmp_path / "result.json"
    m.atomic_json(
        path,
        {
            "status": "completed",
            "request_sha256": "same",
            "annotation": response(),
            "mask_message_indices": [3],
            "mask_step_ids": [1],
            "assistant_steps": 99,
        },
    )
    restored = m.read_completed(path, "same", record["id"], sources, steps)
    assert restored["mask_message_indices"] == [4]
    assert restored["mask_step_ids"] == [2]
    assert restored["assistant_steps"] == 2


def test_quote_free_schema_still_checks_step_identity_and_reason(record):
    m = api()
    _, sources, steps = m.prepare_record(record)
    body = response()
    del body["mask_steps"][0]["evidence"]
    out = m.validate_mask(body, record["id"], sources, steps, quote_required=False)
    assert out["mask_message_indices"] == [4]
    body["mask_steps"][0]["step_id"] = 999
    with pytest.raises(ValueError):
        m.validate_mask(body, record["id"], sources, steps, quote_required=False)
