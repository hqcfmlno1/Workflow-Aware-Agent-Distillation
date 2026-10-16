"""Verify target provenance and supervision coverage with the official tokenizer."""

import copy

import pytest
from transformers import AutoTokenizer

from scripts.data.prepare_action_focused_sft import serialize
from workflow_distillation import weighted_sft_data as data


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("data/raw/qwen35_4b_tokenizer", local_files_only=True)


def record(tokenizer):
    tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }

    def call(command):
        return {"type": "function", "function": {"name": "bash", "arguments": {"command": command}}}

    messages = [
        {"role": "system", "content": "Use Bash. Submit with the completion marker."},
        {"role": "user", "content": "Fix café. Literal <tool_call> in task text."},
        {
            "role": "assistant",
            "reasoning_content": "A fake <tool_call> and </think> here are reasoning.",
            "content": "Read the relevant file.",
            "tool_calls": [call("printf 'café\\n'\nprintf \"quoted\\n\""), call("cat src.py")],
        },
        {
            "role": "tool",
            "content": "Observation: <tool_call> and COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        },
        {"role": "tool", "content": "File contents."},
        {
            "role": "assistant",
            "reasoning_content": "The change is verified.",
            "content": "Submit the patch.",
            "tool_calls": [call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")],
        },
    ]
    return {
        "trajectory_id": "example",
        "task_key": "example/repo#1",
        "messages": messages,
        "tools": [tool],
        **serialize(messages, [tool], tokenizer),
    }


def group_for(result, tokenizer, phrase):
    text = tokenizer.decode(result["input_ids"], skip_special_tokens=False)
    start = text.index(phrase)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return {
        int(group)
        for group, (left, right) in zip(result["groups"], encoded["offset_mapping"], strict=True)
        if left < start + len(phrase) and right > start
    }


def test_regions_follow_roles_not_literal_tags(tokenizer):
    source = record(tokenizer)
    result = data.annotate_record(source, tokenizer)
    assert result["input_ids"] == tokenizer.encode(source["text"], add_special_tokens=False)
    assert group_for(result, tokenizer, "A fake <tool_call> and </think>") == {1}
    assert group_for(result, tokenizer, "Read the relevant file.") == {2}
    assert group_for(result, tokenizer, "Observation: <tool_call>") == {0}
    assert group_for(result, tokenizer, "Literal <tool_call> in task text.") == {0}
    assert group_for(result, tokenizer, "printf 'café") == {3}
    assert group_for(result, tokenizer, "cat src.py") == {3}
    assert group_for(
        result, tokenizer, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
    ) == {4}
    assert group_for(result, tokenizer, "<|im_start|>assistant\n<think>\n") == {0}
    assert group_for(result, tokenizer, "Submit the patch.") == {2}
    assert result["groups"][-2] == 4  # End-of-message token after terminal call.
    counts = result["group_counts"]
    assert result["denominator"] == counts[1] + 0.75 * counts[2] + 1.5 * counts[3] + 2 * counts[4]


def test_history_reasoning_omission_is_recorded(tokenizer):
    source = record(tokenizer)
    source["messages"].insert(5, {"role": "user", "content": "Continue with this clarification."})
    source.update(serialize(source["messages"], source["tools"], tokenizer))
    result = data.annotate_record(source, tokenizer)
    assert result["omitted_reasoning_messages"] == [2]
    assert group_for(result, tokenizer, "Read the relevant file.") == {2}
    assert group_for(result, tokenizer, "The change is verified.") == {1}


def test_empty_reasoning_masks_token_merged_with_supplied_prefix(tokenizer):
    source = record(tokenizer)
    source["messages"][2]["reasoning_content"] = ""
    source.update(serialize(source["messages"], source["tools"], tokenizer))
    result = data.annotate_record(source, tokenizer)
    assert result["input_ids"] == tokenizer.encode(source["text"], add_special_tokens=False)
    assert result["boundary_masked_tokens"] >= 1
    assert group_for(result, tokenizer, "<|im_start|>assistant\n<think>\n") == {0}
    assert group_for(result, tokenizer, "cat src.py") == {3}


def test_rejects_payload_that_does_not_match_official_render(tokenizer):
    source = copy.deepcopy(record(tokenizer))
    source["text"] += "tampered"
    with pytest.raises(ValueError, match="render"):
        data.annotate_record(source, tokenizer)


def test_window_plan_supervises_each_target_once_and_keeps_predecessor():
    groups = [0, 0, 1, 1, 3, 0, 0, 2, 4, 4, 4, 4, 0]
    windows = data.plan_windows(groups, max_length=6, target_length=3)
    covered = []
    for window in windows:
        start, target_start, end = window
        assert end - start <= 6
        assert start < target_start
        covered.extend(i for i in range(target_start, end) if groups[i])
    assert covered == [2, 3, 4, 7, 8, 9, 10, 11]


def test_full_history_policy_rejects_overflow_instead_of_shortening():
    with pytest.raises(ValueError, match="full.history"):
        data.plan_windows([0] + [1] * 20, max_length=8, target_length=None)
    assert data.plan_windows([0, 1, 3, 4], max_length=8, target_length=None) == [(0, 1, 4)]
