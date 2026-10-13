import copy
import json

import pytest

from scripts.data import prepare_action_focused_sft as p


def example():
    command = 'printf "xin chào\\n"\ncat <<\'EOF\'\n"quoted" $HOME\nEOF'
    tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return {
        "tools": [json.dumps(tool)],
        "messages": [
            {"role": "system", "content": "Use Bash."},
            {"role": "user", "content": "Fix it."},
            {
                "role": "assistant",
                "content": "Running",
                "reasoning_content": command,
                "tool_calls": [call],
            },
            {"role": "tool", "content": command},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        **copy.deepcopy(call),
                        "id": "c2",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
                            ),
                        },
                    }
                ],
            },
        ],
        "reference_patch": "SECRET",
    }


def test_normalize_preserves_commands_reasoning_and_terminal_action():
    row = example()
    original = copy.deepcopy(row)
    messages, tools, checks = p.normalize(row)
    assert (
        messages[2]["tool_calls"][0]["function"]["arguments"]["command"]
        == (json.loads(row["messages"][2]["tool_calls"][0]["function"]["arguments"])["command"])
    )
    assert messages[2]["reasoning_content"] == row["messages"][2]["reasoning_content"]
    assert tools[0]["function"]["name"] == "bash"
    assert any(c["status"] == "unverified" for c in checks)
    assert row == original


def test_terminal_submission_after_cd_is_preserved():
    row = example()
    command = "cd /testbed && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
    row["messages"][-1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": command})
    messages, _, checks = p.normalize(row)
    assert messages[-1]["tool_calls"][0]["function"]["arguments"]["command"] == command
    assert checks[-1]["kind"] == "terminal_observation"


def test_source_prefetch_caches_rows_without_patches(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    config = {"dataset": {"revision": "test"}, "local_paths": {"raw": str(tmp_path)}}
    source = p.Source(config, offline=True)
    table = pa.Table.from_pylist(
        [{"trajectory_id": "one", **example()}, {"trajectory_id": "two", **example()}]
    ).select(["trajectory_id", "messages", "tools"])
    group_path = source.cache / "shard.group0.parquet"
    pq.write_table(table, group_path)
    rows = [
        {"trajectory_id": tid, "shard": "shard", "row_group": 0, "row_offset": index}
        for index, tid in enumerate(("one", "two"))
    ]
    source.plan_prefetch(rows)
    source.warm()
    monkeypatch.setattr(source, "group", lambda *args: pytest.fail("Group read twice"))
    assert source.load(rows[1])["trajectory_id"] == "two"
    assert "reference_patch" not in source.load(rows[0])


def test_multiple_calls_and_schema_errors_are_logged_without_rewriting():
    row = example()
    bad = copy.deepcopy(row["messages"][2]["tool_calls"][0])
    bad["id"] = "bad"
    bad["function"]["arguments"] = '{"command": 123}'
    row["messages"][2]["tool_calls"].append(bad)
    row["messages"].insert(4, {"role": "tool", "content": "error"})
    messages, _, checks = p.normalize(row)
    assert messages[2]["tool_calls"][1]["function"]["arguments"]["command"] == 123
    assert any(c["status"] == "invalid" and c["kind"] == "arguments" for c in checks)


def test_invalid_json_or_sequence_rejected():
    row = example()
    row["messages"][2]["tool_calls"][0]["function"]["arguments"] = '"{\\"command\\":\\"ls\\"}"'
    with pytest.raises(ValueError, match="object"):
        p.normalize(row)
    row = example()
    row["messages"].insert(2, {"role": "tool", "content": "orphan"})
    with pytest.raises(ValueError, match="orphan"):
        p.normalize(row)


@pytest.mark.parametrize("instance", ["Owner__Repo-0012", "owner__repo_12", "owner_repo_pr12"])
def test_canonical_key(instance):
    assert p.task_key("Owner/Repo", instance) == "owner/repo#12"


def test_selection_reproducible_balanced_and_task_disjoint():
    rows = [
        {
            "task_key": f"r/{repo}#{i}",
            "repo": f"r/{repo}",
            "category": category,
            "trajectory_id": f"{category}-{repo}-{i}",
        }
        for category in p.CATEGORIES
        for repo in range(3)
        for i in range(6)
    ]
    first = p.select(rows, lambda r: r, train_quota=3, validation_quota=1)
    second = p.select(list(reversed(rows)), lambda r: r, train_quota=3, validation_quota=1)
    # Tasks shared across categories must never be selected twice.
    assert first == second
    selected = first["train"] + first["validation"]
    assert len({r["task_key"] for r in selected}) == len(selected)
    for category in p.CATEGORIES:
        assert sum(r["category"] == category for r in first["train"]) == 3


def test_shortage_fails_and_bad_candidate_replaced():
    rows = [
        {"task_key": f"r/x#{cat}{i}", "repo": "r/x", "category": cat, "trajectory_id": f"{cat}-{i}"}
        for cat in p.CATEGORIES
        for i in range(4)
    ]

    def load(row):
        if row["trajectory_id"].endswith("-0"):
            raise ValueError("bad record")
        return row

    result = p.select(rows, load, train_quota=2, validation_quota=1)
    assert len(result["train"]) == 8
    with pytest.raises(p.QuotaError):
        p.select(rows, load, train_quota=4, validation_quota=1)


def test_official_template_and_action_spans(tokenizer):
    row = example()
    messages, tools, _ = p.normalize(row)
    rendered = p.serialize(messages, tools, tokenizer)
    assert rendered["text"] == tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    assert rendered["token_count"] == len(
        tokenizer.encode(rendered["text"], add_special_tokens=False)
    )
    assert len(rendered["action_candidates"]) == 2
    for region in rendered["action_candidates"]:
        start, end = region["char_span"]
        assert rendered["text"][start:end].startswith("<tool_call>")
        assert rendered["text"][start:end].endswith("</tool_call>")
        assert region["message_index"] in (2, 4)


def test_action_spans_ignore_fake_tags_and_support_multiple_calls(tokenizer):
    row = example()
    fake = "<tool_call>\n<function=bash>\n</tool_call>"
    row["messages"][2]["reasoning_content"] = fake
    row["messages"][3]["content"] = fake
    call = copy.deepcopy(row["messages"][2]["tool_calls"][0])
    call["id"] = "extra"
    call["function"]["arguments"] = json.dumps({"command": "printf '</tool_call>'"})
    row["messages"][2]["tool_calls"].append(call)
    row["messages"].insert(4, {"role": "tool", "content": "done"})
    messages, tools, _ = p.normalize(row)
    rendered = p.serialize(messages, tools, tokenizer)
    assert len(rendered["action_candidates"]) == 3
    start, end = rendered["action_candidates"][1]["char_span"]
    assert "printf '</tool_call>'" in rendered["text"][start:end]
    assert rendered["text"][start:end].endswith("</function>\n</tool_call>")


def test_pool_keeps_unsuccessful_validation_attempts_out_of_phase2():
    rows = [
        {"task_key": "a/b#1", "trajectory_id": "v1", "resolved": 1},
        {"task_key": "a/b#1", "trajectory_id": "v2", "resolved": 0},
        {"task_key": "a/b#2", "trajectory_id": "t1", "resolved": 1},
    ]
    pool = p.partition_pool(rows, {"a/b#1": "validation"}, {"v1"}, set())
    assert [r["partition"] for r in pool] == ["validation", "validation", "phase2_candidate"]
    assert [r["selected_phase1"] for r in pool] == [True, False, False]


def test_operation_report_is_descriptive_only():
    assert p.operation_type("pytest -q") == "test"
    assert p.operation_type("git diff") == "git"
    assert p.operation_type("$EXEC script.sh") == "dynamic_or_other"
    assert (
        p.operation_type("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt") == "submit"
    )


def test_artifact_integrity_rejects_tampering(tmp_path):
    (tmp_path / "train.jsonl").write_text("original\n")
    hashes = {"train.jsonl": p.digest("original\n")}
    p.verify_hashes(tmp_path, hashes)
    (tmp_path / "train.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        p.verify_hashes(tmp_path, hashes)


@pytest.mark.parametrize("kind", ["message", "call", "schema", "late_system", "no_user"])
def test_malformed_candidate_is_logged_and_replaced(kind, tokenizer):
    row = example()
    if kind == "message":
        row["messages"][2] = "bad"
    elif kind == "call":
        row["messages"][2]["tool_calls"] = ["bad"]
    elif kind == "schema":
        tool = json.loads(row["tools"][0])
        tool["function"]["parameters"]["type"] = "not_a_json_type"
        row["tools"][0] = json.dumps(tool)
    elif kind == "late_system":
        row["messages"].insert(2, {"role": "system", "content": "bad"})
    else:
        row["messages"][1]["role"] = "assistant"
    candidates = [
        {"task_key": f"a/b#{cat}{i}", "repo": "a/b", "category": cat, "trajectory_id": f"{cat}-{i}"}
        for cat in p.CATEGORIES
        for i in range(3)
    ]
    seen = set()

    def load(candidate):
        category = candidate["category"]
        if category not in seen:
            seen.add(category)
            messages, tools, _ = p.normalize(row)
            p.serialize(messages, tools, tokenizer)
            pytest.fail("Malformed row was accepted")
        return candidate

    result = p.select(candidates, load, train_quota=1, validation_quota=1)
    assert len(result["train"]) == 4
    assert len(result["errors"]) == 4


@pytest.fixture
def tokenizer():
    from pathlib import Path

    from transformers import AutoTokenizer

    path = Path("data/raw/qwen35_4b_tokenizer")
    if not path.exists():
        pytest.skip("Official tokenizer not downloaded yet")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_split_heldout_preserves_tasks_and_is_deterministic():
    rows = [
        {"category": category, "task_key": f"repo/{i}#{ci}", "trajectory_id": f"{ci}-{i}"}
        for ci, category in enumerate(p.CATEGORIES)
        for i in range(25)
    ]
    phase = {
        "seed": 42,
        "validation_per_category": dict(zip(p.CATEGORIES, [13, 13, 12, 12], strict=True)),
        "test_per_category": dict(zip(p.CATEGORIES, [12, 12, 13, 13], strict=True)),
    }
    first = p.split_heldout({"train": [], "validation": rows, "errors": []}, phase)
    second = p.split_heldout({"train": [], "validation": rows[::-1], "errors": []}, phase)
    assert first == second
    assert len(first["validation"]) == len(first["test"]) == 50
    valid = {r["task_key"] for r in first["validation"]}
    test = {r["task_key"] for r in first["test"]}
    assert not valid & test
    assert valid | test == {r["task_key"] for r in rows}
    pool = p.partition_pool(rows, dict.fromkeys(test, "test"), set(), set())
    assert all(r["partition"] == "test" for r in pool if r["task_key"] in test)


def test_verify_selection_rejects_test_leakage():
    phase = {
        "train_per_category": 1,
        "validation_per_category": dict.fromkeys(p.CATEGORIES, 1),
        "test_per_category": dict.fromkeys(p.CATEGORIES, 1),
    }
    result = {
        split: [{"category": c, "task_key": f"{split}/{c}"} for c in p.CATEGORIES]
        for split in ("train", "validation", "test")
    }
    p.verify_selection(result, set(), {"phase1": phase})
    result["test"][0]["task_key"] = result["validation"][0]["task_key"]
    with pytest.raises(ValueError, match="leakage"):
        p.verify_selection(result, set(), {"phase1": phase})


def test_keep_validation_and_release_former_test_tasks():
    rows = [
        {"category": category, "task_key": f"repo/{i}#{ci}"}
        for ci, category in enumerate(p.CATEGORIES)
        for i in range(25)
    ]
    old = {
        "seed": 42,
        "validation_per_category": dict(zip(p.CATEGORIES, [13, 13, 12, 12], strict=True)),
        "test_per_category": dict(zip(p.CATEGORIES, [12, 12, 13, 13], strict=True)),
    }
    old_result = p.split_heldout({"train": [], "validation": rows}, old)
    new = {k: v for k, v in old.items() if k != "test_per_category"}
    new["heldout_per_category"] = 25
    new_result = p.split_heldout({"train": [], "validation": rows}, new)
    assert new_result["validation"] == old_result["validation"]
    assert "test" not in new_result
    assert p.split_names(new) == ("train", "validation")
