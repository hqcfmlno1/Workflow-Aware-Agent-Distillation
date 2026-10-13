import json

import pytest

from scripts.data.convert_success_trajectories import convert_record
from tests.test_action_focused_sft import example
from tests.test_action_focused_sft import tokenizer as tokenizer_fixture


@pytest.fixture
def tokenizer():
    return tokenizer_fixture.__wrapped__()


def test_full_conversion_keeps_payload_and_partition(tokenizer):
    row = {**example(), "trajectory_id": "one", "task_key": "a/b#1", "partition": "validation"}
    result = convert_record(row, tokenizer)
    assert result["partition"] == "validation"
    assert result["serialization_status"] == "ok"
    assert "reference_patch" not in result
    assert result["messages"][2]["reasoning_content"] == row["messages"][2]["reasoning_content"]
    assert result["text"] == tokenizer.apply_chat_template(
        result["messages"], tools=result["tools"], tokenize=False, add_generation_prompt=False
    )


def test_failed_conversion_keeps_original_and_reports_error(tokenizer):
    row = {**example(), "trajectory_id": "bad"}
    row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "invalid JSON"
    result = convert_record(row, tokenizer)
    assert result["serialization_status"] == "error"
    assert result["raw_messages"] == row["messages"]
    assert result["text"] is None
    assert result["error"]
    json.dumps(result)


def test_shard_roundtrip_resume_and_verification(tmp_path, tokenizer, monkeypatch):
    import gzip

    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.data import convert_success_trajectories as c

    rows = []
    for index, partition in enumerate(("train_pool", "validation", "excluded_omnicode")):
        row = {
            **example(),
            "trajectory_id": str(index),
            "task_key": f"a/b#{index}",
            "partition": partition,
        }
        # Match the source Arrow schema (arguments are JSON strings).
        row.pop("reference_patch")
        if index == 2:
            row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "invalid JSON"
        rows.append(row)
    source = tmp_path / "source.parquet"
    target = tmp_path / "part.jsonl.gz"
    pq.write_table(pa.Table.from_pylist(rows), source)
    monkeypatch.setattr(c, "_TOKENIZER", tokenizer)
    identity = {"test": True}
    receipt = c.convert_part(source, target, identity)
    assert receipt["status"] == {"ok": 2, "error": 1}
    assert c.convert_part(source, target, identity) == receipt
    counts, partitions, ids = c.verify_part(source, target, identity)
    assert counts == receipt["status"]
    assert partitions == receipt["partitions"]
    assert ids == ["0", "1", "2"]
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        result = [json.loads(line) for line in stream]
    assert result[2]["raw_messages"] == pq.read_table(source).to_pylist()[2]["messages"]
    with pytest.raises(ValueError, match="mismatch"):
        c.convert_part(source, target, {"different": True})
    with target.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        c.verify_part(source, target, identity)


@pytest.mark.parametrize("kind", ["invalid_schema", "no_user"])
def test_schema_and_template_failures_are_retained(kind, tokenizer):
    row = {**example(), "trajectory_id": "bad"}
    if kind == "invalid_schema":
        tool = json.loads(row["tools"][0])
        tool["function"]["parameters"]["type"] = "invalid-type"
        row["tools"] = [json.dumps(tool)]
    else:
        row["messages"][1]["role"] = "assistant"
    result = convert_record(row, tokenizer)
    assert result["serialization_status"] == "error"
    assert result["raw_messages"] == row["messages"]
    assert result["raw_tools"] == row["tools"]
