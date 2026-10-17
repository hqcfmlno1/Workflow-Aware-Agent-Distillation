"""One-epoch data selection and compact token storage."""

import copy
import gzip
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from scripts.data.prepare_action_focused_sft import serialize
from scripts.data.prepare_epoch_sft import eligible_record, prepare
from workflow_distillation.epoch_data import EpochDataset, epoch_batches


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("data/raw/qwen35_4b_tokenizer", local_files_only=True)


def _record(tokenizer, trajectory_id, task_key):
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
    messages = [
        {"role": "system", "content": "Use Bash."},
        {"role": "user", "content": "Fix the issue."},
        {
            "role": "assistant",
            "reasoning_content": "Inspect.",
            "content": "Run command.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "cat file"}},
                }
            ],
        },
        {"role": "tool", "content": "file content"},
        {
            "role": "assistant",
            "reasoning_content": "Done.",
            "content": "Submit.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
                    },
                }
            ],
        },
    ]
    return {
        "trajectory_id": trajectory_id,
        "task_key": task_key,
        "partition": "train_pool",
        "resolved": 1,
        "messages": messages,
        "tools": [tool],
        **serialize(messages, [tool], tokenizer),
    }


def _source(tmp_path, rows, tokenizer):
    source = tmp_path / "source"
    source.mkdir()
    part = source / "part-00000.jsonl.gz"
    with gzip.open(part, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    import hashlib

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "tokenizer": {
            "id": "Qwen/Qwen3.5-4B",
            "revision": "pinned",
            "template_sha256": sha(Path("data/raw/qwen35_4b_tokenizer/chat_template.jinja")),
        },
        "artifacts": {part.name: {"sha256": sha(part), "rows": len(rows)}},
    }
    (source / "report.json").write_text(json.dumps(report))
    return source


def test_eligibility_uses_full_success_pool_and_strict_token_limit():
    base = {"partition": "train_pool", "resolved": 1, "token_count": 29999}
    assert eligible_record(base)
    assert not eligible_record({**base, "token_count": 30000})
    assert not eligible_record({**base, "partition": "validation"})
    assert not eligible_record({**base, "resolved": 0})


def test_prepare_preserves_distinct_trajectories_for_same_task_and_round_trips(tmp_path, tokenizer):
    a = _record(tokenizer, "a", "same-task")
    b = _record(tokenizer, "b", "same-task")
    excluded = copy.deepcopy(a)
    excluded["trajectory_id"] = "val"
    excluded["partition"] = "validation"
    long = copy.deepcopy(a)
    long["trajectory_id"] = "long"
    long["token_count"] = 30000
    source = _source(tmp_path, [a, excluded, b, long], tokenizer)
    out = tmp_path / "out"
    prepare(source, out, Path("data/raw/qwen35_4b_tokenizer"))
    ds = EpochDataset(out / "index.json")
    assert [row["trajectory_id"] for row in ds.records] == ["a", "b"]
    assert [row["task_key"] for row in ds.records] == ["same-task", "same-task"]
    assert ds.manifest["records"] == 2
    assert ds.manifest["source_report_sha256"]
    for i, original in enumerate((a, b)):
        result = ds.record(i)
        assert result["input_ids"].tolist() == tokenizer.encode(
            original["text"], add_special_tokens=False
        )
        assert result["groups"][0] == 0
        assert result["full_token_count"] == original["token_count"]
        assert sum(result["groups"] == 4) > 0
        assert result["denominator"] > 0
    EpochDataset(out / "index.json", verify=False)
    with pytest.raises(FileExistsError):
        prepare(source, out, Path("data/raw/qwen35_4b_tokenizer"))


def test_annotation_failure_does_not_publish_index(tmp_path, tokenizer):
    bad = _record(tokenizer, "bad", "task")
    bad["text"] += "tampered"
    source = _source(tmp_path, [bad], tokenizer)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="render"):
        prepare(source, out, Path("data/raw/qwen35_4b_tokenizer"))
    assert not (out / "index.json").exists()


def test_reader_rejects_binary_corruption(tmp_path, tokenizer):
    source = _source(tmp_path, [_record(tokenizer, "a", "task")], tokenizer)
    out = tmp_path / "out"
    prepare(source, out, Path("data/raw/qwen35_4b_tokenizer"))
    with (out / "groups.bin").open("r+b") as stream:
        stream.seek(4)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        EpochDataset(out / "index.json")


def test_epoch_batches_cover_each_index_once_and_are_deterministic():
    batches = epoch_batches(5, seed=42)
    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert sorted(i for batch in batches for i in batch) == list(range(5))
    assert batches == epoch_batches(5, seed=42)
    assert epoch_batches(0) == []
