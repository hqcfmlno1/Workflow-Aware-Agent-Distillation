import copy
import json

import pytest

from scripts.data import select_rule_trajectories as s


def record(commands=None, observations=None, **extra):
    commands = commands or ["ls", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    observations = observations if observations is not None else [{"returncode": 0, "output": "ok"}]
    messages = [{"role": "user", "content": "fix bug"}]
    for i, command in enumerate(commands):
        messages.append(
            {
                "role": "assistant",
                "reasoning_content": "think",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"call{i}",
                        "function": {"name": "bash", "arguments": {"command": command}},
                    }
                ],
            }
        )
        if i < len(observations):
            obs = observations[i]
            messages.append(
                {"role": "tool", "content": obs if isinstance(obs, str) else json.dumps(obs)}
            )
    return {
        "trajectory_id": "a",
        "task_key": "a/b#1",
        "repo": "a/b",
        "category": "bug-fix",
        "resolved": 1,
        "partition": "train_pool",
        "serialization_status": "ok",
        "checks": [{"kind": "observation_link", "status": "unverified"}],
        "token_count": 100,
        "messages": messages,
        **extra,
    }


def test_shell_errors_require_returncode_and_shell_diagnostic():
    assert s.shell_error({"returncode": 127, "output": "bash: line 1: rg: command not found"})
    assert s.shell_error(
        {"returncode": 2, "output": "bash: -c: line 1: syntax error near unexpected token `)'"}
    )
    assert s.shell_error(
        {"returncode": 127, "output": r"previous output\nbash: line 1: rg: command not found\n"}
    )
    for obs in [
        {"returncode": 0, "output": "bash: line 1: rg: command not found"},
        {"returncode": 1, "output": "1 failed in pytest"},
        {"returncode": 1, "output": ""},
        {"returncode": 2, "output": "pytest collection error"},
        {"returncode": 127, "output": "tests mention command not found"},
    ]:
        assert not s.shell_error(obs)


def test_duplicates_preserve_commands_and_do_not_penalize_test_after_edit():
    row = record(
        ["pytest", "pytest", "edit", "pytest"], [{"returncode": 1, "output": "failed"}] * 4
    )
    original = copy.deepcopy(row)
    features = s.features(row)
    assert features["redundant_repeats"] == 1
    assert features["bash_usage_errors"] == 0
    assert features["other_nonzero_observations"] == 4
    assert row == original
    row["messages"][3]["tool_calls"][0]["function"]["arguments"]["command"] = "pytest "
    assert s.features(row)["redundant_repeats"] == 0


def test_multiple_calls_repeat_block_and_terminal_missing_observation():
    row = record()
    assert s.features(row)["unverified_observations"] == 0
    assert s.features(row)["tool_call_count"] == 2
    calls = row["messages"][1]["tool_calls"]
    calls.append(
        {"type": "function", "function": {"name": "bash", "arguments": {"command": "pwd"}}}
    )
    row["messages"].insert(3, {"role": "tool", "content": '{"returncode":0,"output":"/repo"}'})
    assert s.features(row)["tool_call_count"] == 3
    assert s.features(row)["unverified_observations"] == 0


def test_unknown_observation_is_not_error_and_invalid_schema_is_rejected():
    row = record(observations=["not JSON"])
    assert s.features(row)["unverified_observations"] == 1
    assert s.features(row)["bash_usage_errors"] == 0
    row["checks"].append({"kind": "arguments", "status": "invalid"})
    assert "invalid_tool_arguments" in s.features(row)["exclusion_reasons"]
    assert "partition:validation" in s.features(record(partition="validation"))["exclusion_reasons"]


def candidates():
    rows = []
    for ci, category in enumerate(s.CATEGORIES):
        for task in range(10 + ci):
            for attempt in range(2):
                rows.append(
                    {
                        **s.features(record()),
                        "category": category,
                        "repo": f"owner/r{task % 3}",
                        "task_key": f"owner/r{task % 3}#{ci * 100 + task}",
                        "trajectory_id": f"{ci}-{task}-{attempt}",
                        "token_count": 100 + attempt,
                    }
                )
    return rows


def test_selection_reproducible_proportional_and_prefers_quality_over_length():
    rows = candidates()
    rows[0]["bash_usage_errors"] = 1
    first = s.select(rows, total=20, seed=42)
    assert first == s.select(rows[::-1], total=20, seed=42)
    assert len(first["selection"]) == len({r["task_key"] for r in first["selection"]}) == 20
    assert sum(first["quotas"].values()) == 20
    winner = next(r for r in first["winners"] if r["task_key"] == rows[0]["task_key"])
    assert winner["trajectory_id"] == rows[1]["trajectory_id"]
    assert first["quotas"] == s.proportional_quotas(
        {c: 10 + i for i, c in enumerate(s.CATEGORIES)}, 20
    )


def test_singletons_excluded_but_two_successes_one_valid_candidate_allowed():
    rows = candidates()
    rows[0]["exclusion_reasons"] = ["invalid_tool_arguments"]
    result = s.select(rows, total=20)
    assert any(r["trajectory_id"] == rows[1]["trajectory_id"] for r in result["winners"])
    singleton = {
        **rows[0],
        "task_key": "single/task#1",
        "trajectory_id": "singleton",
        "exclusion_reasons": [],
    }
    result = s.select(rows + [singleton], total=20)
    assert not any(r["task_key"] == singleton["task_key"] for r in result["winners"])
    with pytest.raises(ValueError, match="Insufficient"):
        s.select(rows, total=1000)


def test_inconsistent_task_partition_fails():
    rows = candidates()
    rows[1]["partition"] = "validation"
    with pytest.raises(ValueError, match="partition"):
        s.select(rows, total=20)


def test_pipeline_copies_payloads_and_detects_tampering(tmp_path):
    import gzip

    from scripts.data.export_success_trajectories import sha256_file
    from scripts.data.prepare_action_focused_sft import write_json

    source, cache, output = (tmp_path / name for name in ("source", "cache", "output"))
    source.mkdir()
    all_rows = []
    for ci, category in enumerate(s.CATEGORIES):
        for task in range(3):
            for attempt in range(2):
                all_rows.append(
                    record(
                        category=category,
                        task_key=f"owner/repo{task}#{ci * 10 + task}",
                        trajectory_id=f"{ci}-{task}-{attempt}",
                        token_count=100 + attempt,
                        text="official template text\nUnicode: tiếng Việt",
                    )
                )
    heldout = record(partition="validation", task_key="held/out#1", trajectory_id="heldout")
    artifacts = {}
    for part, rows in enumerate((all_rows[::2] + [heldout], all_rows[1::2])):
        name = f"part-{part:05d}.jsonl.gz"
        with gzip.open(source / name, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        artifacts[name] = {"sha256": sha256_file(source / name), "rows": len(rows)}
    write_json(
        source / "report.json", {"artifacts": artifacts, "rows": 25, "tokenizer": {}, "dataset": {}}
    )
    write_json(
        source / "verification_report.json",
        {
            "report_sha256": sha256_file(source / "report.json"),
            "rows": 25,
            "all_source_payloads_and_official_renders_verified": True,
        },
    )
    s.run(source, output, cache, total=8, seed=42, workers=1)
    manifest = json.loads((output / "task_selection_manifest.json").read_text())
    assert len(manifest["selection"]) == 8
    assert all(r["trajectory_id"].endswith("-0") for r in manifest["selection"])
    s.run(source, output, cache, total=8, seed=42, workers=1, verify=True)
    assert json.loads((output / "verification_report.json").read_text())["rows"] == 8
    before = [r["trajectory_id"] for r in manifest["selection"]]
    s.run(source, output, cache, total=8, seed=42, workers=1)
    assert before == [
        r["trajectory_id"]
        for r in json.loads((output / "task_selection_manifest.json").read_text())["selection"]
    ]
    records = []
    for part in output.glob("part-*.jsonl.gz"):
        with gzip.open(part, "rt", encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream)
    originals = {r["trajectory_id"]: r for r in all_rows}
    assert all(r == originals[r["trajectory_id"]] for r in records)
    cache_path = cache / "part-00001.features.json"
    saved_cache = json.loads(cache_path.read_text())
    changed = copy.deepcopy(saved_cache)
    competitor = next(r for r in changed["rows"] if r["trajectory_id"].endswith("-1"))
    competitor["bash_usage_errors"] = 999
    changed["features_sha256"] = s.rows_digest(changed["rows"])
    cache_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="[Cc]ache.*source"):
        s.run(source, output, cache, total=8, seed=42, workers=1, verify=True)
    cache_path.write_text(json.dumps(saved_cache))
    report_path = output / "report.json"
    saved_report = json.loads(report_path.read_text())
    for field in ("quotas", "eligible_tasks", "repositories", "languages", "metric_totals"):
        changed_report = copy.deepcopy(saved_report)
        changed_report[field] = {"tampered": 999}
        report_path.write_text(json.dumps(changed_report))
        with pytest.raises(ValueError, match="report"):
            s.run(source, output, cache, total=8, seed=42, workers=1, verify=True)
    report_path.write_text(json.dumps(saved_report))
    target = next(output.glob("part-*.jsonl.gz"))
    with target.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        s.run(source, output, cache, total=8, seed=42, workers=1, verify=True)
