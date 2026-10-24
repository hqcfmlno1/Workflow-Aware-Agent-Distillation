import copy
import json

import numpy as np
import pytest


def candidates():
    return [
        dict(
            trajectory_id=f"{category}-{i}-{j}",
            task_key=f"{category}-{i}",
            repo=f"repo-{i}",
            language=["python", "go", "ts"][i % 3],
            category=category,
            total_tokens=10000 + i * 500,
            tqs_score=0.61,
        )
        for category, count in [
            ("bug-fix", 15),
            ("feature-request", 10),
            ("enhancement", 5),
            ("other", 5),
        ]
        for i in range(count)
        for j in range(2)
    ]


def test_split_is_fixed_task_disjoint_and_excludes_sibling_attempts():
    from workflow_distillation.lora_fp16.data import select_split

    rows = candidates()
    result = select_split(rows, seed=42)
    assert result == select_split(list(reversed(rows)), seed=42)
    held = result["validation_task_keys"]
    assert len(held) == 20
    assert len(result["validation_ids"]) == 20
    assert len(result["excluded_sibling_ids"]) == 20
    by_id = {r["trajectory_id"]: r for r in rows}
    assert not {by_id[t]["task_key"] for t in result["train_ids"]} & set(held)
    assert result["validation_categories"] == {
        "bug-fix": 10,
        "feature-request": 6,
        "enhancement": 2,
        "other": 2,
    }
    assert set(
        result["train_ids"] + result["validation_ids"] + result["excluded_sibling_ids"]
    ) == set(by_id)


def test_split_rejects_duplicate_ids_and_incomplete_category():
    from workflow_distillation.lora_fp16.data import select_split

    rows = candidates()
    with pytest.raises(ValueError):
        select_split(rows + [copy.copy(rows[0])])
    with pytest.raises(ValueError):
        select_split(rows[:2])


def test_portable_bundle_preserves_full_ids_groups_and_rejects_corruption(tmp_path):
    from workflow_distillation.epoch_data import EpochDataset, sha256_file
    from workflow_distillation.lora_fp16.data import load_bundle, write_bundle

    source = tmp_path / "source"
    source.mkdir()
    ids = np.array([1, 2, 3, 4, 5, 6], dtype="<u4")
    groups = np.array([0, 1, 3, 0, 2, 4], dtype="u1")
    ids.tofile(source / "tokens.bin")
    groups.tofile(source / "groups.bin")
    records = [
        dict(
            trajectory_id=str(i),
            task_key=str(i),
            offset=3 * i,
            length=3,
            full_token_count=3,
            denominator=[2.5, 2.75][i],
            group_counts=([1, 1, 0, 1, 0] if i == 0 else [1, 0, 1, 0, 1]),
        )
        for i in range(2)
    ]
    index = {
        "format_version": 1,
        "tokens": 6,
        "records": 2,
        "data": records,
        "weights": [0, 1, 0.75, 1.5, 2],
        "file_sha256": {n: sha256_file(source / n) for n in ["tokens.bin", "groups.bin"]},
    }
    (source / "index.json").write_text(json.dumps(index))
    split = dict(
        train_ids=["1"], validation_ids=["0"], excluded_sibling_ids=[], validation_task_keys=["0"]
    )
    dest = tmp_path / "bundle"
    write_bundle(EpochDataset(source / "index.json"), split, dest)
    dataset, actual = load_bundle(dest / "index.json")
    assert actual == split
    assert dataset.record(0)["input_ids"].tolist() == [4, 5, 6]
    assert dataset.record(1)["groups"].tolist() == [0, 1, 3]
    assert dataset.record(1)["full_token_count"] == 3
    (dest / "groups.bin").write_bytes(bytes(6))
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(dest / "index.json")
