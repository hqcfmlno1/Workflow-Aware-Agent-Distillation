from scripts.data import export_success_trajectories as e


def test_success_filter_preserves_attempts_and_protects_heldout_tasks():
    def row(tid, number, resolved=1, thinking=True):
        return {
            "repo": "Owner/Repo",
            "instance_id": f"owner__repo-{number}",
            "trajectory_id": tid,
            "resolved": resolved,
            "teacher_model": {"name": "Qwen3.8-27B", "enable_thinking": thinking},
        }

    rows = [
        row("a", 1),
        row("b", 1),
        row("c", 2),
        row("d", 3),
        row("e", 4),
        row("f", 5, resolved=0),
        row("g", 6, thinking=False),
    ]
    selected = e.success_rows(
        rows,
        "Qwen3.8-27B",
        {"owner/repo#3"},
        {"owner/repo#1": "validation", "owner/repo#2": "test"},
    )
    assert [r["trajectory_id"] for r in selected] == ["a", "b", "c", "d", "e"]
    assert [r["partition"] for r in selected] == [
        "validation",
        "validation",
        "test",
        "excluded_omnicode",
        "train_pool",
    ]


def test_export_group_preserves_raw_payload(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        [
            {"trajectory_id": "failed", "messages": "unused", "tools": "[]"},
            {
                "trajectory_id": "passed",
                "messages": 'Unicode: tiếng Việt\n"quotes"\\n',
                "tools": '["{\\"arguments\\": \\"x\\"}"]',
            },
        ]
    )
    rows = [{"trajectory_id": "passed", "row_offset": 1, "partition": "train_pool"}]
    output = e.project_success(table, rows)
    pq.write_table(output, tmp_path / "part.parquet")
    saved = pq.read_table(tmp_path / "part.parquet")
    assert saved.select(table.column_names).equals(table.take(pa.array([1])))
    assert saved["partition"].to_pylist() == ["train_pool"]
