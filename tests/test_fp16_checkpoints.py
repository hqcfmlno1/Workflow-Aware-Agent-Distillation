import json

from test_training_checkpoints import fake_hub, save, setup_training, step  # noqa: F401


def test_hub_retention_preserves_epoch_boundaries_beside_recent_recovery(
    tmp_path, fake_hub  # noqa: F811
):
    from workflow_distillation.lora_fp16.checkpoints import CheckpointManager, HuggingFacePublisher

    training = setup_training()
    publisher = HuggingFacePublisher(
        "owner/dedicated-fp16",
        prefix="fp16",
        keep_last=2,
        pinned_steps=[1, 3],
    )
    manager = CheckpointManager(
        tmp_path / "source",
        {},
        keep_last=2,
        pinned_steps=[1, 3],
        publisher=publisher,
    )
    for number in range(1, 6):
        step(training)
        save(manager, training, number)
    files = fake_hub.revisions[fake_hub.head[0]]
    names = {name.split("/")[1] for name in files if "/checkpoint-" in name}
    assert names == {f"checkpoint-{number:08d}" for number in (1, 3, 4, 5)}
    assert len(fake_hub.revisions) == 1
    assert json.loads(files["fp16/latest.json"])["checkpoint"] == "checkpoint-00000005"
