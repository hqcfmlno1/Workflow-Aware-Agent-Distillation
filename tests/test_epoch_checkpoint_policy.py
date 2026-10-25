import random
from pathlib import Path

import numpy as np
import torch
from test_training_checkpoints import fake_hub, setup_training  # noqa: F401


def test_epoch_only_saves_and_uploads_both_epoch_ends(tmp_path, fake_hub):  # noqa: F811
    from scripts.training.train_epoch_checkpoints import epoch_loop
    from workflow_distillation.lora_fp16.checkpoints import CheckpointManager, HuggingFacePublisher

    state = setup_training()
    manager = CheckpointManager(tmp_path / "checkpoints", {"original_identity": 20},
                                keep_last=1, pinned_steps=[295, 590])
    publisher = HuggingFacePublisher("owner/epochs", prefix="run", keep_last=1,
                                     pinned_steps=[295, 590])

    def save(position):
        if manager.due(position["global_step"], force=True):
            manager.save(model=state[0], optimizer=state[1], scheduler=state[2], scaler=state[3],
                         progress=position)

    # Use the real checkpoint format/publisher; save_every=20 is deliberately
    # passed by the frozen original config and must be overridden at the IO layer.
    loop = epoch_loop(publisher, tmp_path)
    loop([[0, 1, 2, 3]] * 590, steps_per_epoch=295,
         progress=dict(global_step=0, next_batch=0, samples_consumed=0),
         update=lambda b, n: {}, evaluate=lambda epoch: None, save=save,
         emit=lambda e: None, save_every=20, should_stop=lambda: False)
    names = {p.name for p in (tmp_path / "checkpoints").glob("checkpoint-*")}
    assert names == {"checkpoint-00000295", "checkpoint-00000590"}
    files = fake_hub.revisions[fake_hub.head[0]]
    assert {n.split("/")[1] for n in files if "/checkpoint-" in n} == names


def test_graceful_mid_epoch_stop_saves_locally_without_upload(tmp_path, fake_hub):  # noqa: F811
    from scripts.training.train_epoch_checkpoints import epoch_loop
    from workflow_distillation.lora_fp16.checkpoints import HuggingFacePublisher

    publisher = HuggingFacePublisher("owner/epochs", prefix="run", keep_last=1, pinned_steps=[3, 6])
    updates, local = [], []

    def update(batch, number):
        updates.append(number)
        return {}

    epoch_loop(publisher, Path(tmp_path))(
        [[0]] * 6, steps_per_epoch=3,
        progress=dict(global_step=0, next_batch=0, samples_consumed=0),
        update=update, evaluate=lambda epoch: None,
        save=lambda position: local.append(position.copy()), emit=lambda e: None,
        save_every=20, should_stop=lambda: len(updates) >= 2,
    )
    assert local == [dict(global_step=2, next_batch=2, samples_consumed=2)]
    assert fake_hub.revisions[fake_hub.head[0]] == {}


def test_epoch_upload_restores_training_rng_even_when_network_raises():
    import pytest

    from scripts.training.train_epoch_checkpoints import publish_preserving_rng

    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

    def failing_upload(path):
        random.random()
        np.random.random()
        torch.rand(5)
        raise ConnectionError("network unavailable")

    with pytest.raises(ConnectionError):
        publish_preserving_rng(failing_upload, Path("checkpoint-00000295"))
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_resume_epoch_boundary_retries_publication_before_another_update(tmp_path, fake_hub):  # noqa: F811
    from scripts.training.train_epoch_checkpoints import epoch_loop
    from workflow_distillation.lora_fp16.checkpoints import CheckpointManager, HuggingFacePublisher

    state = setup_training()
    manager = CheckpointManager(tmp_path / "checkpoints", {}, pinned_steps=[1, 2])
    manager.save(model=state[0], optimizer=state[1], scheduler=state[2], scaler=state[3],
                 progress=dict(global_step=1, next_batch=1, samples_consumed=1))
    publisher = HuggingFacePublisher("owner/retry", prefix="run", keep_last=1, pinned_steps=[1, 2])

    def update(batch, number):
        assert "run/checkpoint-00000001/manifest.json" in fake_hub.revisions[fake_hub.head[0]]
        return {}

    def save(position):
        if manager.due(position["global_step"], force=True):
            manager.save(model=state[0], optimizer=state[1], scheduler=state[2], scaler=state[3],
                         progress=position)

    epoch_loop(publisher, tmp_path)(
        [[0], [0]], steps_per_epoch=1,
        progress=dict(global_step=1, next_batch=1, samples_consumed=1),
        update=update, evaluate=lambda epoch: None, save=save, emit=lambda e: None,
        save_every=20, should_stop=lambda: False,
    )
