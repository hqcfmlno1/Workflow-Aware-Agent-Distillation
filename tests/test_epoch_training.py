"""One epoch and interrupted continuation must supervise each real sample once."""

import pytest

from workflow_distillation import epoch_training
from workflow_distillation.epoch_data import epoch_batches


def test_cursor_rejects_replaying_or_omitting_consumed_examples():
    batches = epoch_batches(5)
    with pytest.raises(ValueError, match="cursor"):
        epoch_training.validate_progress(
            batches, {"global_step": 1, "next_batch": 1, "samples_consumed": 1},
        )


def test_first_update_is_saved_and_resume_finishes_odd_epoch_without_replay(tmp_path):
    from test_training_checkpoints import load, setup_training, step

    from workflow_distillation.training_checkpoints import CheckpointManager

    training = setup_training()
    manager = CheckpointManager(tmp_path, {}, keep_last=1)
    batches = epoch_batches(7)
    visited, losses = [], []
    last_saved = []

    def update(batch, _number):
        visited.extend(batch)
        losses.append(step(training))
        return {"loss": float(losses[-1])}

    def save(progress):
        model, optimizer, scheduler, scaler = training
        if manager.due(progress["global_step"], force=True):
            manager.save(model=model, optimizer=optimizer, scheduler=scheduler,
                         scaler=scaler, progress=progress)
            last_saved.append(progress["global_step"])

    result = epoch_training.run_epoch_loop(
        batches, progress=epoch_training.initial_progress(), step=update, save=save,
        stop=lambda: len(losses) == 1, emit=lambda _event: None, save_every=5,
    )
    assert result["status"] == "stopped_checkpointed"
    assert last_saved == [1]
    assert result["progress"]["samples_consumed"] == 2
    # Compute the uninterrupted suffix, then reconstruct all training state.
    expected = [step(training) for _ in batches[1:]]
    training = setup_training()
    restored = load(manager, training)
    result = epoch_training.run_epoch_loop(
        batches, progress=restored, step=update, save=save, stop=lambda: False,
        emit=lambda _event: None, save_every=5,
    )
    assert result["status"] == "completed"
    assert result["progress"] == {"global_step": 4, "next_batch": 4, "samples_consumed": 7}
    assert sorted(visited) == list(range(7))
    assert len(set(visited)) == 7
    for actual, reference in zip(losses[1:], expected, strict=True):
        assert actual.equal(reference)
    assert last_saved == [1, 4]


def test_completed_resume_does_not_run_or_overwrite_a_checkpoint():
    def unexpected(*_args):
        raise AssertionError("Completed epoch must not replay updates or save again")

    result = epoch_training.run_epoch_loop(
        epoch_batches(1), progress={"global_step": 1, "next_batch": 1, "samples_consumed": 1},
        step=unexpected, save=unexpected, stop=unexpected, emit=lambda _event: None, save_every=5,
    )
    assert result["status"] == "completed"


def test_logged_process_records_both_output_streams_and_propagates_failure(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "first run.log"
    with pytest.raises(subprocess.CalledProcessError):
        epoch_training.run_logged_process(
            [sys.executable, "-u", "-c",
             "import sys; print('forward'); print('stderr', file=sys.stderr); sys.exit(3)"],
            path, timeout=10,
        )
    assert path.read_text().splitlines() == ["forward", "stderr"]


def test_log_write_failure_stops_training_process_instead_of_silently_losing_logs(
    tmp_path, monkeypatch,
):
    import sys
    from pathlib import Path

    log = tmp_path / "first run.log"
    original = Path.open

    class FullDisk:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def write(self, _line):
            raise OSError("log disk full")

    def open_file(path, *args, **kwargs):
        return FullDisk() if path == log else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError, match="log disk full"):
        epoch_training.run_logged_process(
            [sys.executable, "-u", "-c", "import time; print('forward'); time.sleep(60)"],
            log, timeout=5,
        )
