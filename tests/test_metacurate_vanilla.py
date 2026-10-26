import copy
import json

import pytest
import torch
import torch.nn.functional as F
from test_training_checkpoints import fake_hub  # noqa: F401


def vanilla_config(**kwargs):
    from workflow_distillation.lora_fp16.config import RunConfig

    return RunConfig(
        base_dtype="bfloat16", loss_normalization="vanilla",
        group_weights=(0, 1, 1, 1, 1), **kwargs,
    )


def test_bf16_model_load_and_scaler_policy_leave_fp16_default_intact():
    from workflow_distillation.lora_fp16.config import RunConfig
    from workflow_distillation.lora_fp16.model import model_kwargs, scaler_enabled

    assert model_kwargs(vanilla_config())["dtype"] == "bfloat16"
    assert not scaler_enabled(vanilla_config())
    assert scaler_enabled(RunConfig())
    assert model_kwargs(RunConfig())["dtype"] == "float16"
    with pytest.raises(ValueError):
        RunConfig(base_dtype="float32")


def test_vanilla_accumulation_matches_plain_assistant_ce_and_gradients():
    from test_fp16_training import tiny_state

    from workflow_distillation.lora_fp16.engine import train_update

    actual, config = tiny_state(), vanilla_config()
    expected = copy.deepcopy(actual)
    records = [
        dict(input_ids=[1, 2, 3, 4, 5, 6][:n], groups=[0, 1, 1, 2, 3, 4][:n],
             denominator=99999, trajectory_id=str(n))
        for n in (6, 4)
    ]
    losses = []
    for r in records:
        ids, groups = torch.tensor([r["input_ids"]]), torch.tensor([r["groups"]])
        logits = expected.head(expected.decoder(input_ids=ids).last_hidden_state)
        labels = ids.masked_fill(groups == 0, -100)
        losses.append(F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:]))
    loss = sum(losses) / 2
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters, 1.0)
    expected.optimizer.step()
    metrics = train_update(actual, records, config, torch.device("cpu"))
    assert metrics["loss"] == pytest.approx(loss.item(), rel=1e-6)
    for a, b in zip(actual.parameters, expected.parameters, strict=True):
        torch.testing.assert_close(a, b)


def test_complete_corpus_has_no_holdout_and_rejects_corrupt_length(tmp_path):
    from workflow_distillation.epoch_data import sha256_file
    from workflow_distillation.lora_fp16.metacurate import load_corpus, write_corpus

    records = [
        dict(trajectory_id="a", task_key="same", input_ids=[1, 2, 3, 4],
             groups=[0, 1, 3, 4], resolved=True, full_token_count=4),
        dict(trajectory_id="b", task_key="same", input_ids=[2, 3, 4, 5],
             groups=[0, 1, 3, 4], resolved=True, full_token_count=4),
    ]
    target = tmp_path / "tokens"
    write_corpus(records, target, {"source": "fixture"})
    receipt = dict(index_sha256=sha256_file(target / "index.json"), selected_ids=["a", "b"])
    corpus, split = load_corpus(target / "index.json", receipt)
    assert split["train_ids"] == ["a", "b"]
    assert split["validation_ids"] == split["excluded_sibling_ids"] == []
    assert corpus.record(1)["denominator"] == 3
    assert corpus.record(0)["input_ids"].tolist() == [1, 2, 3, 4]
    manifest = json.loads((target / "index.json").read_text())
    manifest["data"][0]["full_token_count"] = 5
    (target / "index.json").write_text(json.dumps(manifest))
    receipt["index_sha256"] = sha256_file(target / "index.json")
    with pytest.raises(ValueError, match="complete"):
        load_corpus(target / "index.json", receipt)


@pytest.mark.parametrize("change", [
    {"resolved": False}, {"full_token_count": 3},
    {"input_ids": [1] * 36001, "groups": [0] + [1] * 36000, "full_token_count": 36001},
])
def test_corpus_never_truncates_or_accepts_unsuccessful_records(tmp_path, change):
    from workflow_distillation.lora_fp16.metacurate import write_corpus

    record = dict(trajectory_id="a", task_key="t", input_ids=[1, 2], groups=[0, 1],
                  resolved=True, full_token_count=2)
    with pytest.raises(ValueError):
        write_corpus([dict(record, **change)], tmp_path / "tokens", {})


def test_all_1180_trajectories_are_consumed_twice_and_epoch_saves_are_unconditional():
    from workflow_distillation.lora_fp16.engine import run_loop, schedule

    batches = schedule(1180, epochs=2, accumulation=4, seed=42)
    assert len(batches) == 590
    assert sorted(sum(batches[:295], [])) == list(range(1180))
    assert sorted(sum(batches[295:], [])) == list(range(1180))
    saves = []
    result = run_loop(
        batches, steps_per_epoch=295,
        progress=dict(global_step=0, next_batch=0, samples_consumed=0),
        update=lambda b, n: {}, evaluate=lambda epoch: None,
        save=lambda p: saves.append(p["global_step"]), emit=lambda event: None,
        save_every=20, should_stop=lambda: False,
    )
    assert result["samples_consumed"] == 2360
    assert 295 in saves and 590 in saves and 20 in saves


def test_latest_one_retention_pins_both_epoch_ends(tmp_path):
    from test_fp16_training import tiny_state

    from workflow_distillation.lora_fp16.checkpoints import CheckpointManager

    state = tiny_state()
    manager = CheckpointManager(tmp_path, {}, keep_last=1, pinned_steps=[295, 590])
    for step in (280, 295, 300, 580, 590):
        manager.save(model=state.model, optimizer=state.optimizer, scaler=state.scaler,
                     scheduler=None, progress={"global_step": step})
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-00000295", "checkpoint-00000590",
    ]


def test_single_recovery_hub_policy_preserves_all_pinned_epochs(tmp_path, fake_hub):  # noqa: F811
    from test_training_checkpoints import save, setup_training, step

    from workflow_distillation.lora_fp16.checkpoints import CheckpointManager, HuggingFacePublisher

    training = setup_training()
    publisher = HuggingFacePublisher(
        "owner/metacurate-only", prefix="vanilla", keep_last=1, pinned_steps=[1, 3],
    )
    manager = CheckpointManager(tmp_path / "run", {}, publisher=publisher,
                                keep_last=1, pinned_steps=[1, 3])
    for number in range(1, 6):
        step(training)
        save(manager, training, number)
    files = fake_hub.revisions[fake_hub.head[0]]
    assert {n.split("/")[1] for n in files if "/checkpoint-" in n} == {
        "checkpoint-00000001", "checkpoint-00000003", "checkpoint-00000005",
    }
