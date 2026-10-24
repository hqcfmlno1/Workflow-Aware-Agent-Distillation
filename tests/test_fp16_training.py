import copy

import pytest
import torch


def tiny_state():
    from types import SimpleNamespace

    torch.manual_seed(3)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(11, 4)

        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(last_hidden_state=self.embed(input_ids))

    decoder = Decoder()
    head = torch.nn.Linear(4, 11, bias=False).requires_grad_(False)
    model = torch.nn.ModuleDict({"decoder": decoder, "head": head})
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    return SimpleNamespace(
        model=model,
        decoder=decoder,
        head=head,
        parameters=list(decoder.parameters()),
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
    )


def records():
    return [
        dict(
            input_ids=[1, 2, 3, 4][:n],
            groups=[0, 1, 2, 3][:n],
            denominator=sum([0, 1, 0.75, 1.5][:n]),
            full_token_count=n,
            trajectory_id=str(n),
        )
        for n in (4, 3, 2)
    ]


def test_accumulation_matches_uniform_trajectory_mean_including_short_batch():
    from workflow_distillation.lora_fp16.config import RunConfig
    from workflow_distillation.lora_fp16.engine import train_update
    from workflow_distillation.weighted_sft_loss import weighted_logits_loss

    a = tiny_state()
    b = copy.deepcopy(a)
    losses = []
    for r in records():
        ids = torch.tensor([r["input_ids"]])
        g = torch.tensor([r["groups"]])
        h = b.decoder(input_ids=ids).last_hidden_state
        losses.append(weighted_logits_loss(b.head(h), ids.masked_fill(g == 0, -100), g))
    loss = sum(losses) / len(losses)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(b.parameters, 1.0)
    b.optimizer.step()
    b.optimizer.zero_grad(set_to_none=True)
    result = train_update(a, records(), RunConfig(), torch.device("cpu"))
    assert result["loss"] == pytest.approx(float(loss.detach()), rel=1e-6)
    assert result["grad_norm"] == pytest.approx(float(norm), rel=1e-6)
    for x, y in zip(a.parameters, b.parameters, strict=True):
        torch.testing.assert_close(x, y)


def test_validation_restores_training_mode_rng_and_does_not_update_weights():
    from workflow_distillation.lora_fp16.config import RunConfig
    from workflow_distillation.lora_fp16.engine import evaluate

    state = tiny_state()
    state.model.train()
    before = copy.deepcopy(state.model.state_dict())
    rng = torch.get_rng_state().clone()
    result = evaluate(state, records(), RunConfig(), torch.device("cpu"))
    assert result["trajectories"] == 3 and result["loss"] > 0
    assert state.model.training and state.decoder.training
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        torch.testing.assert_close(value, state.model.state_dict()[key])
    assert all(p.grad is None for p in state.parameters)


def test_schedule_never_accumulates_across_epochs_and_cursor_is_strict():
    from workflow_distillation.lora_fp16.engine import schedule, validate_cursor

    batches = schedule(5, epochs=2, accumulation=4, seed=42)
    assert [len(b) for b in batches] == [4, 1, 4, 1]
    assert sorted(sum(batches[:2], [])) == list(range(5))
    assert sorted(sum(batches[2:], [])) == list(range(5))
    validate_cursor(batches, dict(global_step=2, next_batch=2, samples_consumed=5))
    with pytest.raises(ValueError):
        validate_cursor(batches, dict(global_step=2, next_batch=2, samples_consumed=6))


def test_loop_revalidates_epoch_when_resuming_at_boundary_and_retains_final_save():
    from workflow_distillation.lora_fp16.engine import run_loop

    batches = [[0, 1], [2], [1, 0], [2]]
    validations = []
    saves = []
    result = run_loop(
        batches,
        steps_per_epoch=2,
        progress=dict(global_step=2, next_batch=2, samples_consumed=3),
        update=lambda b, n: {"loss": 1.0},
        evaluate=lambda epoch: validations.append(epoch),
        save=lambda p: saves.append(p.copy()),
        emit=lambda x: None,
        save_every=10,
        should_stop=lambda: False,
    )
    assert validations == [1, 2]
    assert result["global_step"] == 4 and result["samples_consumed"] == 6
    assert saves[-1]["global_step"] == 4


def test_epoch_checkpoints_survive_recovery_pruning_and_resume_optimizer(tmp_path):
    from workflow_distillation.lora_fp16.checkpoints import CheckpointManager

    state = tiny_state()
    manager = CheckpointManager(tmp_path, {"experiment": "fp16"}, keep_last=2, pinned_steps=[2, 4])
    for step in range(1, 7):
        manager.save(
            model=state.model,
            optimizer=state.optimizer,
            scaler=state.scaler,
            scheduler=None,
            progress={"global_step": step},
        )
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-00000002",
        "checkpoint-00000004",
        "checkpoint-00000005",
        "checkpoint-00000006",
    ]
    restored = tiny_state()
    result = manager.load(
        "latest",
        model=restored.model,
        optimizer=restored.optimizer,
        scaler=restored.scaler,
        scheduler=None,
    )
    assert result["global_step"] == 6
    for x, y in zip(state.parameters, restored.parameters, strict=True):
        torch.testing.assert_close(x, y)


def test_interrupted_resume_matches_uninterrupted_dropout_adam_scheduler_and_rng(tmp_path):
    from workflow_distillation.lora_fp16.cli import make_scheduler
    from workflow_distillation.lora_fp16.config import RunConfig
    from workflow_distillation.lora_fp16.engine import train_update
    from workflow_distillation.training_checkpoints import CheckpointManager

    def noisy_output(module, arguments, output):
        output.last_hidden_state = torch.nn.functional.dropout(
            output.last_hidden_state, p=0.2, training=module.training
        )
        return output

    config = RunConfig()
    state = tiny_state()
    state.decoder.register_forward_hook(noisy_output)
    scheduler = make_scheduler(state.optimizer, config, 4)
    batches = [records()[:2], records()[2:], records()[:2], records()[2:]]
    manager = CheckpointManager(tmp_path, {"experiment": "stateful"})
    for batch in batches[:2]:
        train_update(state, batch, config, torch.device("cpu"))
        scheduler.step()
    manager.save(
        model=state.model,
        optimizer=state.optimizer,
        scaler=state.scaler,
        scheduler=scheduler,
        progress=dict(global_step=2, next_batch=2, samples_consumed=3),
    )
    expected = []
    for batch in batches[2:]:
        expected.append(train_update(state, batch, config, torch.device("cpu"))["loss"])
        scheduler.step()
    rng = torch.get_rng_state().clone()
    restored = tiny_state()
    restored.decoder.register_forward_hook(noisy_output)
    other_scheduler = make_scheduler(restored.optimizer, config, 4)
    manager.load(
        "latest",
        model=restored.model,
        optimizer=restored.optimizer,
        scaler=restored.scaler,
        scheduler=other_scheduler,
    )
    actual = []
    for batch in batches[2:]:
        actual.append(train_update(restored, batch, config, torch.device("cpu"))["loss"])
        other_scheduler.step()
    assert actual == pytest.approx(expected, abs=0, rel=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert scheduler.state_dict() == other_scheduler.state_dict()
    for x, y in zip(state.parameters, restored.parameters, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
