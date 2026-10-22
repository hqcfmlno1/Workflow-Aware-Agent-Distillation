"""The GPU execution adapter must preserve causal attention and gradients."""

import copy
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("masked", [False, True])
def test_attention_preserves_causal_outputs_and_input_gradients(masked):
    from workflow_distillation.gpu_profiles.rtx4080_32gb.runtime import flash_sdpa
    from workflow_distillation.t4_attention import repeated_kv_sdpa

    torch.manual_seed(19)
    tensors = [torch.randn(2, heads, 9, 8, requires_grad=True) for heads in (4, 2, 2)]
    mask = torch.ones(9, 9, dtype=torch.bool).tril() if masked else None
    module = SimpleNamespace(is_causal=True)
    expected, _ = repeated_kv_sdpa(module, *tensors, mask, scaling=0.3)
    expected_grad = torch.autograd.grad(expected.square().sum(), tensors)
    actual, _ = flash_sdpa(module, *tensors, mask, scaling=0.3)
    actual_grad = torch.autograd.grad(actual.square().sum(), tensors)
    torch.testing.assert_close(actual, expected)
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize("mode", ["cpu", "gpu", "hybrid"])
def test_activation_storage_preserves_backward_and_counts_saved_bytes(mode):
    from workflow_distillation.gpu_profiles.rtx4080_32gb.runtime import ActivationStorage

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    storage = ActivationStorage(mode, gpu_budget_bytes=12)
    with storage:
        y = x.square().sum()
    y.backward()
    torch.testing.assert_close(x.grad, torch.tensor([2.0, 4.0, 6.0]))
    assert storage.saved_bytes == 12


def test_unknown_storage_mode_is_rejected():
    from workflow_distillation.gpu_profiles.rtx4080_32gb.runtime import ActivationStorage

    with pytest.raises(ValueError):
        ActivationStorage("automatic")


def test_execution_transition_records_runtime_without_changing_optimizer_or_data():
    from workflow_distillation.gpu_profiles.rtx4080_32gb.runtime import execution_identity

    base = {"script_sha256": "old", "records_sha256": "data", "settings": {
        "config": {"learning_rate": 5e-5, "rank": 32, "global_batch_size": 4,
                   "attention": "t4_repeated_kv", "saved_inputs": "CPU pinned"}}}
    parent = {"complete": True, "world_size": 2, "identity": copy.deepcopy(base),
              "progress": {"global_step": 1028, "next_batch": 1028, "samples_consumed": 2692}}
    profile = {"attention": "flash_repeated_kv", "activation_storage": "cpu",
               "gpu_budget_bytes": 0}
    result, binding = execution_identity(parent, "parenthash", base, profile,
                                         launcher_sha="new", runtime_sha="runtime")
    assert result["records_sha256"] == "data"
    assert result["settings"]["config"] == {
        "learning_rate": 5e-5, "rank": 32, "global_batch_size": 4,
        "attention": "flash_repeated_kv", "saved_inputs": "cpu"}
    assert result["settings"]["gpu_execution"]["parent_progress"]["global_step"] == 1028
    assert result["settings"]["gpu_execution"]["runtime_sha256"] == "runtime"
    assert binding["parent"]["identity"] == base
    assert parent["identity"] == base


def test_execution_transition_refuses_a_parent_from_another_training_identity():
    from workflow_distillation.gpu_profiles.rtx4080_32gb.runtime import execution_identity

    with pytest.raises(ValueError):
        execution_identity({"complete": True, "world_size": 2, "identity": {"wrong": 1}},
                           "hash", {}, {"attention": "flash_repeated_kv",
                           "activation_storage": "cpu", "gpu_budget_bytes": 0},
                           launcher_sha="new", runtime_sha="runtime")


def test_storage_only_transition_preserves_the_existing_attention_kernel():
    from workflow_distillation.gpu_profiles.rtx4080_32gb.runtime import execution_identity

    base = {"settings": {"config": {"attention": "t4_repeated_kv", "saved_inputs": "CPU"}}}
    parent = {"complete": True, "world_size": 2, "identity": base,
              "progress": {"global_step": 1027}}
    result, _ = execution_identity(
        parent, "parent", base, {"attention": "t4_repeated_kv",
        "activation_storage": "hybrid", "gpu_budget_bytes": 2147483648},
        launcher_sha="new", runtime_sha="runtime",
    )
    assert result["settings"]["config"]["attention"] == "t4_repeated_kv"
    assert result["settings"]["config"]["saved_inputs"] == "hybrid"


def test_installed_storage_preserves_the_cache_cleanup_context_class_contract(monkeypatch):
    from test_host_memory_recovery import load_recovery

    from workflow_distillation.gpu_profiles.rtx4080_32gb.runtime import install_execution

    original = torch.autograd.graph.save_on_cpu
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", original)
    install_execution({"attention": "t4_repeated_kv", "activation_storage": "hybrid",
                       "gpu_budget_bytes": 4096})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: None, raising=False)
    events = []
    # Exercise the real recovery integration, including its type-level wrapping.
    load_recovery().install_cache_cleanup(torch, events.append)
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        loss = x.square().sum()
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor([4.0, 6.0]))
    assert [event["phase"] for event in events] == ["before_forward"]
