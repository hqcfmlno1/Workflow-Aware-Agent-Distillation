"""Numerical gradient checks; run in the remote PyTorch environment."""

import pytest

torch = pytest.importorskip("torch")

from workflow_distillation.weighted_sft_loss import (  # noqa: E402
    chunked_hidden_loss,
    weighted_logits_loss,
)


def test_weighted_ce_uses_shifted_targets_and_weighted_token_count():
    logits = torch.zeros((1, 5, 3), requires_grad=True)
    labels = torch.tensor([[-100, 0, 1, 2, -100]])
    groups = torch.tensor([[0, 1, 3, 4, 0]])
    loss = weighted_logits_loss(logits, labels, groups)
    assert loss.item() == pytest.approx(1.0986122886681098)
    loss.backward()
    # Target 0 with weight 1; target 1 with weight 1.5; target 2 with weight 2.
    assert logits.grad[0, 0, 0].item() == pytest.approx(-2 / 13.5)
    assert logits.grad[0, 1, 1].item() == pytest.approx(-1 / 4.5)
    assert logits.grad[0, 2, 2].item() == pytest.approx(-4 / 13.5)
    assert logits.grad[0, 3:].abs().sum().item() == 0


def test_chunks_preserve_full_trajectory_loss_and_gradients():
    torch.manual_seed(42)
    head = torch.nn.Linear(4, 5, bias=False)
    head.requires_grad_(False)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    labels = torch.tensor([[-100, 0, 1, -100, 2, 3, 4]])
    groups = torch.tensor([[0, 1, 2, 0, 3, 4, 1]])
    reference = weighted_logits_loss(head(hidden), labels, groups)
    reference_grad = torch.autograd.grad(reference, hidden)[0]
    actual = chunked_hidden_loss(
        hidden, head, labels, groups, denominator=6.25, scale=1.0, chunk_size=2
    )
    actual_grad = torch.autograd.grad(actual, hidden)[0]
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_grad, reference_grad)


def test_missing_groups_and_unequal_trajectory_lengths_use_trajectory_mean():
    logits = torch.zeros((2, 4, 2), requires_grad=True)
    labels = torch.tensor([[-100, 0, -100, -100], [-100, 1, 1, 1]])
    groups = torch.tensor([[0, 3, 0, 0], [0, 1, 1, 4]])
    result = weighted_logits_loss(logits, labels, groups)
    result.backward()
    assert logits.grad[0, 0, 0].item() == pytest.approx(-0.25)
    assert logits.grad[1, 0, 1].item() == pytest.approx(-0.0625)
    assert logits.grad[1, 2, 1].item() == pytest.approx(-0.125)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a remote GPU")
def test_chunked_loss_moves_targets_and_states_to_a_separate_head_device():
    # A split decoder returns hidden states away from its tied embedding/head.
    # Forgetting either direction of the copy breaks the loss or its gradient.
    torch.manual_seed(17)
    head = torch.nn.Linear(4, 5, bias=False).cuda().requires_grad_(False)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    labels = torch.tensor([[-100, 0, 1, -100, 2, 3, 4]])
    groups = torch.tensor([[0, 1, 2, 0, 3, 4, 1]])
    reference = weighted_logits_loss(head(hidden.cuda()), labels.cuda(), groups.cuda())
    expected_grad = torch.autograd.grad(reference, hidden)[0]
    actual = chunked_hidden_loss(hidden, head, labels, groups, 6.25, 1.0, chunk_size=2)
    actual_grad = torch.autograd.grad(actual, hidden)[0]
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two remote GPUs")
def test_partition_checkpoint_preserves_dropout_rng_on_destination_gpu():
    from workflow_distillation.optimized_qlora import checkpoint_across_devices

    # Accelerate moves inputs inside the layer forward, after checkpoint has
    # inspected argument devices. Native checkpoint would save only GPU 0 RNG.
    hidden = torch.randn(4, 8, device="cuda:0", requires_grad=True)

    def dispatched_layer(states):
        return torch.nn.functional.dropout(states.to("cuda:1"), p=0.5, training=True)

    torch.cuda.manual_seed_all(42)
    reference = dispatched_layer(hidden)
    expected_output = reference.detach()
    expected_grad = torch.autograd.grad(reference.square().sum(), hidden)[0]
    del reference
    torch.cuda.manual_seed_all(42)
    actual = checkpoint_across_devices(dispatched_layer, hidden)
    actual_grad = torch.autograd.grad(actual.square().sum(), hidden)[0]
    torch.testing.assert_close(actual, expected_output)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_delta_rule_blocking_preserves_full_history_and_all_input_gradients(device):
    from contextlib import nullcontext

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a remote GPU")
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    from workflow_distillation.optimized_qlora import blocked_delta_rule

    torch.manual_seed(28)
    dtype = torch.float16 if device == "cuda" else torch.float32
    tensors = [torch.randn(1, 137, 2, 8, device=device, dtype=dtype, requires_grad=True)
               for _ in range(3)]
    decay = torch.full((1, 137, 2), -0.1, device=device, requires_grad=True)
    beta = torch.full((1, 137, 2), 0.5, device=device, requires_grad=True)
    initial = torch.randn(1, 2, 8, 8, device=device, requires_grad=True)
    inputs = (*tensors, decay, beta, initial)
    options = {"output_final_state": True, "use_qk_l2norm_in_kernel": True}
    def autocast():
        return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()

    with autocast():
        expected, expected_state = torch_chunk_gated_delta_rule(
            *tensors, decay, beta, initial_state=initial, **options,
        )
    expected_grads = torch.autograd.grad(
        expected.square().sum() + expected_state.square().sum(), inputs,
    )
    def layer(q, k, v, g, b, state):
        return blocked_delta_rule(torch_chunk_gated_delta_rule, q, k, v, g, b,
                                  initial_state=state, block_size=64, **options)

    with autocast(), torch.autograd.graph.save_on_cpu(pin_memory=device == "cuda"):
        actual, actual_state = checkpoint(layer, *inputs, use_reentrant=False)
    actual_grads = torch.autograd.grad(actual.square().sum() + actual_state.square().sum(), inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_repeated_kv_attention_matches_native_gqa_outputs_and_gradients():
    from types import SimpleNamespace

    from workflow_distillation.t4_attention import repeated_kv_sdpa

    torch.manual_seed(7)
    q = torch.randn(1, 4, 7, 8, requires_grad=True)
    k = torch.randn(1, 2, 7, 8, requires_grad=True)
    v = torch.randn(1, 2, 7, 8, requires_grad=True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True, scale=0.3
    )
    actual, _ = repeated_kv_sdpa(SimpleNamespace(is_causal=True), q, k, v, None, scaling=0.3)
    torch.testing.assert_close(actual.transpose(1, 2), expected)
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v), retain_graph=True)
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
