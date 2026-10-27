from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_span_lora_scoring import Characters


def hybrid_reference():
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(23)
    cfg = Qwen3_5TextConfig(
        vocab_size=130,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [1, 1, 2],
            "partial_rotary_factor": 1.0,
        },
    )
    cfg._attn_implementation = "sdpa"
    decoder = Qwen3_5TextModel(cfg).eval()
    head = torch.nn.Linear(32, 130, bias=False).eval()
    return SimpleNamespace(
        decoder=decoder, head=head, model=torch.nn.ModuleDict({"decoder": decoder, "head": head})
    )


def turns():
    text = "History:Alpha.Beta.TTObservation:Gamma.Delta.UU"
    a, b = text.index("Alpha"), text.index("TT")
    c, d = text.index("Gamma"), text.index("UU")
    return [
        dict(
            text=text,
            spans=[[a, a + 6], [a + 6, b]],
            action_chars=[b, b + 2],
            action_ids=Characters().encode("TT"),
        ),
        dict(
            text=text,
            spans=[[c, c + 6], [c + 6, d]],
            action_chars=[d, d + 2],
            action_ids=Characters().encode("UU"),
        ),
    ]


def test_cached_hybrid_branches_match_full_and_reuse_unablated_history():
    from workflow_distillation.lora_fp16.span_cached_scoring import PrefixScorer
    from workflow_distillation.lora_fp16.span_scoring import score_turn

    state = hybrid_reference()
    scorer = PrefixScorer(state, Characters(), prefill_chunk_tokens=5)
    for turn in turns():
        expected = score_turn(state, Characters(), turn, chunk_size=2)
        actual = scorer.score_turn(turn, chunk_size=2)
        np.testing.assert_allclose(actual["ablated_logps"], expected["ablated_logps"], atol=2e-5)
        np.testing.assert_allclose(actual["deltas"], expected["deltas"], atol=2e-5)
        np.testing.assert_allclose(actual["weights"], expected["weights"], atol=1e-5)
        assert actual["full_logp"] == pytest.approx(expected["full_logp"], abs=2e-5)
    assert scorer.stats["prefix_resets"] == 1
    assert scorer.stats["reused_prefix_tokens"] > 0
    # Compare with a fresh cache: previous ablated branches must not contaminate history.
    fresh = PrefixScorer(state, Characters(), prefill_chunk_tokens=5).score_turn(turns()[1])
    np.testing.assert_allclose(actual["deltas"], fresh["deltas"], atol=2e-5)


def test_singleton_skips_forwards_but_rejects_changed_action_tokens():
    from workflow_distillation.lora_fp16.span_cached_scoring import PrefixScorer

    scorer = PrefixScorer(hybrid_reference(), Characters())
    turn = turns()[0]
    turn["spans"] = [[turn["spans"][0][0], turn["spans"][-1][1]]]
    result = scorer.score_turn(turn)
    assert result["weights"] == [1.0]
    assert result["diagnostics"] == "singleton_constant_weight_no_forwards"
    assert scorer.stats["forward_tokens"] == 0
    with pytest.raises(ValueError, match="target"):
        scorer.score_turn(dict(turn, action_ids=[1, 2]))


def test_partial_variant_resume_skips_completed_forward_and_preserves_scores():
    from workflow_distillation.lora_fp16.span_cached_scoring import PrefixScorer

    state = hybrid_reference()
    scorer = PrefixScorer(state, Characters())
    callbacks = []
    expected = scorer.score_turn(turns()[0])
    actual = scorer.score_turn(
        turns()[0],
        completed={"0": expected["full_logp"]},
        on_variant=lambda i, v: callbacks.append(i),
    )
    assert callbacks == [1, 2]
    np.testing.assert_allclose(actual["deltas"], expected["deltas"], atol=2e-5)


def test_changed_prefix_resets_cache_instead_of_reusing_different_history():
    from workflow_distillation.lora_fp16.span_cached_scoring import PrefixScorer

    scorer = PrefixScorer(hybrid_reference(), Characters())
    turn = turns()[0]
    scorer.score_turn(turn)
    changed = dict(turn, text="X" + turn["text"][1:])
    actual = scorer.score_turn(changed)
    fresh = PrefixScorer(scorer.state, Characters()).score_turn(changed)
    np.testing.assert_allclose(actual["deltas"], fresh["deltas"], atol=2e-5)
    assert scorer.stats["prefix_resets"] == 2


def test_cached_identity_is_separate_and_reference_reuse_is_strict(tmp_path):
    from dataclasses import replace

    from workflow_distillation.lora_fp16.config import RunConfig
    from workflow_distillation.lora_fp16.span_cache import ScoreStore
    from workflow_distillation.lora_fp16.span_cached_scoring import (
        reference_identity,
        reference_source,
    )
    from workflow_distillation.lora_fp16.span_scoring import reference_identity as old_identity

    cfg = RunConfig()
    cached = replace(cfg, span_scoring_backend="prefix-cache")
    assert reference_identity(cfg, "prov") == old_identity(cfg, "prov")
    assert reference_identity(cached, "prov") != old_identity(cfg, "prov")
    path = tmp_path / "old"
    source = ScoreStore(path, "corpus", old_identity(cfg, "prov"))
    source.write_reference(0, [1, 2], [0, 1], [-1])
    reused = reference_source(path, cached, "corpus", "prov")
    np.testing.assert_array_equal(reused.read_reference(0, [1, 2], [0, 1]), [-1])
    with pytest.raises(ValueError, match="identity"):
        reference_source(path, replace(cached, base_dtype="bfloat16"), "corpus", "prov")
    with pytest.raises(FileNotFoundError):
        reference_source(tmp_path / "missing", cached, "corpus", "prov")
