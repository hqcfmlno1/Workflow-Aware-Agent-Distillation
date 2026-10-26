import copy

import numpy as np
import pytest
from test_weighted_sft_data import record, tokenizer  # noqa: F401


def api():
    from workflow_distillation.lora_fp16 import span_data

    return span_data


def test_segmentation_preserves_text_and_caps_shortest_adjacent_pairs():
    m = api()
    text = "".join(f"This is a sufficiently long sentence number {i}.\n" for i in range(35))
    spans = m.segment_reasoning(text)
    assert len(spans) == 24
    assert "".join(text[a:b] for a, b in spans) == text
    assert m.segment_reasoning("short") == [(0, 5)]


def test_ranks_deadzone_negative_scores_and_single_span_follow_upstream():
    m = api()
    assert m.span_weights([0.0]) == [1.0]
    assert m.span_weights([2.0, 2.0]) == pytest.approx([0.55, 0.55])
    assert m.span_weights([-2.0, -1.0, -1.0, 1.0]) == pytest.approx([0.1, 0.55, 0.55, 1])
    assert m.span_weights([0, 0.01]) == pytest.approx([0.46, 0.64])
    with pytest.raises(ValueError):
        m.span_weights([float("nan")])


def test_exact_render_multi_call_submit_and_independent_anchor(tokenizer):  # noqa: F811
    from workflow_distillation.weighted_sft_data import annotate_record

    m = api()
    source = record(tokenizer)
    anno = annotate_record(source, tokenizer)
    out = m.prepare_record(
        source["messages"], source["tools"], tokenizer, anno["input_ids"], anno["groups"]
    )
    assert out["text"] == source["text"]
    assert len(out["turns"]) == 2
    first, last = out["turns"]
    a, b = first["action_tokens"]
    assert "cat src.py" in tokenizer.decode(anno["input_ids"][a:b])
    assert first["call_count"] == 2
    assert last["submit"] is True
    assert out["omitted_reasoning_messages"] == []
    groups = np.array(anno["groups"])
    gamma = np.array(out["anchor_weights"])
    assert np.all(gamma[groups == 0] == 0)
    assert np.all(gamma[groups == 1] == 1)
    assert set(gamma[groups == 4]) == {1e-6, 1}
    for turn in out["turns"]:
        a, b = turn["action_tokens"]
        left = turn["action_chars"][0]
        assert (
            tokenizer.encode(out["text"][:left], add_special_tokens=False) == anno["input_ids"][:a]
        )
        assert turn["action_ids"] == anno["input_ids"][a:b]


def test_template_omission_and_changed_ids_fail_closed(tokenizer):  # noqa: F811
    from scripts.data.prepare_action_focused_sft import serialize
    from workflow_distillation.weighted_sft_data import annotate_record

    m = api()
    source = record(tokenizer)
    source["messages"].insert(5, {"role": "user", "content": "Continue."})
    source.update(serialize(source["messages"], source["tools"], tokenizer))
    anno = annotate_record(source, tokenizer)
    out = m.prepare_record(
        source["messages"], source["tools"], tokenizer, anno["input_ids"], anno["groups"]
    )
    assert out["omitted_reasoning_messages"] == [2]
    assert out["turns"][0]["spans"] == []
    ids = copy.copy(anno["input_ids"])
    ids[-1] += 1
    with pytest.raises(ValueError, match="token"):
        m.prepare_record(source["messages"], source["tools"], tokenizer, ids, anno["groups"])


def test_character_overlap_weights_do_not_modify_protocol():
    m = api()
    assert m.token_weights(
        [(0, 2), (2, 6), (6, 8), (8, 8)], [(0, 4), (4, 8)], [0.1, 1.0]
    ) == pytest.approx([0.1, 0.55, 1, 1])
    assert m.token_weights([(0, 4)], [(1, 3)], [0.1]) == pytest.approx([0.55])
