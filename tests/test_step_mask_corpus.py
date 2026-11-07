"""Whole-step masking must preserve context and all unselected target labels."""

import numpy as np
import pytest
from transformers import AutoTokenizer


def test_mask_selected_step_including_calls_but_preserve_other_step_and_inputs():
    from test_weighted_sft_data import record

    from workflow_distillation.step_mask_corpus import apply_step_mask
    from workflow_distillation.step_mask_token_provenance import annotate_record as annotate_owned
    from workflow_distillation.weighted_sft_data import annotate_record

    tokenizer = AutoTokenizer.from_pretrained("data/raw/qwen35_4b_tokenizer", local_files_only=True)
    source = record(tokenizer)
    baseline = annotate_record(source, tokenizer)
    owned = annotate_owned(source, tokenizer, include_message_indices=True)
    result = apply_step_mask(owned, [2])
    np.testing.assert_array_equal(result["input_ids"], baseline["input_ids"])
    groups = np.asarray(result["groups"])
    owners = np.asarray(owned["token_message_indices"])
    old = np.asarray(baseline["groups"])
    assert np.any((owners == 2) & (old == 1))
    assert np.any((owners == 2) & (old == 3))
    assert not np.any(groups[owners == 2])
    np.testing.assert_array_equal(groups[owners != 2], old[owners != 2])
    assert np.any(groups[owners == 5] == 4)
    assert result["masked_target_tokens"] == int(np.sum((owners == 2) & (old > 0)))
    assert result["denominator"] == int(np.sum(groups > 0))
    with pytest.raises(ValueError):
        apply_step_mask(owned, [3])  # tool observation cannot be a mask target


def test_zero_remaining_targets_fail_instead_of_dividing_by_zero():
    from workflow_distillation.step_mask_corpus import apply_step_mask

    row = {
        "input_ids": [1, 2, 3],
        "groups": [0, 1, 3],
        "token_message_indices": [-1, 2, 2],
        "assistant_message_indices": [2],
    }
    with pytest.raises(ValueError, match="target"):
        apply_step_mask(row, [2])
