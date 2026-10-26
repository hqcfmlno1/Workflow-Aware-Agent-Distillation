import json
from pathlib import Path

import pytest


def test_fp16_model_arguments_are_unquantized_and_vision_is_cpu():
    from workflow_distillation.lora_fp16.config import RunConfig
    from workflow_distillation.lora_fp16.model import model_kwargs

    kwargs = model_kwargs(RunConfig())
    assert kwargs["dtype"] == "float16"
    assert "quantization_config" not in kwargs
    assert kwargs["device_map"]["model.visual"] == "cpu"
    assert kwargs["device_map"]["model.language_model.layers.31"] == 0
    assert kwargs["attn_implementation"] == "sdpa"
    for values in [
        dict(base_dtype="float32"),
        dict(quantization="nf4"),
        dict(epochs=0),
        dict(accumulation=0),
        dict(learning_rate=float("nan")),
    ]:
        with pytest.raises(ValueError):
            RunConfig(**values)


def test_committed_selection_is_disjoint_and_contains_all_original_candidates():
    from workflow_distillation.lora_fp16.data import select_split

    receipt = json.loads(Path("configs/training/gpus/rtx5090_32gb/selection.json").read_text())
    rows = receipt["candidates"]
    assert len(rows) == 1140
    assert min(r["tqs_score"] for r in rows) >= 0.6
    assert max(r["total_tokens"] for r in rows) == 29949
    assert receipt["split"] == select_split(rows)
    assert len(receipt["split"]["train_ids"]) == 1112
    assert len(receipt["split"]["validation_ids"]) == 20
