import json
from pathlib import Path

import pytest


def test_asset_receipt_matches_selection_and_all_local_payloads():
    from workflow_distillation.lora_fp16.assets import verify_assets

    receipt = json.loads(Path("configs/training/gpus/rtx5090_32gb/data.json").read_text())
    selection = json.loads(Path("configs/training/gpus/rtx5090_32gb/selection.json").read_text())
    assert receipt["repo_type"] == "dataset"
    assert len(receipt["revision"]) == 40
    assert receipt["files"]["index.json"] == selection["index_sha256"]
    # Repository tests also run without the ignored local corpus.
    local = Path("data/processed/tqs1140_holdout20_fp16")
    if local.is_dir():
        verify_assets(local, receipt)


def test_asset_verification_rejects_partial_download(tmp_path):
    from workflow_distillation.lora_fp16.assets import verify_assets

    (tmp_path / "index.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_assets(tmp_path, {"files": {"index.json": "wrong"}})


def test_download_cache_string_is_verified_before_atomic_copy(tmp_path):
    from workflow_distillation.epoch_data import sha256_file
    from workflow_distillation.lora_fp16.assets import copy_verified

    cached = tmp_path / "cached"
    cached.write_bytes(b"pinned tokens")
    target = tmp_path / "target"
    copy_verified(str(cached), target, sha256_file(cached))
    assert target.read_bytes() == b"pinned tokens"
    with pytest.raises(ValueError, match="checksum"):
        copy_verified(str(cached), target, "bad")
    assert target.read_bytes() == b"pinned tokens"
