"""Protect read-only, immutable, atomic inference-checkpoint downloads."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from workflow_distillation.checkpoint_download import download_checkpoint, resolve_checkpoint


def hub_fixture(tmp_path, monkeypatch):
    import huggingface_hub

    weights = b"verified-adapter"
    name = "checkpoint-00000180"
    manifest = dict(
        format_version=1,
        complete=True,
        world_size=1,
        progress=dict(global_step=180, samples_consumed=720, next_batch=180),
        identity=dict(
            model="Qwen/Qwen3.5-4B",
            revision="base-revision",
            settings=dict(config=dict(base_dtype="float16", quantization="none")),
        ),
        files={
            "adapter.pt": hashlib.sha256(weights).hexdigest(),
            "rank-00000.pt": "unused-optimizer-checksum",
        },
    )
    files = {
        "run/latest.json": json.dumps({"checkpoint": name}).encode(),
        f"run/{name}/manifest.json": json.dumps(manifest).encode(),
        f"run/{name}/adapter.pt": weights,
    }
    fetched = []

    def info(repo, **kwargs):
        assert repo == "owner/train"
        assert kwargs == {"repo_type": "model", "revision": "main"}
        return SimpleNamespace(sha="a" * 40)

    def fetch(repo, filename, **kwargs):
        assert repo == "owner/train"
        assert kwargs == {"repo_type": "model", "revision": "a" * 40}
        fetched.append(filename)
        target = tmp_path / "cache" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[filename])
        return str(target)

    # Deliberately exposes no create_repo/upload/create_commit methods.
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(repo_info=info))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    return manifest, files, fetched


def test_latest_download_pins_source_and_never_downloads_optimizer(tmp_path, monkeypatch):
    _, _, fetched = hub_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    result = download_checkpoint("owner/train", out, prefix="run")
    folder = out / "checkpoint-00000180"
    assert result["directory"] == str(folder.resolve())
    assert result["repo"] == "owner/train" and result["revision"] == "a" * 40
    assert (folder / "adapter.pt").read_bytes() == b"verified-adapter"
    assert result["progress"]["global_step"] == 180
    assert all("rank-" not in name for name in fetched)
    assert set(p.name for p in folder.iterdir()) == {"manifest.json", "adapter.pt", "source.json"}
    first = (folder / "source.json").read_bytes()
    again = download_checkpoint("owner/train", out, prefix="run")
    assert again == result
    assert (folder / "source.json").read_bytes() == first


def test_resolve_fetches_only_metadata_and_keeps_real_training_repo(tmp_path, monkeypatch):
    _, _, fetched = hub_fixture(tmp_path, monkeypatch)
    result = resolve_checkpoint("owner/train", prefix="run")
    assert result["repo"] == "owner/train"
    assert result["checkpoint"] == "checkpoint-00000180"
    assert fetched == ["run/latest.json", "run/checkpoint-00000180/manifest.json"]


@pytest.mark.parametrize("bad", ["../../escape", "/absolute", "checkpoint-bad"])
def test_rejects_unsafe_pointer_before_writing_output(tmp_path, monkeypatch, bad):
    _, files, _ = hub_fixture(tmp_path, monkeypatch)
    files["run/latest.json"] = json.dumps({"checkpoint": bad}).encode()
    with pytest.raises(ValueError, match="checkpoint"):
        download_checkpoint("owner/train", tmp_path / "out", prefix="run")
    assert not (tmp_path / "out/checkpoint-00000180").exists()


def test_corrupt_download_is_never_published_or_overwrites_valid_cache(tmp_path, monkeypatch):
    _, files, _ = hub_fixture(tmp_path, monkeypatch)
    files["run/checkpoint-00000180/adapter.pt"] = b"bad"
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="checksum"):
        download_checkpoint("owner/train", out, prefix="run")
    assert not (out / "checkpoint-00000180").exists()
    assert not (out / "latest.json").exists()


def test_refuses_same_step_from_another_source(tmp_path, monkeypatch):
    hub_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    result = download_checkpoint("owner/train", out, prefix="run")
    origin = out / "checkpoint-00000180/source.json"
    changed = json.loads(origin.read_text())
    changed["repo"] = "owner/other-run"
    origin.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="source"):
        download_checkpoint("owner/train", out, prefix="run")
    assert (out / "checkpoint-00000180/adapter.pt").read_bytes() == b"verified-adapter"
    assert json.loads((out / "latest.json").read_text()) == result


def test_cache_reuse_refreshes_pinned_revision_and_trusted_metadata(tmp_path, monkeypatch):
    from workflow_distillation import checkpoint_download as module

    hub_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    first = download_checkpoint("owner/train", out, prefix="run")
    current = {k: v for k, v in first.items() if k != "directory"}
    current["revision"] = "b" * 40
    monkeypatch.setattr(module, "resolve_checkpoint", lambda *a, **kw: current)
    origin = out / "checkpoint-00000180/source.json"
    edited = json.loads(origin.read_text())
    edited["training_config"]["quantization"] = "nf4"
    origin.write_text(json.dumps(edited))
    result = download_checkpoint("owner/train", out, prefix="run")
    assert result["revision"] == "b" * 40
    assert result["training_config"]["quantization"] == "none"
    assert json.loads(origin.read_text()) == result
    assert json.loads((out / "latest.json").read_text()) == result
