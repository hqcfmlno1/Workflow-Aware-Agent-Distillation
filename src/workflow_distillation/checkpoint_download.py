"""Read-only downloads of inference weights from an existing training repository.

No repository creation, uploads, commits, optimizer deserialization or GPU use.
One immutable HF commit is used for latest.json, manifest and adapter weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

DEFAULT_REPO = "hqcfmlno1/QwenStacy-4b-fp16-tqs060-reasoning200-run1"
DEFAULT_PREFIX = "fp16-tqs060-reasoning200"
CHECKPOINT = re.compile(r"checkpoint-\d{8,}$")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefix(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Invalid checkpoint prefix")
    return str(path)


def _manifest(path, checkpoint):
    manifest = json.loads(Path(path).read_text())
    step = manifest.get("progress", {}).get("global_step")
    checksum = manifest.get("files", {}).get("adapter.pt", "")
    if (
        manifest.get("format_version") != 1
        or manifest.get("complete") is not True
        or type(step) is not int
        or step != int(checkpoint.split("-")[1])
        or not isinstance(checksum, str)
        or not re.fullmatch("[a-f0-9]{64}", checksum)
        or not isinstance(manifest.get("identity"), dict)
    ):
        raise ValueError("Invalid/incomplete checkpoint manifest")
    return manifest


def resolve_checkpoint(
    repo=DEFAULT_REPO,
    *,
    prefix=DEFAULT_PREFIX,
    revision="main",
    checkpoint="latest",
    repo_type="model",
):
    """Resolve metadata only. Caller keeps this returned revision for the whole test."""
    from huggingface_hub import HfApi, hf_hub_download

    if repo_type not in ("model", "dataset"):
        raise ValueError("Expected model or dataset repo type")
    prefix = _prefix(prefix)
    commit = HfApi().repo_info(repo, repo_type=repo_type, revision=revision).sha
    if checkpoint == "latest":
        pointer = hf_hub_download(
            repo, f"{prefix}/latest.json", repo_type=repo_type, revision=commit
        )
        checkpoint = json.loads(Path(pointer).read_text())["checkpoint"]
    if not isinstance(checkpoint, str) or not CHECKPOINT.fullmatch(checkpoint):
        raise ValueError("Invalid checkpoint name")
    path = hf_hub_download(
        repo, f"{prefix}/{checkpoint}/manifest.json", repo_type=repo_type, revision=commit
    )
    manifest = _manifest(path, checkpoint)
    identity = manifest["identity"]
    return dict(
        repo=repo,
        repo_type=repo_type,
        prefix=prefix,
        revision=commit,
        checkpoint=checkpoint,
        progress=manifest["progress"],
        model=identity.get("model"),
        base_revision=identity.get("revision"),
        training_config=identity.get("settings", {}).get("config", {}),
        manifest_sha256=_sha256(path),
        adapter_sha256=manifest["files"]["adapter.pt"],
    )


def _verify(directory, snapshot):
    directory = Path(directory)
    for name, key in (("manifest.json", "manifest_sha256"), ("adapter.pt", "adapter_sha256")):
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != snapshot[key]:
            raise ValueError(f"{name} checksum mismatch")
    _manifest(directory / "manifest.json", snapshot["checkpoint"])


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download_checkpoint(
    repo=DEFAULT_REPO,
    destination="outputs/checkpoints/reasoning200",
    *,
    prefix=DEFAULT_PREFIX,
    revision="main",
    checkpoint="latest",
    repo_type="model",
):
    """Atomically store only manifest/adapter/provenance; reject conflicting caches."""
    import fcntl

    from huggingface_hub import hf_hub_download

    snapshot = resolve_checkpoint(
        repo, prefix=prefix, revision=revision, checkpoint=checkpoint, repo_type=repo_type
    )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / snapshot["checkpoint"]
    with (destination / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_symlink():
            raise ValueError("Checkpoint target must not be a symlink")
        if target.exists():
            _verify(target, snapshot)
            origin = json.loads((target / "source.json").read_text())
            if any(
                origin.get(k) != snapshot[k]
                for k in (
                    "repo",
                    "repo_type",
                    "prefix",
                    "checkpoint",
                    "manifest_sha256",
                    "adapter_sha256",
                )
            ) or origin.get("directory") != str(target):
                raise ValueError("Cached checkpoint source differs; choose another destination")
            # Reuse verified bytes, but take provenance/config from the current
            # immutable Hub manifest, never from an editable local receipt.
            result = dict(snapshot, directory=str(target))
            _write_json(target / "source.json", result)
        else:
            with tempfile.TemporaryDirectory(
                prefix=".checkpoint-download-", dir=destination
            ) as tmp:
                stage = Path(tmp) / snapshot["checkpoint"]
                stage.mkdir()
                for filename in ("manifest.json", "adapter.pt"):
                    cached = hf_hub_download(
                        repo,
                        f"{snapshot['prefix']}/{snapshot['checkpoint']}/{filename}",
                        repo_type=repo_type,
                        revision=snapshot["revision"],
                    )
                    shutil.copyfile(cached, stage / filename)
                _verify(stage, snapshot)
                result = dict(snapshot, directory=str(target))
                _write_json(stage / "source.json", result)
                stage.rename(target)
        _write_json(destination / "latest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoints/reasoning200"))
    parser.add_argument(
        "--metadata-only", action="store_true", help="No adapter weights downloaded"
    )
    args = parser.parse_args(argv)
    kwargs = dict(
        prefix=args.prefix,
        revision=args.revision,
        checkpoint=args.checkpoint,
        repo_type=args.repo_type,
    )
    result = (
        resolve_checkpoint(args.repo, **kwargs)
        if args.metadata_only
        else download_checkpoint(args.repo, args.output, **kwargs)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
