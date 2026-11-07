"""Reuse a complete historical scoring backup; no GPU, API or new scoring."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.metacurate import load_corpus
from workflow_distillation.lora_fp16.span_cache import NAMES, ScoreStore, atomic_json, sample_hash
from workflow_distillation.lora_fp16.span_step_mask import project_cached_record

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--published-sidecar", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/processed/metacurate1180_vanilla"
    )
    parser.add_argument(
        "--masked", type=Path, default=ROOT / "data/processed/metacurate1180_stepmask_v1"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/processed/metacurate1180_span_stepmask_scores_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh score output; never overwrite historical scores")
    frozen = json.loads(args.published_sidecar.read_text())
    source_receipt = dict(
        index_sha256=sha256_file(args.source / "index.json"),
        selected_ids=[r["trajectory_id"] for r in frozen["data"]],
    )
    original, _ = load_corpus(args.source / "index.json", source_receipt)
    masked, _ = load_corpus(
        args.masked / "index.json", json.loads((args.masked / "selection.json").read_text())
    )
    if (
        frozen.get("complete") is not True
        or frozen["corpus_sha256"] != source_receipt["index_sha256"]
        or len(original) != len(masked)
    ):
        raise ValueError("Published sidecar/source corpus differs or incomplete")
    original_hashes = {n: hashlib.sha256() for n in NAMES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".span-stepmask-", dir=args.output.parent) as tmp:
        stage = Path(tmp) / "scores"
        store = ScoreStore(stage, sha256_file(args.masked / "index.json"), frozen["reference"])
        old_targets = new_targets = 0
        with tarfile.open(args.backup, "r:gz") as archive:
            identity = json.load(archive.extractfile("identity.json"))
            if identity != {k: frozen[k] for k in ("corpus_sha256", "reference")}:
                raise ValueError("Backup identity differs from published historical sidecar")
            for i in range(len(original)):
                old, new = original.record(i), masked.record(i)
                member = archive.extractfile(f"parts/{i:06d}.complete.npz")
                with np.load(io.BytesIO(member.read()), allow_pickle=False) as data:
                    if str(data["sample_hash"]) != sample_hash(old["input_ids"], old["groups"]):
                        raise ValueError(f"Historical sample alignment differs at {i}")
                    arrays = {n: data[n].copy() for n in NAMES}
                row = frozen["data"][i]
                if (
                    row["trajectory_id"] != old["trajectory_id"]
                    or row["offset"] != old_targets
                    or row["count"] != len(arrays["ref_logp"])
                    or row["sample_hash"] != sample_hash(old["input_ids"], old["groups"])
                ):
                    raise ValueError(f"Published sidecar row differs at {i}")
                selected = project_cached_record(old, new, arrays)
                for name in NAMES:
                    original_hashes[name].update(np.asarray(arrays[name], dtype="<f4").tobytes())
                old_targets += len(arrays["ref_logp"])
                new_targets += len(selected["ref_logp"])
                store.write_record(
                    i, new["input_ids"], new["groups"], *(selected[n] for n in NAMES)
                )
                if (i + 1) % 100 == 0:
                    print(json.dumps(dict(event="projected", records=i + 1)), flush=True)
        if old_targets != frozen["tokens"] or any(
            original_hashes[n].hexdigest() != frozen["file_sha256"][f"{n}.bin"] for n in NAMES
        ):
            raise ValueError("Reconstructed original scoring arrays differ from trained run")
        manifest = store.finalize(masked.record(i) for i in range(len(masked)))
        derivation = dict(
            method="strict-target-subset-v1",
            source_corpus_sha256=source_receipt["index_sha256"],
            source_sidecar_sha256=sha256_file(args.published_sidecar),
            source_array_sha256=frozen["file_sha256"],
            backup_sha256=sha256_file(args.backup),
            annotations_sha256=sha256_file(args.masked / "annotations.json"),
            original_targets=old_targets,
            retained_targets=new_targets,
            excluded_targets=old_targets - new_targets,
            input_tokens_unchanged=True,
            retained_values_identical=True,
            rescored=False,
            anchor_masked=True,
        )
        manifest["derivation"] = derivation
        atomic_json(stage / "index.json", manifest)
        atomic_json(stage / "summary.json", dict(complete=True, records=len(masked), **derivation))
        shutil.rmtree(stage / "parts")
        stage.rename(args.output)
    print(json.dumps(dict(output=str(args.output), **derivation), indent=2), flush=True)


if __name__ == "__main__":
    main()
